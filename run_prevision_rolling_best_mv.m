% function [kf_est_returns, kf_cum_returns, metrics, kf_next_year_weights] = run_prevision_rolling_best_mv( ...
%                     Prices_returns, portfolio_index, years, num_low_esg, ...
%                     esg_scores_aligned, short_bound_high, short_bound_low, threshold, flag_plot)
% 
%     % -------------------------------------------------------------------------
%     % Purpose:
%     %   Rolling min-var training + NEXT-YEAR (12m) forecasts via Kalman filter.
%     %   Returns final-horizon (next-year) estimated returns from KF weights,
%     %   their cumulative curve, and rich metrics including KF vs Real comparison.
%     % -------------------------------------------------------------------------
% 
%     % ===== Ensure the **second** column is the name column (NAME_COL=2) =====
%     NAME_COL = 2;
%     if numel(Prices_returns.Properties.VariableNames) >= NAME_COL
%         if ~strcmp(Prices_returns.Properties.VariableNames{NAME_COL}, 'CompanyCommonName')
%             Prices_returns.Properties.VariableNames{NAME_COL} = 'CompanyCommonName';
%         end
%     else
%         error('Prices_returns must have at least two columns (IDs + CompanyCommonName).');
%     end
% 
%     % Indices and window lengths
%     RET_START = 3;                  % first column with returns
%     TRAIN_M   = 60;                 % training window length (periods)
% 
%     t0   = RET_START;
%     tend = t0 + TRAIN_M - 1;
% 
%     nAssets = size(esg_scores_aligned,1);
%     Troll   = years * 12;
% 
%     if width(Prices_returns) < tend
%         error('Not enough return columns for the first training window (need at least %d).', tend);
%     end
% 
%     Minimum_Portfolios = zeros(nAssets, Troll);
% 
%     % =========================
%     % 1) Rolling min-var train
%     % =========================
%     for i = 1:Troll
%         train_cols = t0 : tend;
% 
%         % Table fed to your estimators: [ID, Name, ReturnCols]
%         Train_tbl  = Prices_returns(:, [1, NAME_COL, train_cols]);
% 
%         % Estimation on training data
%         [mean_returns, cov_matrix] = compute_mean_and_covariance(Train_tbl);
% 
%         [~, ~, Portfolio] = compute_best_in_class_frontier( ...
%             mean_returns, cov_matrix, esg_scores_aligned, threshold, ...
%             nAssets, num_low_esg(portfolio_index), short_bound_high, short_bound_low);
% 
%         % Frontier limits: col 1 is Minimum-Variance weights
%         [w_limits, ~] = estimateFrontierLimits(Portfolio);
%         w_minvar = w_limits(:,1);
% 
%         Minimum_Portfolios(:,i) = w_minvar;
% 
%         % slide the window
%         t0   = t0   + 1;
%         tend = tend + 1;
% 
%         if tend > width(Prices_returns)
%             error('Ran out of return columns while sliding the window at i=%d.', i);
%         end
%     end
% 
%     % Placeholders
%     metrics           = struct();
%     kf_est_returns    = [];
%     kf_cum_returns    = [];
%     HORIZON           = 12;
% 
%     % ==========================================
%     % 3) Kalman filter: forecast next 12 weights
%     % ==========================================
%     W   = Minimum_Portfolios;          % nAssets x Troll
%     phi = zeros(nAssets,1);
%     q   = zeros(nAssets,1);
% 
%     if Troll >= 3
%         Wt  = W(:,2:end);
%         Wtm = W(:,1:end-1);
%         for a = 1:nAssets
%             x_prev = Wtm(a,:)';    % w_{t-1}
%             x_curr = Wt(a,:)';     % w_t
%             denom = (x_prev'*x_prev);
%             if denom > 0
%                 phi_a = (x_prev'*x_curr) / denom;
%             else
%                 phi_a = 0;
%             end
%             phi(a) = min(max(phi_a, 0), 0.999); % keep dynamics reasonable
%             e = x_curr - phi(a)*x_prev;
%             q(a) = max(var(e, 1), 1e-6);       % process noise (population var with floor)
%         end
%     else
%         phi(:) = 0.9;
%         q(:)   = 1e-4;
%     end
% 
%     % State-space matrices
%     A = diag(phi);     % transition
%     Q = diag(q);       % process noise
% 
%     % Initialize at last observed min-var weights
%     x = W(:,end);      % state
% 
%     % State covariance: stationary approx for AR(1)
%     P = diag(q ./ max(1 - phi.^2, 1e-3));
%     P(~isfinite(P)) = 1e-2;
%     P = P + 1e-6*eye(nAssets);  % numerical stability
% 
%     % 12-step prediction (no measurement updates in the future)
%     kf_next_year_weights = zeros(nAssets, HORIZON);
% 
%     for h = 1:HORIZON
%         % Predict step
%         x = A * x;               % x_{t|t-1}
%         P = A * P * A' + Q;      % P_{t|t-1}
% 
%         % Enforce feasibility: box + simplex
%         x = min(max(x, short_bound_low - 1e-12), short_bound_high + 1e-12);
%         x = x / sum(x);
% 
%         kf_next_year_weights(:,h) = x;
%     end
% 
%     % ================================================
%     % 4) "Real" next-year weights (same logic as cycle)
%     %    and diagnostic comparison vs KF
%     % ================================================
%     real_next_year_weights = nan(nAssets, HORIZON);
%     t0_real   = t0;    % from where the previous loop left
%     tend_real = tend;
% 
%     h_eff = 0;
%     for h = 1:HORIZON
%         if tend_real > width(Prices_returns)
%             break; % not enough data to form another training window
%         end
% 
%         train_cols_h = t0_real : tend_real;
%         Train_tbl_h  = Prices_returns(:, [1, NAME_COL, train_cols_h]);
% 
%         [mean_returns_h, cov_matrix_h] = compute_mean_and_covariance(Train_tbl_h);
% 
%         [~, ~, Portfolio_h] = compute_best_in_class_frontier( ...
%             mean_returns_h, cov_matrix_h, esg_scores_aligned, threshold, ...
%             nAssets, num_low_esg(portfolio_index), short_bound_high, short_bound_low);
% 
%         [w_limits_h, ~] = estimateFrontierLimits(Portfolio_h);
%         w_minvar_real_h = w_limits_h(:,1);
% 
%         % Light cleanup only (Portfolio already enforces feasibility)
%         w_minvar_real_h = min(max(w_minvar_real_h, short_bound_low - 1e-12), short_bound_high + 1e-12);
%         w_minvar_real_h = w_minvar_real_h / sum(w_minvar_real_h);
% 
%         h_eff = h_eff + 1;
%         real_next_year_weights(:, h_eff) = w_minvar_real_h;
% 
%         % slide
%         t0_real   = t0_real   + 1;
%         tend_real = tend_real + 1;
%     end
% 
%     if h_eff == 0
%         metrics.kf_vs_real = struct( ...
%             'available_steps', 0, ...
%             'note', 'Not enough columns to compute real next-year weights after the rolling window.' ...
%         );
%         return;
%     end
% 
%     % Align horizons and compute weight diagnostics
%     K  = min(HORIZON, h_eff);
%     KF = kf_next_year_weights(:, 1:K);
%     RL = real_next_year_weights(:, 1:K);
% 
%     D  = KF - RL;                          % deviations per step (weights)
%     L1 = sum(abs(D), 1);                   % L1 per step
%     L2 = sqrt(sum(D.^2, 1));               % L2 per step
%     COS = zeros(1, K);
%     for j = 1:K
%         COS(j) = dot(KF(:,j), RL(:,j)) / (norm(KF(:,j)) * norm(RL(:,j)) + eps);
%     end
% 
%     per_asset_MAE = mean(abs(D), 2, 'omitnan');
% 
%     Step = (1:K).';
%     compare_tbl = table(Step, L1.', L2.', COS.', ...
%         'VariableNames', {'Step', 'L1_sumAbs', 'L2_norm', 'CosineSim'});
% 
%     metrics.real_next_year_weights = RL;
%     metrics.kf_vs_real = struct( ...
%         'available_steps', K, ...
%         'summary_per_step', compare_tbl, ...
%         'per_asset_MAE', per_asset_MAE, ...
%         'diff_matrix', D, ...
%         'kf_weights', KF, ...
%         'real_weights', RL ...
%     );
% 
%     % ==========================================================
%     % 5) FINAL-HORIZON (next-year) returns & METRICS: KF vs Real
%     %    Multiply each month’s asset return by the respective
%     %    horizon weight vector; store diagnostics, compute metrics
%     % ==========================================================
%     % columns for the next K months after 'tend'
%     next_cols = (tend + 1):(tend + K);
%     next_cols = next_cols(next_cols <= width(Prices_returns));
%     K_eff     = numel(next_cols);
% 
%     kf_est_returns   = nan(1, K_eff);
%     real_est_returns = nan(1, K_eff);
% 
%     for j = 1:K_eff
%         rj = Prices_returns{:, next_cols(j)}; rj = rj(:);
%         kf_est_returns(j)   = sum(KF(:, j) .* rj, 'omitnan');
%         real_est_returns(j) = sum(RL(:, j) .* rj, 'omitnan');
%     end
% 
%     kf_cum_returns = cumprod(1 + kf_est_returns(:));
% 
%     metrics.kf_vs_real.kf_est_returns   = kf_est_returns(:);
%     metrics.kf_vs_real.real_est_returns = real_est_returns(:);
% 
%     % ============================
%     % 5.a) FINAL 12-month metrics
%     % ============================
%     if K_eff > 0
%         frequency      = 12;      % monthly
%         risk_free_rate = 0;       % set if available
% 
%         % ---- KF metrics (final horizon) ----
%         kf_ann_ret  = annualized_return(kf_est_returns, frequency);
%         kf_ann_vol  = annualized_volatility(kf_est_returns, frequency);
%         kf_sharpe   = sharpe_ratio(kf_est_returns, frequency, risk_free_rate);
%         kf_mdd      = max_drawdown(kf_est_returns);
%         kf_calmar   = calmar_ratio(kf_est_returns, frequency);
% 
%         kf_table = table( ...
%             kf_ann_ret * 100, ...
%             kf_ann_vol * 100, ...
%             kf_sharpe, ...
%             kf_mdd * 100, ...
%             kf_calmar, ...
%             'VariableNames', { ...
%                 'AnnualizedReturn_Pct', ...
%                 'AnnualizedVolatility_Pct', ...
%                 'SharpeRatio', ...
%                 'MaxDrawdown_Pct', ...
%                 'CalmarRatio' ...
%             });
% 
%         % ---- REAL metrics (final horizon) ----
%         rl_ann_ret  = annualized_return(real_est_returns, frequency);
%         rl_ann_vol  = annualized_volatility(real_est_returns, frequency);
%         rl_sharpe   = sharpe_ratio(real_est_returns, frequency, risk_free_rate);
%         rl_mdd      = max_drawdown(real_est_returns);
%         rl_calmar   = calmar_ratio(real_est_returns, frequency);
% 
%         rl_table = table( ...
%             rl_ann_ret * 100, ...
%             rl_ann_vol * 100, ...
%             rl_sharpe, ...
%             rl_mdd * 100, ...
%             rl_calmar, ...
%             'VariableNames', { ...
%                 'AnnualizedReturn_Pct', ...
%                 'AnnualizedVolatility_Pct', ...
%                 'SharpeRatio', ...
%                 'MaxDrawdown_Pct', ...
%                 'CalmarRatio' ...
%             });
% 
%         % ---- RETURNS-LEVEL L1/L2 (KF vs REAL, final horizon) ----
%         ret_L1 = sum(abs(kf_est_returns(:) - real_est_returns(:)), 'omitnan');
%         ret_L2 = sqrt(sum((kf_est_returns(:) - real_est_returns(:)).^2, 'omitnan'));
% 
%         % ---- WEIGHTS-LEVEL AGGREGATED L1/L2/COS (from compare_tbl) ----
%         w_L1_mean = mean(L1, 'omitnan');
%         w_L1_max  = max(L1, [], 'omitnan');
%         w_L2_mean = mean(L2, 'omitnan');
%         w_L2_max  = max(L2, [], 'omitnan');
%         w_COS_mean = mean(COS, 'omitnan');
% 
%         % ---- Side-by-side comparison (KF vs REAL) + L1/L2 diagnostics ----
%         comp_table = table( ...
%             kf_table.AnnualizedReturn_Pct,  rl_table.AnnualizedReturn_Pct, ...
%             kf_table.AnnualizedVolatility_Pct, rl_table.AnnualizedVolatility_Pct, ...
%             kf_table.SharpeRatio,          rl_table.SharpeRatio, ...
%             kf_table.MaxDrawdown_Pct,      rl_table.MaxDrawdown_Pct, ...
%             kf_table.CalmarRatio,          rl_table.CalmarRatio, ...
%             ret_L1, ret_L2, ...
%             w_L1_mean, w_L1_max, w_L2_mean, w_L2_max, w_COS_mean, ...
%             'VariableNames', { ...
%                 'KF_AnnRet_%','Real_AnnRet_%', ...
%                 'KF_AnnVol_%','Real_AnnVol_%', ...
%                 'KF_Sharpe','Real_Sharpe', ...
%                 'KF_MDD_%','Real_MDD_%', ...
%                 'KF_Calmar','Real_Calmar', ...
%                 'Returns_L1','Returns_L2', ...
%                 'Weights_L1_mean','Weights_L1_max','Weights_L2_mean','Weights_L2_max','Weights_Cosine_mean' ...
%             });
% 
%         % Store in metrics
%         metrics.kf_vs_real.performance_kf_final    = kf_table;
%         metrics.kf_vs_real.performance_real_final  = rl_table;
%         metrics.kf_vs_real.performance_comp_final  = comp_table;
% 
%         % ============================
%         % 5.b) Final-horizon plots
%         % ============================
%         if exist('flag_plot','var') && flag_plot
%             % (i) Per-period returns (final 12 months)
%             fig3 = figure('Name', sprintf('Next-Year Est. Returns — Portfolio %d', portfolio_index));
%             plot(kf_est_returns(:), '-', 'LineWidth', 1.5, 'DisplayName', 'KF est. returns'); hold on;
%             plot(real_est_returns(:), '-', 'LineWidth', 1.5, 'DisplayName', 'Real est. returns');
%             grid on; xlabel('Horizon month'); ylabel('Return');
%             title(sprintf('Next-Year Estimated Returns (Weights × Returns) — Portfolio %d', portfolio_index));
%             legend('show', 'Location', 'best');
% 
%             % (ii) Cumulative wealth (final 12 months)
%             fig4 = figure('Name', sprintf('Next-Year Cumulative Wealth — Portfolio %d', portfolio_index));
%             plot(cumprod(1 + kf_est_returns(:)), '-', 'LineWidth', 2, 'DisplayName', 'KF cumulative'); hold on;
%             plot(cumprod(1 + real_est_returns(:)), '-', 'LineWidth', 2, 'DisplayName', 'Real cumulative');
%             grid on; xlabel('Horizon month'); ylabel('Cumulative Wealth');
%             title(sprintf('Next-Year Cumulative Wealth — KF vs Real — Portfolio %d', portfolio_index));
%             legend('show', 'Location', 'best');
% 
%             % Console summary
%             fprintf('\n--- FINAL 12-MONTH METRICS (KF) — Portfolio %d ---\n', portfolio_index);
%             disp(kf_table);
%             fprintf('\n--- FINAL 12-MONTH METRICS (REAL) — Portfolio %d ---\n', portfolio_index);
%             disp(rl_table);
%             fprintf('\n--- FINAL 12-MONTH METRICS (KF vs REAL + L1/L2) — Portfolio %d ---\n', portfolio_index);
%             disp(comp_table);
%         end
%     end
% end


function [kf_est_returns, kf_cum_returns, metrics, kf_next_year_weights] = run_prevision_rolling_best_mv( ...
                    Prices_returns, portfolio_index, years, num_low_esg, ...
                    esg_scores_aligned, short_bound_high, short_bound_low, threshold, flag_plot)

    % -------------------------------------------------------------------------
    % Purpose:
    %   Rolling min-var training + NEXT-YEAR (12m) forecasts via Kalman filter
    %   with monthly measurement updates from real min-var weights.
    %   Weight feasibility is enforced ONLY by simple clipping + sum-normalization.
    % -------------------------------------------------------------------------

    % ===== Ensure the **second** column is the name column (NAME_COL=2) =====
    NAME_COL = 2;
    if numel(Prices_returns.Properties.VariableNames) >= NAME_COL
        if ~strcmp(Prices_returns.Properties.VariableNames{NAME_COL}, 'CompanyCommonName')
            Prices_returns.Properties.VariableNames{NAME_COL} = 'CompanyCommonName';
        end
    else
        error('Prices_returns must have at least two columns (IDs + CompanyCommonName).');
    end

    % Indici e finestre
    RET_START = 3;                  % prima colonna con i returns
    TRAIN_M   = 60;                 % lunghezza finestra training (mesi)

    t0   = RET_START;
    tend = t0 + TRAIN_M - 1;

    nAssets = size(esg_scores_aligned,1);
    Troll   = years * 12;

    if width(Prices_returns) < tend
        error('Not enough return columns for the first training window (need at least %d).', tend);
    end

    Minimum_Portfolios = zeros(nAssets, Troll);

    % =========================
    % 1) Rolling min-var train
    % =========================
    for i = 1:Troll
        train_cols = t0 : tend;

        % Tabella per gli stimatori: [ID, Name, ReturnCols]
        Train_tbl  = Prices_returns(:, [1, NAME_COL, train_cols]);

        % Stima momenti
        [mean_returns, cov_matrix] = compute_mean_and_covariance(Train_tbl);

        % Frontier con vincoli ESG/box (il tuo metodo)
        [~, ~, Portfolio] = compute_best_in_class_frontier( ...
            mean_returns, cov_matrix, esg_scores_aligned, threshold, ...
            nAssets, num_low_esg(portfolio_index), short_bound_high, short_bound_low);

        % Colonna 1 = pesi Minimum-Variance
        [w_limits, ~] = estimateFrontierLimits(Portfolio);
        w_minvar = w_limits(:,1);

        % (solo clipping+normalizzazione per coerenza)
        w_minvar = min(max(w_minvar, short_bound_low - 1e-12), short_bound_high + 1e-12);
        w_minvar = w_minvar / sum(w_minvar);

        Minimum_Portfolios(:,i) = w_minvar;

        % slide della finestra
        t0   = t0   + 1;
        tend = tend + 1;

        if tend > width(Prices_returns)
            error('Ran out of return columns while sliding the window at i=%d.', i);
        end
    end

    % Placeholders
    metrics           = struct();
    kf_est_returns    = [];
    kf_cum_returns    = [];
    HORIZON           = 12;

    % ================================================
    % 2) Pesi "reali" del prossimo anno (per l'update)
    % ================================================
    real_next_year_weights = nan(nAssets, HORIZON);
    t0_real   = t0;
    tend_real = tend;

    h_eff = 0;
    for h = 1:HORIZON
        if tend_real > width(Prices_returns)
            break; % non ci sono abbastanza colonne per un'altra finestra
        end

        train_cols_h = t0_real : tend_real;
        Train_tbl_h  = Prices_returns(:, [1, NAME_COL, train_cols_h]);

        [mean_returns_h, cov_matrix_h] = compute_mean_and_covariance(Train_tbl_h);

        [~, ~, Portfolio_h] = compute_best_in_class_frontier( ...
            mean_returns_h, cov_matrix_h, esg_scores_aligned, threshold, ...
            nAssets, num_low_esg(portfolio_index), short_bound_high, short_bound_low);

        [w_limits_h, ~] = estimateFrontierLimits(Portfolio_h);
        w_minvar_real_h = w_limits_h(:,1);

        % Solo clipping + normalizzazione (no proiezioni complesse)
        w_minvar_real_h = min(max(w_minvar_real_h, short_bound_low - 1e-12), short_bound_high + 1e-12);
        w_minvar_real_h = w_minvar_real_h / sum(w_minvar_real_h);

        h_eff = h_eff + 1;
        real_next_year_weights(:, h_eff) = w_minvar_real_h;

        % slide
        t0_real   = t0_real   + 1;
        tend_real = tend_real + 1;
    end

    if h_eff == 0
        metrics.kf_vs_real = struct( ...
            'available_steps', 0, ...
            'note', 'Not enough columns to compute real next-year weights after the rolling window.' ...
        );
        return;
    end

    K  = min(HORIZON, h_eff);
    RL = real_next_year_weights(:, 1:K);   % osservazioni mese per mese (es. 2023)

    % ================================================
    % 3) Kalman filter con UPDATE (predict→update mensile)
    %    Modello per-asset AR(1) nello stato dei pesi
    % ================================================
    W   = Minimum_Portfolios;   % nAssets x Troll (storia pesi min-var fino a fine rolling)
    phi = zeros(nAssets,1);
    q   = zeros(nAssets,1);

    % Stima AR(1) per riga (asset)
    if size(W,2) >= 3
        Wt  = W(:,2:end);
        Wtm = W(:,1:end-1);
        for a = 1:nAssets
            x_prev = Wtm(a,:)'; x_curr = Wt(a,:)';
            denom  = (x_prev'*x_prev);
            phi_a  = (denom>0) * (x_prev'*x_curr)/max(denom,eps);
            phi(a) = min(max(phi_a,0),0.999);
            e      = x_curr - phi(a)*x_prev;
            q(a)   = max(var(e,1),1e-6);    % process noise
        end
    else
        phi(:) = 0.9; 
        q(:)   = 1e-4;
    end

    A = diag(phi);             % transizione
    Q = diag(q);               % rumore di processo
    H = eye(nAssets);          % osservazione diretta dei pesi
    R = 1e-4*eye(nAssets);     % rumore di misura (tunable)

    % Stato iniziale = ultimo peso min-var osservato
    x = W(:,end);

    % Covarianza iniziale: approx stazionaria AR(1)
    P = diag(q ./ max(1 - phi.^2, 1e-3));
    P(~isfinite(P)) = 1e-2;
    P = P + 1e-6*eye(nAssets);

    kf_next_year_weights = zeros(nAssets, K);
    I = eye(nAssets);

    for h = 1:K
        % --- Predict ---
        x_pred = A * x;
        P_pred = A * P * A' + Q;

        % --- Update con pesi reali del mese h ---
        y  = RL(:,h);
        S  = H * P_pred * H' + R;       % cov. innovazione
        Kt = P_pred * H' / S;           % Kalman gain
        x  = x_pred + Kt * (y - H*x_pred);
        P  = (I - Kt*H) * P_pred * (I - Kt*H)' + Kt*R*Kt'; % Joseph form

        % --- Enforce feasibility: box + simplex (SOLO clipping+normalizzazione) ---
        x = min(max(x, short_bound_low - 1e-12), short_bound_high + 1e-12);
        x = x / sum(x);

        kf_next_year_weights(:,h) = x;
    end

    % ==========================================================
    % 4) RETURNS & METRICS su orizzonte finale (K mesi)
    % ==========================================================
    next_cols = (tend + 1):(tend + K);
    next_cols = next_cols(next_cols <= width(Prices_returns));
    K_eff     = numel(next_cols);

    kf_est_returns   = nan(1, K_eff);
    real_est_returns = nan(1, K_eff);

    for j = 1:K_eff
        rj = Prices_returns{:, next_cols(j)}; rj = rj(:);
        kf_est_returns(j)   = sum(kf_next_year_weights(:, j) .* rj, 'omitnan');
        real_est_returns(j) = sum(RL(:, j) .* rj, 'omitnan');
    end

    kf_cum_returns = cumprod(1 + kf_est_returns(:));

    % Diagnostiche pesi
    D  = kf_next_year_weights(:,1:K_eff) - RL(:,1:K_eff);
    L1 = sum(abs(D), 1);
    L2 = sqrt(sum(D.^2, 1));
    COS = zeros(1, K_eff);
    for j = 1:K_eff
        COS(j) = dot(kf_next_year_weights(:,j), RL(:,j)) / (norm(kf_next_year_weights(:,j)) * norm(RL(:,j)) + eps);
    end
    per_asset_MAE = mean(abs(D), 2, 'omitnan');
    Step = (1:K_eff).';
    compare_tbl = table(Step, L1.', L2.', COS.', ...
        'VariableNames', {'Step', 'L1_sumAbs', 'L2_norm', 'CosineSim'});

    metrics.real_next_year_weights = RL(:,1:K_eff);
    metrics.kf_vs_real = struct( ...
        'available_steps', K_eff, ...
        'summary_per_step', compare_tbl, ...
        'per_asset_MAE', per_asset_MAE, ...
        'diff_matrix', D, ...
        'kf_weights', kf_next_year_weights(:,1:K_eff), ...
        'real_weights', RL(:,1:K_eff) ...
    );

    % ============================
    % 5) METRICS (final horizon)
    % ============================
    if K_eff > 0
        frequency      = 12;      % monthly
        risk_free_rate = 0;       % set if available

        % ---- KF metrics ----
        kf_ann_ret  = annualized_return(kf_est_returns, frequency);
        kf_ann_vol  = annualized_volatility(kf_est_returns, frequency);
        kf_sharpe   = sharpe_ratio(kf_est_returns, frequency, risk_free_rate);
        kf_mdd      = max_drawdown(kf_est_returns);
        kf_calmar   = calmar_ratio(kf_est_returns, frequency);

        kf_table = table( ...
            kf_ann_ret * 100, ...
            kf_ann_vol * 100, ...
            kf_sharpe, ...
            kf_mdd * 100, ...
            kf_calmar, ...
            'VariableNames', { ...
                'AnnualizedReturn_Pct', ...
                'AnnualizedVolatility_Pct', ...
                'SharpeRatio', ...
                'MaxDrawdown_Pct', ...
                'CalmarRatio' ...
            });

        % ---- REAL metrics ----
        rl_ann_ret  = annualized_return(real_est_returns, frequency);
        rl_ann_vol  = annualized_volatility(real_est_returns, frequency);
        rl_sharpe   = sharpe_ratio(real_est_returns, frequency, risk_free_rate);
        rl_mdd      = max_drawdown(real_est_returns);
        rl_calmar   = calmar_ratio(real_est_returns, frequency);

        rl_table = table( ...
            rl_ann_ret * 100, ...
            rl_ann_vol * 100, ...
            rl_sharpe, ...
            rl_mdd * 100, ...
            rl_calmar, ...
            'VariableNames', { ...
                'AnnualizedReturn_Pct', ...
                'AnnualizedVolatility_Pct', ...
                'SharpeRatio', ...
                'MaxDrawdown_Pct', ...
                'CalmarRatio' ...
            });

        % ---- RETURNS-LEVEL L1/L2 (KF vs REAL) ----
        ret_L1 = sum(abs(kf_est_returns(:) - real_est_returns(:)), 'omitnan');
        ret_L2 = sqrt(sum((kf_est_returns(:) - real_est_returns(:)).^2, 'omitnan'));

        % ---- WEIGHTS-LEVEL AGGREGATES ----
        w_L1_mean = mean(L1, 'omitnan');
        w_L1_max  = max(L1, [], 'omitnan');
        w_L2_mean = mean(L2, 'omitnan');
        w_L2_max  = max(L2, [], 'omitnan');
        w_COS_mean = mean(COS, 'omitnan');

        comp_table = table( ...
            kf_table.AnnualizedReturn_Pct,  rl_table.AnnualizedReturn_Pct, ...
            kf_table.AnnualizedVolatility_Pct, rl_table.AnnualizedVolatility_Pct, ...
            kf_table.SharpeRatio,          rl_table.SharpeRatio, ...
            kf_table.MaxDrawdown_Pct,      rl_table.MaxDrawdown_Pct, ...
            kf_table.CalmarRatio,          rl_table.CalmarRatio, ...
            ret_L1, ret_L2, ...
            w_L1_mean, w_L1_max, w_L2_mean, w_L2_max, w_COS_mean, ...
            'VariableNames', { ...
                'KF_AnnRet_%','Real_AnnRet_%', ...
                'KF_AnnVol_%','Real_AnnVol_%', ...
                'KF_Sharpe','Real_Sharpe', ...
                'KF_MDD_%','Real_MDD_%', ...
                'KF_Calmar','Real_Calmar', ...
                'Returns_L1','Returns_L2', ...
                'Weights_L1_mean','Weights_L1_max','Weights_L2_mean','Weights_L2_max','Weights_Cosine_mean' ...
            });

        metrics.kf_vs_real.performance_kf_final    = kf_table;
        metrics.kf_vs_real.performance_real_final  = rl_table;
        metrics.kf_vs_real.performance_comp_final  = comp_table;

        if exist('flag_plot','var') && flag_plot
            % (i) Per-period returns
            figure('Name', sprintf('Next-Year Est. Returns — Portfolio %d', portfolio_index));
            plot(kf_est_returns(:), '-', 'LineWidth', 1.5, 'DisplayName', 'KF est. returns'); hold on;
            plot(real_est_returns(:), '-', 'LineWidth', 1.5, 'DisplayName', 'Real est. returns');
            grid on; xlabel('Horizon month'); ylabel('Return');
            title(sprintf('Next-Year Estimated Returns (Weights × Returns) — Portfolio %d', portfolio_index));
            legend('show', 'Location', 'best');

            % (ii) Cumulative wealth
            figure('Name', sprintf('Next-Year Cumulative Wealth — Portfolio %d', portfolio_index));
            plot(cumprod(1 + kf_est_returns(:)), '-', 'LineWidth', 2, 'DisplayName', 'KF cumulative'); hold on;
            plot(cumprod(1 + real_est_returns(:)), '-', 'LineWidth', 2, 'DisplayName', 'Real cumulative');
            grid on; xlabel('Horizon month'); ylabel('Cumulative Wealth');
            title(sprintf('Next-Year Cumulative Wealth — KF vs Real — Portfolio %d', portfolio_index));
            legend('show', 'Location', 'best');

            % Console summary
            fprintf('\n--- FINAL 12-MONTH METRICS (KF) — Portfolio %d ---\n', portfolio_index);
            disp(kf_table);
            fprintf('\n--- FINAL 12-MONTH METRICS (REAL) — Portfolio %d ---\n', portfolio_index);
            disp(rl_table);
            fprintf('\n--- FINAL 12-MONTH METRICS (KF vs REAL + L1/L2) — Portfolio %d ---\n', portfolio_index);
            disp(comp_table);
        end
    end
end
