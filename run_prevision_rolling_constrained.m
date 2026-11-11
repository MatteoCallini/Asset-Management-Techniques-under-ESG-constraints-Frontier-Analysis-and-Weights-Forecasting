function [all_oos_returns, all_oos_cum, metrics] = run_prevision_rolling_constrained( ...
                    Prices_returns, portfolio_index, portfolio_type, years, thresholds, ...
                    esg_scores_aligned, flag, max_short)

% -------------------------------------------------------------------------
% Purpose:
%   Rolling backtest with ESG-constrained frontier. Fixes KF instability
%   under short selling via a constrained Kalman filter (predict+update),
%   gain damping, adaptive measurement noise, and convex projection to:
%       sum(w)=1, lb <= w <= ub, and sum(abs(w)) <= Lmax.
%
% Outputs (unchanged):
%   all_oos_returns : 1×T row vector of concatenated OOS returns (per window).
%   all_oos_cum     : 1×T cumulative wealth over OOS periods.
%   metrics         : struct with
%       .per_window            table of metrics per window (from run_performance)
%       .overall               overall metrics on concatenated OOS
%       .kf_filter             filtered weights history (W_filt) and options
%       .kf_forecast           next-year (12m) KF forecasted weights
%       .kf_vs_real (optional) diagnostics KF vs Real on next 12m:
%                              weights L1/L2/cosine, returns L1/L2,
%                              and performance tables (KF/Real/compare)
% -------------------------------------------------------------------------

    % ---------- Basic table hygiene ----------
    if ~any(strcmp(Prices_returns.Properties.VariableNames, 'CompanyCommonName'))
        Prices_returns.Properties.VariableNames{1} = 'CompanyCommonName';
    end

    % ---------- Indices and window parameters ----------
    NAME_COL  = 2;    % column containing asset identifiers
    RET_START = 3;    % first column containing returns
    TRAIN_M   = 60;   % training window length (periods)
    TEST_M    = 12;   % testing window length (periods)

    % Containers
    all_oos_returns   = [];
    per_window_metrics = cell(years, 1);
    per_window_labels  = strings(years, 1);

    % We'll also store the min-variance weights each training roll (for KF)
    nAssets = size(esg_scores_aligned,1);
    Minimum_Portfolios = zeros(nAssets, years);   % one column per roll

    % Rolling pointers
    t0 = RET_START;
    tend = t0 + TRAIN_M - 1;   % inclusive index of last train column

    for j = 1:years
        % ---------- Training slice ----------
        train_cols = t0 : tend;
        Train_tbl  = Prices_returns(:, [1, NAME_COL, train_cols]);

        % Estimation inputs
        [mean_returns, cov_matrix] = compute_mean_and_covariance(Train_tbl);

        % Frontier (signature depends on 'flag')
        if flag
            [~, ~, Portfolio] = compute_esg_constrained_frontier( ...
                                mean_returns, cov_matrix, esg_scores_aligned, ...
                                thresholds(portfolio_index), size(esg_scores_aligned, 1), max_short, size(esg_scores_aligned, 1), flag, max_short);
        else
            [~, ~, Portfolio] = compute_esg_constrained_frontier( ...
                                mean_returns, cov_matrix, esg_scores_aligned, ...
                                thresholds(portfolio_index), size(esg_scores_aligned, 1), max_short, size(esg_scores_aligned, 1));
        end

        % Grab the minimum-variance corner weights for this roll
        [w_limits, ~] = estimateFrontierLimits(Portfolio);
        w_minvar = w_limits(:,1);              % nAssets x 1
        Minimum_Portfolios(:, j) = w_minvar;   % store for KF

        % ---------- Testing slice ----------
        test_cols = (tend + 1) : (tend + TEST_M);   % strictly OOS
        if test_cols(end) > width(Prices_returns)
            % If we run out of columns, cut the last window short
            test_cols = test_cols(test_cols <= width(Prices_returns));
        end
        Test_tbl  = Prices_returns(:, [NAME_COL, test_cols]);

        % Evaluate OOS using the Portfolio from the current training window
        [oos_ret, ~, mt] = run_performance(Test_tbl, Portfolio, mean_returns, 1, portfolio_type, false);

        % Store
        all_oos_returns  = [all_oos_returns, oos_ret]; %#ok<AGROW>
        per_window_metrics{j} = mt;
        per_window_labels(j)  = sprintf('Window %d (train %d m, test %d m)', j, TRAIN_M, numel(test_cols));

        % Slide windows by TEST_M months
        t0   = t0   + TEST_M;
        tend = tend + TEST_M;

        % Guard against running past data end
        if tend >= width(Prices_returns)
            years = j;                         % truncate loops’ effective years
            Minimum_Portfolios = Minimum_Portfolios(:,1:years);
            per_window_metrics = per_window_metrics(1:years);
            per_window_labels  = per_window_labels(1:years);
            break;
        end
    end

    % ---------- Aggregate OOS performance ----------
    all_oos_returns = all_oos_returns(:)';                  % row
    all_oos_cum     = cumprod(1 + all_oos_returns);         % row

    % ---------- Per-window metrics table ----------
    if ~isempty(per_window_metrics) && ~isempty(per_window_metrics{1})
        T = per_window_metrics{1};
        varnames = T.Properties.VariableNames;

        per_window_mat = zeros(numel(per_window_metrics), numel(varnames));
        rows = 0;
        for k = 1:numel(per_window_metrics)
            if ~isempty(per_window_metrics{k})
                rows = rows + 1;
                per_window_mat(rows, :) = per_window_metrics{k}{1, :};
            end
        end
        per_window_mat = per_window_mat(1:rows, :);
        metrics.per_window = array2table(per_window_mat, 'VariableNames', varnames);
        metrics.per_window.Window = per_window_labels(1:rows);
        metrics.per_window = movevars(metrics.per_window, 'Window', 'Before', 1);
    else
        metrics.per_window = table();
    end

    % ---------- Overall OOS metrics ----------
    freq = 12; rf = 0;
    overall_ann_return = annualized_return(all_oos_returns, freq);
    overall_ann_vol    = annualized_volatility(all_oos_returns, freq);
    overall_sharpe     = sharpe_ratio(all_oos_returns, freq, rf);
    overall_mdd        = max_drawdown(all_oos_returns);
    overall_calmar     = calmar_ratio(all_oos_returns, freq);

    metrics.overall = table( ...
        overall_ann_return * 100, ...
        overall_ann_vol * 100, ...
        overall_sharpe, ...
        overall_mdd * 100, ...
        overall_calmar, ...
        'VariableNames', { ...
            'AnnualizedReturn_Pct', ...
            'AnnualizedVolatility_Pct', ...
            'SharpeRatio', ...
            'MaxDrawdown_Pct', ...
            'CalmarRatio' ...
        });

    % ---------- Plot aggregated OOS cumulative ----------
    fig = figure(gcf); hold on;
    lgd = findobj(fig, 'Type', 'Legend');
    if isempty(lgd)
        lgd = legend('show', 'Location', 'best');
        set(lgd, 'AutoUpdate', 'on');
    end
    plot(all_oos_cum, '-', 'LineWidth', 2, 'Color', rand(1,3), ...
        'DisplayName', sprintf('Portfolio %d', portfolio_index));

    % =====================================================================
    %                          CONSTRAINED KALMAN FILTER
    %      (predict+update on historical min-var weights, then 12m forecast)
    % =====================================================================
    W = Minimum_Portfolios;               % nAssets x years   (observed weights per roll)
    if size(W,2) >= 2
        % Bounds & gross leverage for shorting (tune as needed)
        lb   = -max_short * ones(nAssets,1);  % per-asset short limit
        ub   =  1 * ones(nAssets,1);          % per-asset long cap (adjust if desired)
        Lmax =  1 + 2*max_short;              % gross leverage cap (e.g., 1.2 if max_short=0.1)

        % Filter + Forecast (12 months)
        HORIZON = 12;
        [W_filt, kf_next_year_weights] = kf_weights_constrained( ...
            W, lb, ub, Lmax, HORIZON, ...
            'DampPhi',0.98, 'GainDamp',0.85, 'RLevel',1e-4, 'RAdaptive',true, ...
            'UseQuadprog',true);

        metrics.kf_filter.W_raw       = W;
        metrics.kf_filter.W_filt      = W_filt;
        metrics.kf_filter.bounds.lb   = lb;
        metrics.kf_filter.bounds.ub   = ub;
        metrics.kf_filter.bounds.Lmax = Lmax;
        metrics.kf_forecast.weights   = kf_next_year_weights;

        % ---------------- Optional: compare KF vs REAL on next 12 months ----------------
        % Build REAL next-year weights by re-optimizing over the next 12 rolls
        t0_real   = t0;     % where the main loop left off
        tend_real = tend;
        real_next_year_weights = nan(nAssets, HORIZON);
        h_eff = 0;

        for h = 1:HORIZON
            if tend_real >= width(Prices_returns)
                break;  % no more data
            end
            train_cols_h = t0_real : tend_real;
            Train_tbl_h  = Prices_returns(:, [1, NAME_COL, train_cols_h]);

            [mu_h, Sig_h] = compute_mean_and_covariance(Train_tbl_h);

            if flag
                [~, ~, Port_h] = compute_esg_constrained_frontier( ...
                                    mu_h, Sig_h, esg_scores_aligned, ...
                                    thresholds(portfolio_index), size(esg_scores_aligned, 1), max_short, size(esg_scores_aligned, 1), flag, max_short);
            else
                [~, ~, Port_h] = compute_esg_constrained_frontier( ...
                                    mu_h, Sig_h, esg_scores_aligned, ...
                                    thresholds(portfolio_index), size(esg_scores_aligned, 1), max_short, size(esg_scores_aligned, 1));
            end

            [wL_h, ~] = estimateFrontierLimits(Port_h);
            w_real_h = wL_h(:,1);
            % Project to same feasibility set as KF
            w_real_h = project_weight_vector(w_real_h, lb, ub, Lmax, true);

            h_eff = h_eff + 1;
            real_next_year_weights(:, h_eff) = w_real_h;

            % slide one month
            t0_real   = t0_real   + 1;
            tend_real = tend_real + 1;
        end

        if h_eff > 0
            K  = min(HORIZON, h_eff);
            KF = kf_next_year_weights(:, 1:K);
            RL = real_next_year_weights(:, 1:K);

            % Weights-level diagnostics
            D  = KF - RL;
            L1 = sum(abs(D), 1);                        % per-step L1
            L2 = sqrt(sum(D.^2, 1));                    % per-step L2
            COS = zeros(1, K);
            for j = 1:K
                COS(j) = dot(KF(:,j), RL(:,j)) / (norm(KF(:,j))*norm(RL(:,j)) + eps);
            end
            per_asset_MAE = mean(abs(D), 2, 'omitnan');

            Step = (1:K).';
            compare_tbl = table(Step, L1.', L2.', COS.', ...
                'VariableNames', {'Step','L1_sumAbs','L2_norm','CosineSim'});

            % Returns for those K months
            next_cols = (tend + 1):(tend + K);
            next_cols = next_cols(next_cols <= width(Prices_returns));
            K_eff     = numel(next_cols);

            kf_est_returns   = nan(1, K_eff);
            real_est_returns = nan(1, K_eff);
            for j = 1:K_eff
                rj = Prices_returns{:, next_cols(j)}; rj = rj(:);
                kf_est_returns(j)   = sum(KF(:, j) .* rj, 'omitnan');
                real_est_returns(j) = sum(RL(:, j) .* rj, 'omitnan');
            end

            % Returns-level distances
            ret_L1 = sum(abs(kf_est_returns(:) - real_est_returns(:)), 'omitnan');
            ret_L2 = sqrt(sum((kf_est_returns(:) - real_est_returns(:)).^2, 'omitnan'));

            % Final-horizon performance (KF vs Real)
            frequency      = 12; risk_free_rate = 0;
            kf_table = table( ...
                annualized_return(kf_est_returns, frequency) * 100, ...
                annualized_volatility(kf_est_returns, frequency) * 100, ...
                sharpe_ratio(kf_est_returns, frequency, risk_free_rate), ...
                max_drawdown(kf_est_returns) * 100, ...
                calmar_ratio(kf_est_returns, frequency), ...
                'VariableNames', {'AnnualizedReturn_Pct','AnnualizedVolatility_Pct','SharpeRatio','MaxDrawdown_Pct','CalmarRatio'});

            rl_table = table( ...
                annualized_return(real_est_returns, frequency) * 100, ...
                annualized_volatility(real_est_returns, frequency) * 100, ...
                sharpe_ratio(real_est_returns, frequency, risk_free_rate), ...
                max_drawdown(real_est_returns) * 100, ...
                calmar_ratio(real_est_returns, frequency), ...
                'VariableNames', {'AnnualizedReturn_Pct','AnnualizedVolatility_Pct','SharpeRatio','MaxDrawdown_Pct','CalmarRatio'});

            comp_table = table( ...
                kf_table.AnnualizedReturn_Pct,  rl_table.AnnualizedReturn_Pct, ...
                kf_table.AnnualizedVolatility_Pct, rl_table.AnnualizedVolatility_Pct, ...
                kf_table.SharpeRatio,            rl_table.SharpeRatio, ...
                kf_table.MaxDrawdown_Pct,        rl_table.MaxDrawdown_Pct, ...
                kf_table.CalmarRatio,            rl_table.CalmarRatio, ...
                sum(L1,'omitnan')/K, max(L1,[],'omitnan'), ...
                sum(L2,'omitnan')/K, max(L2,[],'omitnan'), ...
                mean(COS,'omitnan'), ret_L1, ret_L2, ...
                'VariableNames', { ...
                    'KF_AnnRet_%','Real_AnnRet_%', ...
                    'KF_AnnVol_%','Real_AnnVol_%', ...
                    'KF_Sharpe','Real_Sharpe', ...
                    'KF_MDD_%','Real_MDD_%', ...
                    'KF_Calmar','Real_Calmar', ...
                    'W_L1_mean','W_L1_max','W_L2_mean','W_L2_max','W_Cos_mean', ...
                    'Ret_L1','Ret_L2' ...
                });

            % Store diagnostics
            metrics.kf_vs_real.available_steps     = K_eff;
            metrics.kf_vs_real.kf_weights          = KF;
            metrics.kf_vs_real.real_weights        = RL;
            metrics.kf_vs_real.summary_per_step    = compare_tbl;
            metrics.kf_vs_real.per_asset_MAE       = per_asset_MAE;
            metrics.kf_vs_real.kf_est_returns      = kf_est_returns(:);
            metrics.kf_vs_real.real_est_returns    = real_est_returns(:);
            metrics.kf_vs_real.performance_kf      = kf_table;
            metrics.kf_vs_real.performance_real    = rl_table;
            metrics.kf_vs_real.performance_compare = comp_table;
        else
            metrics.kf_vs_real.note = 'Insufficient future columns to compute REAL next-year comparison.';
        end
    else
        metrics.kf_filter.note = 'Insufficient history to run Kalman filter.';
    end
end

% ======================================================================
%                          HELPER FUNCTIONS
% ======================================================================

function [W_filt, kf_next_year_weights] = kf_weights_constrained( ...
        W, lb, ub, Lmax, HORIZON, varargin)
% Constrained Kalman filter on weight history W (n x T).
% Dynamics: per-asset AR(1) with damped phi.
% Measurement: z_t = I * w_t + eps.
% Projection: sum(w)=1, lb<=w<=ub, sum(|w|)<=Lmax.

p = inputParser;
addParameter(p,'DampPhi',0.98);
addParameter(p,'RLevel',1e-4);
addParameter(p,'RAdaptive',true);
addParameter(p,'QFloor',1e-6);
addParameter(p,'GainDamp',0.85);
addParameter(p,'UseQuadprog',true);
parse(p,varargin{:});
opt = p.Results;

[n,T] = size(W);
W_filt = zeros(n,T);

% Estimate AR(1)
phi = zeros(n,1); q = zeros(n,1);
if T >= 3
    for i = 1:n
        x0 = W(i,1:end-1)'; x1 = W(i,2:end)';
        den = x0'*x0;
        phi_i = (den>0) * ((x0'*x1)/max(den,eps));
        phi(i) = min(max(phi_i, -0.95), 0.95) * opt.DampPhi;
        res = x1 - phi(i)*x0;
        q(i) = max(var(res,1), opt.QFloor);
    end
else
    phi(:) = 0.9*opt.DampPhi; q(:) = 1e-4;
end
A = diag(phi); Q = diag(q);

% Init
x = W(:,1);
P = diag(q ./ max(1 - phi.^2, 1e-3));
P(~isfinite(P)) = 1e-2; P = P + 1e-6*eye(n);

H = eye(n);
R = opt.RLevel * eye(n);

% Project initial state
x = project_weight_vector(x, lb, ub, Lmax, opt.UseQuadprog);
W_filt(:,1) = x;

% Filter
for t = 2:T
    z = W(:,t);

    % Predict
    x_pred = A * x;
    P_pred = A * P * A' + Q;

    % Optional projection on prediction
    x_pred = project_weight_vector(x_pred, lb, ub, Lmax, opt.UseQuadprog);

    % Adaptive R
    if opt.RAdaptive
        innov = z - x_pred;
        s = median(abs(innov));
        R_eff = (opt.RLevel + 10*s^2) * eye(n);
    else
        R_eff = R;
    end

    % Update (damped gain)
    S = H * P_pred * H' + R_eff;
    K = P_pred / S;                   % since H=I
    K = opt.GainDamp * K;

    x = x_pred + K * (z - H * x_pred);
    P = (eye(n) - K*H) * P_pred;

    % Projection
    x = project_weight_vector(x, lb, ub, Lmax, opt.UseQuadprog);

    W_filt(:,t) = x;
end

% Forecast HORIZON steps (predict-only)
xF = x; PF = P;
kf_next_year_weights = zeros(n, HORIZON);
for h = 1:HORIZON
    xF = A * xF;
    PF = A * PF * A' + Q;
    xF = project_weight_vector(xF, lb, ub, Lmax, opt.UseQuadprog);
    kf_next_year_weights(:,h) = xF;
end
end

function w = project_weight_vector(v, lb, ub, Lmax, useQP)
% Project to: sum(w)=1, lb<=w<=ub, sum(abs(w))<=Lmax
n = numel(v);
if isscalar(lb), lb = lb*ones(n,1); end
if isscalar(ub), ub = ub*ones(n,1); end
if isempty(Lmax) || ~isfinite(Lmax) || Lmax<=0, Lmax = 1; end

if useQP && exist('quadprog','file')==2
    % Split variables: w = w+ - w-, w+,w- >= 0
    H = 2*eye(2*n);
    f = [-2*v; 2*v];
    % Equality: sum(w+ - w-) = 1
    Aeq = [ones(1,n), -ones(1,n)]; beq = 1;
    % Inequalities:
    % sum(w+ + w-) <= Lmax
    A1 = [ones(1,n), ones(1,n)];  b1 = Lmax;
    % w+ - w- <= ub
    A2 = [eye(n), -eye(n)];       b2 = ub;
    % -w+ + w- <= -lb
    A3 = [-eye(n), eye(n)];       b3 = -lb;
    A = [A1; A2; A3];
    b = [b1; b2; b3];

    lb_split = zeros(2*n,1); ub_split = inf(2*n,1);
    opts = optimoptions('quadprog','Display','off');
    [x,~,exitflag] = quadprog(H,f,A,b,Aeq,beq,lb_split,ub_split,[],opts);
    if exitflag <= 0
        w = project_heuristic(v, lb, ub, Lmax);
    else
        w = x(1:n) - x(n+1:end);
        % Light box & normalize polish
        w = min(max(w, lb), ub);
        w = w / sum(w);
    end
else
    w = project_heuristic(v, lb, ub, Lmax);
end
end

function w = project_heuristic(v, lb, ub, Lmax)
% Heuristic projection: box clip -> gross shrink -> normalize -> polish
w = min(max(v, lb), ub);
g = sum(abs(w));
if g > Lmax
    w = (Lmax / g) * w;
end
s = sum(w);
if abs(s) < 1e-12
    n = numel(w);
    w = max(lb, min(ub, ones(n,1)/n));
    s = sum(w);
end
w = w / s;
w = min(max(w, lb), ub);
w = w / sum(w);
end
