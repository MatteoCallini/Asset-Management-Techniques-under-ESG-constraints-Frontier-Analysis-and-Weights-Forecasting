function [all_oos_returns, all_oos_cum, metrics] = run_prevision_rolling_best( ...
                    Prices_returns, portfolio_index, portfolio_type, years, num_low_esg, ...
                    esg_scores_aligned, short_bound_high, short_bound_low, threshold)

% -------------------------------------------------------------------------
% Purpose:
%   Rolling backtest with fixed-length training and subsequent testing windows.
%   Each iteration estimates a portfolio on a training window and evaluates it
%   on the following test window. Out-of-sample returns are concatenated.
%
% Inputs:
%   Prices_returns     : table with asset identifiers and periodic returns.
%   portfolio_index    : index selecting scenario-specific parameters.
%   portfolio_type     : string selecting the portfolio type for evaluation.
%   years              : number of rolling iterations to perform.
%   num_low_esg        : vector configuring Best-in-Class settings per scenario.
%   esg_scores_aligned : ESG scores aligned to the asset universe.
%   short_bound_high   : upper bound used in portfolio construction.
%   short_bound_low    : lower bound used in portfolio construction.
%   threshold          : threshold parameter for the Best-in-Class construction.
%
% Outputs:
%   all_oos_returns : row vector of concatenated out-of-sample returns.
%   all_oos_cum     : row vector of cumulative returns (1+ret) product.
%   metrics         : struct with per-window and overall performance tables.
% -------------------------------------------------------------------------

    % Ensure first column contains names required by downstream functions
    if ~any(strcmp(Prices_returns.Properties.VariableNames, 'CompanyCommonName'))
        Prices_returns.Properties.VariableNames{1} = 'CompanyCommonName';
    end

    % Indices and window lengths
    NAME_COL = 2;       % column index for asset names
    RET_START = 3;      % first column index containing returns
    TRAIN_M = 60;       % training window length (periods)
    TEST_M  = 12;       % testing window length (periods)

    % Aggregators for results
    all_oos_returns = [];                              
    per_window_metrics = cell(years, 1);               
    per_window_labels  = strings(years, 1);

    t0 = RET_START;

    for j = 1:years
        % Training window selection
        if j==1
            tend = (t0 + TRAIN_M - 2);
        end

        train_cols = t0 : tend;
        Train_tbl  = Prices_returns(:, [1, NAME_COL, train_cols]);

        % Estimation on training data
        [mean_returns, cov_matrix] = compute_mean_and_covariance(Train_tbl);

        [~, ~, Portfolio] = compute_best_in_class_frontier( ...
                                            mean_returns, cov_matrix, esg_scores_aligned, threshold, ...
                                            size(esg_scores_aligned,1), num_low_esg(portfolio_index), short_bound_high, short_bound_low);

        % Testing window selection
        test_cols = tend : (tend + TEST_M);
        Test_tbl  = Prices_returns(:, [NAME_COL, test_cols]);

        % Out-of-sample evaluation
        [oos_ret, ~, mt] = run_performance(Test_tbl, Portfolio, mean_returns, 1, portfolio_type, false);

        % Store results
        all_oos_returns = [all_oos_returns, oos_ret]; 
        per_window_metrics{j} = mt;

        % Label for this iteration
        per_window_labels(j) = sprintf('Window %d (train %d m, test %d m)', j, TRAIN_M, TEST_M);

        % Advance to next iteration
        t0 = t0 + TEST_M;
        tend = tend + TEST_M;
    end

    % Build cumulative performance from concatenated OOS returns
    all_oos_returns = all_oos_returns(:)';         
    all_oos_cum     = cumprod(1 + all_oos_returns);

    % Assemble per-window metrics table
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

    % Plot aggregated out-of-sample cumulative curve
    fig = figure(gcf); hold on;
    lgd = findobj(fig, 'Type', 'Legend');
    if isempty(lgd)
            lgd = legend('show', 'Location', 'best');
            set(lgd, 'AutoUpdate', 'on');
    end
    plot(all_oos_cum, '-', ...
            'LineWidth', 2, ...
            'Color', rand(1,3), ...
            'DisplayName', sprintf('Portfolio %d', portfolio_index));
    
    % Overall performance summary
    freq = 12;
    rf = 0;
    overall_ann_return   = annualized_return(all_oos_returns, freq);
    overall_ann_vol      = annualized_volatility(all_oos_returns, freq);
    overall_sharpe       = sharpe_ratio(all_oos_returns, freq, rf);
    overall_mdd          = max_drawdown(all_oos_returns);
    overall_calmar       = calmar_ratio(all_oos_returns, freq);

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

    % Print overall metrics
    fprintf('\n--- METRICS ESG Portfolio (%d) ---\n', portfolio_index);
    disp(metrics.overall);
end
