function [all_oos_returns, all_oos_cum, metrics] = run_prevision_rolling_minsum( ...
                    Prices_returns, portfolio_index, portfolio_type, years, num_low_esg, ...
                    esg_scores_aligned, flag, max_low_weight, max_short, threshold)

    % RUN_PREVISION_ROLLING_MINSUM
    % -------------------------------------------------------------------------
    % Purpose:
    %   Execute a rolling backtest with fixed training and testing windows.
    %   For each iteration, a portfolio is estimated on a training window and
    %   evaluated on the subsequent test window. Out-of-sample returns from all
    %   iterations are concatenated to form a single performance series.
    %
    % Inputs:
    %   Prices_returns     : table with an identifier column and subsequent periodic returns.
    %   portfolio_index    : scenario selector for parameter vectors.
    %   portfolio_type     : string for the evaluation type (e.g., 'MaxSharpe', 'MinVariance').
    %   years              : number of rolling iterations to run.
    %   num_low_esg        : vector controlling the number of low-ESG names for constraints.
    %   esg_scores_aligned : ESG scores aligned to the asset universe.
    %   flag               : logical/config switch forwarded to the frontier builder.
    %   max_low_weight     : bound parameter for low-ESG aggregate exposure.
    %   max_short          : bound parameter for short exposure.
    %   threshold          : threshold parameter for constraint calibration.
    %
    % Outputs:
    %   all_oos_returns : row vector of concatenated out-of-sample returns.
    %   all_oos_cum     : row vector of cumulative performance over OOS periods.
    %   metrics         : struct with per-window and overall performance tables.
    % -------------------------------------------------------------------------

    % Ensure a standard name column is present for downstream compatibility
    if ~any(strcmp(Prices_returns.Properties.VariableNames, 'CompanyCommonName'))
        Prices_returns.Properties.VariableNames{1} = 'CompanyCommonName';
    end

    % Column and window parameters
    NAME_COL = 2;       % column containing identifiers
    RET_START = 3;      % first column containing returns
    TRAIN_M = 60;       % training window length (periods)
    TEST_M  = 12;       % testing window length (periods)

    % Containers for aggregation
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
        Train_tbl  = Prices_returns(:, [NAME_COL, train_cols]);

        % Estimation on training data
        [mean_returns, cov_matrix] = compute_mean_and_covariance(Train_tbl);

        % Portfolio construction (signature depends on configuration flag)
        if flag
            [~, ~, Portfolio] = compute_esg_constrained_frontier(mean_returns, cov_matrix, ...
                          esg_scores_aligned, threshold, size(esg_scores_aligned,1), max_low_weight, num_low_esg(portfolio_index), true, max_short);
        else
            [~, ~, Portfolio] = compute_esg_constrained_frontier(mean_returns, cov_matrix, ...
                          esg_scores_aligned, threshold, size(esg_scores_aligned,1), max_low_weight, num_low_esg(portfolio_index));
        end

        % Testing window selection
        test_cols = tend : (tend + TEST_M);
        Test_tbl  = Prices_returns(:, [NAME_COL, test_cols]);

        % Out-of-sample evaluation
        [oos_ret, ~, mt] = run_performance(Test_tbl, Portfolio, mean_returns, 1, portfolio_type, false);

        % Store results for this iteration
        all_oos_returns = [all_oos_returns, oos_ret]; %#ok<AGROW>
        per_window_metrics{j} = mt;

        % Basic label for traceability
        per_window_labels(j) = sprintf('Window %d (train %d m, test %d m)', j, TRAIN_M, TEST_M);

        % Advance to next iteration
        t0 = t0 + TEST_M;
        tend = tend + TEST_M;
    end

    % Cumulative performance built from OOS returns
    all_oos_returns = all_oos_returns(:)';         % row vector
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

    % High-level visualization of aggregated OOS performance
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
    
    % Overall performance on the concatenated OOS series
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

    % Console summary
    fprintf('\n--- METRICS ESG Portfolio (%d) ---\n', portfolio_index);
    disp(metrics.overall);
end
