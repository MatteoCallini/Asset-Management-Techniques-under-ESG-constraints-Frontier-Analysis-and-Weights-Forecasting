function [all_oos_returns, all_oos_cum, metrics] = run_prevision_rolling_sector( ...
                    Prices_returns, portfolio_index, portfolio_type, years, sorted_sector_mean, Sector_ESG_table, N_worst, flag, max_short)

    % -------------------------------------------------------------------------
    % Purpose:
    %   Perform a rolling backtest with fixed training and testing windows while
    %   applying a sector-based filter. In each iteration, the portfolio is
    %   estimated on a training window after excluding a set of sectors, and then
    %   evaluated on the subsequent out-of-sample window. Out-of-sample returns
    %   are concatenated across iterations.
    %
    % Inputs:
    %   Prices_returns   : table with an identifier column and subsequent periodic returns.
    %   portfolio_index  : index used to select scenario-specific parameters (e.g., N_worst).
    %   portfolio_type   : portfolio type passed to the evaluator (e.g., 'MaxSharpe', 'MinVariance').
    %   years            : number of rolling iterations to execute.
    %   sorted_sector_mean : table with sector ranking (e.g., by average ESG/score), ascending order.
    %   Sector_ESG_table : table mapping companies to sectors (and optionally ESG info).
    %   N_worst          : vector specifying how many lowest-ranked sectors to exclude per scenario.
    %   flag             : logical/config switch forwarded to the portfolio constructor.
    %   max_short        : parameter forwarded to the portfolio constructor for short bounds.
    %
    % Outputs:
    %   all_oos_returns : row vector of concatenated out-of-sample returns across iterations.
    %   all_oos_cum     : row vector of cumulative performance built from OOS returns.
    %   metrics         : struct with per-window and overall performance summary tables.
    % -------------------------------------------------------------------------

    % Standardize the identifier column name for downstream compatibility
    if ~any(strcmp(Prices_returns.Properties.VariableNames, 'CompanyCommonName'))
        Prices_returns.Properties.VariableNames{1} = 'CompanyCommonName';
    end

    % Column indexing and window sizes
    NAME_COL = 2;       % column with identifiers
    RET_START = 3;      % first returns column
    TRAIN_M = 60;       % training window length
    TEST_M = 12;        % testing window length

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

        % Sector filtering (exclude a set of lowest-ranked sectors)
        worst_sectors = sorted_sector_mean.Sector(1:N_worst(portfolio_index));
        Sector_ESG_filtered = Sector_ESG_table(~ismember(Sector_ESG_table.Sector, worst_sectors), :);

        % Informative display (optional)
        disp('Sectors removed due to low average ESG score:');
        disp(worst_sectors);

        % Selected companies after sector filter
        selected_companies = Sector_ESG_filtered.Company;

        % Estimation inputs on the training slice
        [mean_returns, cov_matrix] = compute_mean_and_covariance(Train_tbl);

        % Align statistics to the filtered universe
        filtered_mean_returns = mean_returns(selected_companies, :);
        filtered_cov_matrix = cov_matrix(selected_companies, selected_companies);

        % Portfolio construction on the filtered universe
        n_filtered = length(selected_companies);
        if flag
            [~, ~, Portfolio] = compute_portfolio_frontier(...
                filtered_mean_returns, filtered_cov_matrix, n_filtered, true, max_short);
        else
            [~, ~, Portfolio] = compute_portfolio_frontier(...
                filtered_mean_returns, filtered_cov_matrix, n_filtered, false, max_short);
        end

        % Testing window selection
        test_cols = tend : (tend + TEST_M);
        Test_tbl  = Prices_returns(:, [NAME_COL, test_cols]);

        % Out-of-sample evaluation using the portfolio estimated on training data
        [oos_ret, ~, mt] = run_performance(Test_tbl, Portfolio, filtered_mean_returns, 1, portfolio_type, false);

        % Store results
        all_oos_returns = [all_oos_returns, oos_ret];
        per_window_metrics{j} = mt;

        % Label for traceability
        per_window_labels(j) = sprintf('Window %d (train %d m, test %d m)', j, TRAIN_M, TEST_M);

        % Advance rolling pointers
        t0 = t0 + TEST_M;
        tend = tend + TEST_M;

    end

    % Build cumulative performance from concatenated OOS returns
    all_oos_returns = all_oos_returns(:)'; % row vector
    all_oos_cum = cumprod(1 + all_oos_returns);

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

    % High-level plot of aggregated OOS performance
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
    
    % Overall metrics on concatenated OOS returns
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

end
