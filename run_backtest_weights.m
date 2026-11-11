function [best_portfolio_returns, best_portfolio_cum_returns, best_metrics, best_weights] = run_backtest_weights(excelFile, sheetName, weights_matrix, mean_returns, esg_scores, top_n)

    % Performs backtesting over a set of portfolios defined by weights.
    % Automatically selects the portfolio with the maximum Sharpe ratio.
    %
    %   INPUTS:
    %   excelFile      = Excel file containing asset prices
    %   sheetName      = Sheet with price data
    %   weights_matrix = [n_assets x n_portfolios] matrix of weights
    %   mean_returns   = mean return table (with asset names as RowNames)
    %   esg_scores     = (optional) ESG scores table
    %   top_n          = (optional) number of assets with highest ESG scores
    %
    %   OUTPUTS:
    %   best_portfolio_returns     = vector of periodic returns of the best portfolio
    %   best_portfolio_cum_returns = cumulative returns
    %   best_metrics               = performance metrics table
    %   best_weights               = weights vector of the best portfolio

    % === 1. Handle optional ESG filter ===
    if nargin > 4
        esg_array = table2array(esg_scores);
        [~, idx] = sort(esg_array, 'descend');
        selected_idx = idx(1:min(top_n, length(idx)));
        selected_names = mean_returns.Properties.RowNames(selected_idx);
    else
        selected_names = mean_returns.Properties.RowNames;
    end

    % === 2. Load prices ===
    BT_raw = readcell(excelFile, 'Sheet', sheetName);
    company_names_bt = string(BT_raw(2:end, 2));
    price_cells = BT_raw(2:end, 3:end);

    [n_assets, n_periods] = size(price_cells);
    price_matrix = NaN(n_assets, n_periods);
    for i = 1:n_assets
        for j = 1:n_periods
            val = price_cells{i, j};
            if isnumeric(val)
                price_matrix(i, j) = val;
            else
                price_matrix(i, j) = NaN;
            end
        end
    end

    % === 3. Compute asset returns ===
    meta_table = table(company_names_bt, company_names_bt, 'VariableNames', {'Name', 'Ticker'});
    price_table = array2table(price_matrix);
    BT_table = [meta_table price_table];
    BT_returns = compute_price_returns(BT_table);
    returns_matrix = BT_returns{:, 3:end};
    returns_matrix(isnan(returns_matrix)) = 0;

    % === 4. Align names between backtest and weights (robust, ESG-aware) ===
    bt_names   = string(BT_returns.Name(:));               % assets in Excel
    all_names  = string(mean_returns.Properties.RowNames);
    esg_array  = table2array(esg_scores);
    
    % Step 1. Match mean_returns universe with weights_matrix rows
    if size(weights_matrix,1) ~= numel(all_names)
        % Restrict to intersection only
        fprintf('⚠ Aligning weights_matrix (%d) with mean_returns (%d)\n', ...
                 size(weights_matrix,1), numel(all_names));
    
        n_assets = size(weights_matrix,1);
        all_names = all_names(1:n_assets);   % force same dimension
        esg_array = esg_array(1:n_assets);   % trim ESG scores to match
    end
    
    % Step 2. Apply ESG filter (optional)
    if nargin > 4
        [~, idx_sorted] = sort(esg_array, 'descend');
        selected_idx = idx_sorted(1:min(top_n, length(idx_sorted)));
    
        selected_names   = all_names(selected_idx);
        weights_matrix   = weights_matrix(selected_idx, :);
        all_names        = all_names(selected_idx);
    else
        selected_names = all_names;
    end
    
    % Step 3. Intersect with assets available in backtest
    [~, idx_bt, idx_all] = intersect(bt_names, all_names, 'stable');
    
    aligned_weights_matrix = zeros(numel(bt_names), size(weights_matrix, 2));
    aligned_weights_matrix(idx_bt, :) = weights_matrix(idx_all, :);


    % === 5. Compute portfolio returns for all portfolios ===
    n_portfolios = size(aligned_weights_matrix, 2);
    all_portfolio_returns = zeros(n_portfolios, size(returns_matrix, 2));
    all_sharpes = zeros(n_portfolios, 1);

    frequency = 12;          % monthly
    risk_free_rate = 0.01;   % annual

    for k = 1:n_portfolios
        w = aligned_weights_matrix(:, k);
        port_ret = (w.' * returns_matrix);
        all_portfolio_returns(k, :) = port_ret;
        all_sharpes(k) = sharpe_ratio(port_ret, frequency, risk_free_rate);
    end

    % === 6. Select best portfolio (max Sharpe) ===
    [~, best_idx] = max(all_sharpes);
    best_weights = aligned_weights_matrix(:, best_idx);
    best_portfolio_returns = all_portfolio_returns(best_idx, :);
    best_portfolio_cum_returns = cumprod(1 + best_portfolio_returns);

    % === 7. Compute performance metrics for the best portfolio ===
    ann_return = annualized_return(best_portfolio_returns, frequency);
    ann_volatility = annualized_volatility(best_portfolio_returns, frequency);
    sharpe = sharpe_ratio(best_portfolio_returns, frequency, risk_free_rate);
    mdd = max_drawdown(best_portfolio_returns);
    calmar = calmar_ratio(best_portfolio_returns, frequency);

    best_metrics = table( ...
        ann_return * 100, ...
        ann_volatility * 100, ...
        sharpe, ...
        mdd * 100, ...
        calmar, ...
        'VariableNames', { ...
            'AnnualizedReturn_Pct', ...
            'AnnualizedVolatility_Pct', ...
            'SharpeRatio', ...
            'MaxDrawdown_Pct', ...
            'CalmarRatio' ...
        });

    % === 8. Plot best portfolio ===
    figure; hold on;
    plot(best_portfolio_cum_returns, '-', ...
         'LineWidth', 2, ...
         'Color', 'b', ...
         'DisplayName', sprintf('Best Portfolio (Sharpe %.2f)', sharpe));
    legend('show', 'Location', 'best');
    title('Backtest - Best Sharpe Portfolio');

    % === 9. Print metrics ===
    fprintf('\n--- BEST PORTFOLIO (Sharpe = %.2f) ---\n', sharpe);
    disp(best_metrics);

end
