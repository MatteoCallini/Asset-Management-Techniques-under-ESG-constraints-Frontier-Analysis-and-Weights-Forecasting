function [portfolio_returns, portfolio_cum_returns, metrics_table] = run_backtest(excelFile, sheetName, Portfolio, mean_returns, portfolio_index, portfolio_type, esg_scores, top_n)

    % Performs backtesting of a portfolio and returns:
    % - periodic return,
    % - cumulative return,
    % - table with performance metrics.
    % portfolio_type: 'MaxSharpe' or 'MinVariance' to specify which portfolio to use

    % Handle optional arguments
    if nargin > 6
        esg_array = table2array(esg_scores);
        [~, idx] = sort(esg_array, 'descend');
        selected_idx = idx(1:min(top_n, length(idx)));
        selected_names = mean_returns.Properties.RowNames(selected_idx);
    else
        selected_names = mean_returns.Properties.RowNames;
    end

    % === 1. Estimate weights based on portfolio type ===
    if strcmp(portfolio_type, 'MaxSharpe')
        [selected_weights, ~] = estimateMaxSharpeRatio(Portfolio(portfolio_index));
    elseif strcmp(portfolio_type, 'MinVariance')
        % Compute the minimum variance portfolio weights
        [selected_weights, ~] =  estimateFrontierLimits(Portfolio(portfolio_index));
        selected_weights = selected_weights(:,1);
    else
        error('Invalid portfolio type. Use "MaxSharpe" or "MinVariance".');
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

    % === 3. Table for compute_price_returns ===
    meta_table = table(company_names_bt, company_names_bt, 'VariableNames', {'Name', 'Ticker'});
    price_table = array2table(price_matrix);
    BT_table = [meta_table price_table];

    % === 4. Compute returns ===
    BT_returns = compute_price_returns(BT_table);

    % === 5. Align names and weights (ROBUST: no logical masks on different arrays) ===
    % Goal: build a weight vector aligned with BT_returns.Name (backtest universe)
    bt_names = string(BT_returns.Name(:));                  % |bt_names| = n_assets_backtest
    [is_in_bt, idx_sel_for_bt] = ismember(bt_names, selected_names);

    % Weights aligned with the BT universe (zeros where the stock is not in the selected set)
    weights_aligned = zeros(numel(bt_names), 1);
    weights_aligned(is_in_bt) = selected_weights(idx_sel_for_bt(is_in_bt));

    % Useful warnings on mismatches
    missing_in_bt = setdiff(selected_names, bt_names);
    if ~isempty(missing_in_bt)
        fprintf('Warning: %d selected not present in backtest:\n', numel(missing_in_bt));
        disp(missing_in_bt(:).');
    end

    % Normalize only if the sum is positive
    s = sum(weights_aligned);
    if s > 0
        weights_aligned = weights_aligned / s;
    else
        warning('Zero weights sum after alignment: check name mapping.');
    end

    % === 6. Portfolio return ===
    returns_matrix = BT_returns{:, 3:end};       % rows = stocks (bt_names), columns = periods
    returns_matrix(isnan(returns_matrix)) = 0;   % avoid NaN in the product
    portfolio_returns = (weights_aligned.' * returns_matrix);
    portfolio_cum_returns = cumprod(1 + portfolio_returns);

    % === 7. Plot ===
    fig = figure(gcf); hold on;
    lgd = findobj(fig, 'Type', 'Legend');
    if isempty(lgd)
        lgd = legend('show', 'Location', 'best');
        set(lgd, 'AutoUpdate', 'on');
    end
    plot(portfolio_cum_returns, '-', ...
         'LineWidth', 2, ...
         'Color', rand(1,3), ...
         'DisplayName', sprintf('Portfolio %d', portfolio_index));

    % === 8. PERFORMANCE METRICS ===
    frequency = 12;          % 12 = monthly
    risk_free_rate = 0.01;   % annual

    ann_return = annualized_return(portfolio_returns, frequency);
    ann_volatility = annualized_volatility(portfolio_returns, frequency);
    sharpe = sharpe_ratio(portfolio_returns, frequency, risk_free_rate);
    mdd = max_drawdown(portfolio_returns);
    calmar = calmar_ratio(portfolio_returns, frequency);

    % === 9. Output table ===
    metrics_table = table( ...
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

    % === 10. Print to console ===
    fprintf('\n--- METRICS ESG Portfolio (%d) ---\n', portfolio_index);
    disp(metrics_table);
end
