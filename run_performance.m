function [portfolio_returns, portfolio_cum_returns, metrics_table] = run_performance(BT_returns, Portfolio, mean_returns, portfolio_index, portfolio_type, flag_plot, esg_scores, top_n)

    % ------------------------------------------------------------------------------
    % Purpose:
    %   Evaluate a given Portfolio object on a return panel, producing periodic
    %   and cumulative portfolio returns and a table of summary performance metrics.
    %
    % Inputs:
    %   BT_returns     : table with an identifier column (CompanyCommonName) and
    %                    subsequent columns containing periodic asset returns.
    %   Portfolio      : Portfolio object (or array) from which weights are derived.
    %   mean_returns   : table/array with expected returns aligned to asset names.
    %   portfolio_index: index selecting which Portfolio element to use.
    %   portfolio_type : string controlling the weight extraction
    %                    ('MaxSharpe' or 'MinVariance').
    %   flag_plot      : logical; if true, plot cumulative returns and print metrics.
    %   esg_scores     : (optional) table/array of ESG scores aligned to assets.
    %   top_n          : (optional) integer; if provided with esg_scores, restrict
    %                    the evaluation to the top_n names by ESG score.
    %
    % Outputs:
    %   portfolio_returns     : 1×T row vector of periodic portfolio returns.
    %   portfolio_cum_returns : 1×T row vector of cumulative (1+ret) product.
    %   metrics_table         : 1×5 table with annualized return/vol, Sharpe,
    %                           max drawdown, and Calmar ratio.
    % ------------------------------------------------------------------------------

    % Optional filtering: if ESG scores and top_n are provided, select top names
    if nargin > 6
        esg_array = table2array(esg_scores);
        [~, idx] = sort(esg_array, 'descend');
        selected_idx = idx(1:min(top_n, length(idx)));
        selected_names = mean_returns.Properties.RowNames(selected_idx);
    else
        selected_names = mean_returns.Properties.RowNames;
    end

    % Derive weights from the chosen portfolio type
    if strcmp(portfolio_type, 'MaxSharpe')
        [selected_weights, ~] = estimateMaxSharpeRatio(Portfolio(portfolio_index));
    elseif strcmp(portfolio_type, 'MinVariance')
        [selected_weights, ~] =  estimateFrontierLimits(Portfolio(portfolio_index));
        selected_weights = selected_weights(:,1);
    else
        error('Invalid portfolio type. Use "MaxSharpe" or "MinVariance".');
    end

    % Map weights to the backtest universe by name, filling missing entries with zero
    bt_names = string(BT_returns.CompanyCommonName(:));                 
    [is_in_bt, idx_sel_for_bt] = ismember(bt_names, selected_names);

    weights_aligned = zeros(numel(bt_names), 1);
    weights_aligned(is_in_bt) = selected_weights(idx_sel_for_bt(is_in_bt));

    % Notify if there are selected names not present in the BT universe
    missing_in_bt = setdiff(selected_names, bt_names);
    if ~isempty(missing_in_bt)
        % fprintf('Warning: %d selected not present in backtest:\n', numel(missing_in_bt));
        % disp(missing_in_bt(:).');
    end

    % Normalize weights if they sum to a positive number
    s = sum(weights_aligned);
    if s > 0
        weights_aligned = weights_aligned / s;
    else
        warning('Zero weights sum after alignment: check name mapping.');
    end

    % Compute periodic and cumulative returns on the evaluation panel
    returns_matrix = BT_returns{:, 3:end};
    returns_matrix(isnan(returns_matrix)) = 0;
    portfolio_returns = (weights_aligned.' * returns_matrix);
    portfolio_cum_returns = cumprod(1 + portfolio_returns);

    % Summary performance statistics (annualized where applicable)
    frequency = 12;       
    risk_free_rate = 0;   

    ann_return = annualized_return(portfolio_returns, frequency);
    ann_volatility = annualized_volatility(portfolio_returns, frequency);
    sharpe = sharpe_ratio(portfolio_returns, frequency, risk_free_rate);
    mdd = max_drawdown(portfolio_returns);
    calmar = calmar_ratio(portfolio_returns, frequency);

    % Output table with key performance metrics
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

    % Optional visualization and console summary
    if flag_plot
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
        fprintf('\n--- METRICS ESG Portfolio (%d) ---\n', portfolio_index);
        disp(metrics_table);
    end
end