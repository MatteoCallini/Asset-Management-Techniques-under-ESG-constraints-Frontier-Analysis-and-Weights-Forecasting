function [FrontierVola, FrontierRet, p] = compute_portfolio_frontier(mean_returns, cov_matrix, top_n, allow_short_selling, short_bound)
    
    % Computes efficient frontier using MATLAB's Portfolio object
    %
    % Inputs:
    % - mean_returns: table with row names (assets) and 'MeanReturn' column
    % - cov_matrix: table with asset-to-asset covariance
    % - top_n: number of top assets (by return) to include
    % - allow_short_selling (optional): boolean flag to allow short selling (default: false)
    % - short_bound (optional): minimum weight allowed if short selling is allowed (default: -0.1)
    %
    % Outputs:
    % - FrontierVola: Frontier Volatilities
    % - FrontierRet: Frontier Returns
    % - p: Portfolio frontier

    % Handle optional arguments
    if nargin < 5
        short_bound = -0.1;
    end
    if nargin < 4
        allow_short_selling = false;
    end

    % Step 1: Select top N assets by expected return
    [~, idx] = sort(mean_returns.MeanReturn, 'descend');
    selected_idx = idx(1:min(top_n, length(idx)));
    asset_names = mean_returns.Properties.RowNames(selected_idx);
    
    % Extract mean and covariance for selected assets
    mu = mean_returns.MeanReturn(selected_idx);
    Sigma = table2array(cov_matrix(asset_names, asset_names));
    
    % Step 2: Create Portfolio object
    NumAssets = length(mu);
    p = Portfolio("NumAssets", NumAssets);
    
    % Step 3: Set constraints
    if allow_short_selling
        lower_bounds = short_bound * ones(NumAssets,1);
    else
        lower_bounds = zeros(NumAssets,1);  % no short selling
    end
    upper_bounds = ones(NumAssets,1);
    p = setBounds(p, lower_bounds, upper_bounds);
    p = setBudget(p, 1, 1);  % full investment
    
    % Step 4: Set asset moments (mean and covariance)
    p = setAssetMoments(p, mu, Sigma);
    
    % Step 5: Estimate frontier
    pwgt = estimateFrontier(p, 100);
    [FrontierVola, FrontierRet] = estimatePortMoments(p, pwgt);

    % % Optional: Plot
    % figure;
    % plot(FrontierVola, FrontierRet,'LineWidth', 2);
    % xlabel('Volatility (Std. Dev.)');
    % ylabel('Expected Return');
    % title(sprintf('Efficient Frontier (Top %d Assets)', top_n));
    % grid on;

end
