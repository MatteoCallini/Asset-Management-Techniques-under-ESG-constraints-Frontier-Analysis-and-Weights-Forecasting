function [FrontierVola, FrontierRet, p] = compute_esg_utility_frontier(mean_returns, cov_matrix, esg_scores, top_n, esg_weight)

    % The function computes the efficient frontier where expected return 
    % includes an ESG utility preference
    %
    % Inputs:
    % - mean_returns: table with 'MeanReturn' column and asset names as row names
    % - cov_matrix: covariance matrix as table
    % - esg_scores: table with 'ESG' column and same row names
    % - top_n: number of top ESG assets to consider
    % - esg_weight: scalar ESG preference multiplier in the utility function
    %
    % Outputs:
    % - FrontierVola: Frontier Volatilities
    % - FrontierRet: Frontier Returns
    % - p: Portffolio Frontier

    if nargin < 5
        esg_weight = 0.1;
    end

    % Step 1: Select top N assets by ESG score
    esg_array = table2array(esg_scores);
    [~, idx] = sort(esg_array, 'descend');
    selected_idx = idx(1:min(top_n, length(idx)));

    asset_names = mean_returns.Properties.RowNames(selected_idx);
    mu = mean_returns.MeanReturn(selected_idx);
    Sigma = table2array(cov_matrix(asset_names, asset_names));
    ESG = table2array(esg_scores(selected_idx, :));

    % Step 2: Create utility vector as ESG-augmented expected return
    utility_mu = mu + esg_weight * ESG;

    % Step 3: Create Portfolio object with utility as expected return
    NumAssets = length(utility_mu);
    p = Portfolio("NumAssets", NumAssets);
    p = setBounds(p, zeros(NumAssets,1), ones(NumAssets,1));
    p = setBudget(p, 1, 1);
    p = setAssetMoments(p, utility_mu, Sigma);  % Use utility-based returns

    % Step 4: Compute efficient frontier
    pwgt = estimateFrontier(p, 100);
    [FrontierVola, FrontierRet] = estimatePortMoments(p, pwgt);

    FrontierRet = FrontierRet - (esg_weight * ESG' * pwgt)';

    % % Optional: Plot
    % figure;
    % plot(FrontierVola, FrontierRet, 'LineWidth', 2);
    % ylabel('ESG-Utility Return');
    % xlabel('Volatility (Std. Dev.)');
    % title(sprintf('Efficient Frontier using ESG Utility (Top %d Assets)', top_n));
    % grid on;
end
