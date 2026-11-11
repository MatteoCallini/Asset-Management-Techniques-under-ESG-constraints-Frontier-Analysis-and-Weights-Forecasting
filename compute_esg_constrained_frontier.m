function [FrontierVola, FrontierRet, p] = compute_esg_constrained_frontier(mean_returns, cov_matrix,...
                                                esg_scores, threshold, top_n, max_low_weight, num_low_esg,...
                                                allow_short_selling, short_bound)

    % Computes efficient frontier with ESG constraints:
    % - weighted ESG >= threshold
    % - total weight on N lowest ESG assets <= max_low_weight
    % - optionally allows short-selling with lower bound

    % Handle optional arguments
    if nargin < 9
        short_bound = -0.1;
    end
    if nargin < 8
        allow_short_selling = false;
    end

    % Step 1: Select top N assets by ESG score
    esg_array = table2array(esg_scores);
    [~, idx] = sort(esg_array, 'descend');
    selected_idx = idx(1:min(top_n, length(idx)));

    asset_names = mean_returns.Properties.RowNames(selected_idx);
    mu = mean_returns.MeanReturn(selected_idx);
    Sigma = table2array(cov_matrix(asset_names, asset_names));
    ESG = esg_array(selected_idx);

    % Step 2: Create Portfolio object
    NumAssets = length(mu);
    p = Portfolio("NumAssets", NumAssets);
    if allow_short_selling
        lower_bounds = short_bound * ones(NumAssets, 1);
    else
        lower_bounds = zeros(NumAssets, 1);
    end
    upper_bounds = ones(NumAssets, 1);
    p = setBounds(p, lower_bounds, upper_bounds);  % With or without short-selling
    p = setBudget(p, 1, 1);                         % Fully invested
    p = setAssetMoments(p, mu, Sigma);             % Mean and covariance

    % Step 3: ESG constraint: ESG' * w >= threshold
    Aesg = -ESG(:)';
    besg = -threshold;
    p = addInequality(p, Aesg, besg);
    
    if top_n ~= num_low_esg
        % Step 4: Limit exposure to lowest ESG assets
        [~, low_idx] = mink(ESG, num_low_esg);  % indices of N worst ESG scores
        A_low = zeros(1, NumAssets);
        A_low(low_idx) = 1;                     % sum of weights for these indices
        b_low = max_low_weight;
        p = addInequality(p, A_low, b_low);
    end

    % Step 5: Estimate frontier
    pwgt = estimateFrontier(p, 100);
    [FrontierVola, FrontierRet] = estimatePortMoments(p, pwgt);

end
