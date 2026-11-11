function [FrontierVola, FrontierRet, p] = compute_best_in_class_frontier(mean_returns, cov_matrix, ...
                                        esg_scores, threshold, top_n, num_low_esg, ...
                                        short_bound_high, short_bound_low)

    % Computes the efficient frontier under "Best in Class" ESG constraints:
    % - High-ESG assets (top_n) must have non-negative weights
    % - Low-ESG assets (num_low_esg) must have negative weights (short positions)
    % - Weighted average ESG >= threshold

    % === Step 1: Sort assets by ESG score ===
    esg_array = table2array(esg_scores);
    [~, idx] = sort(esg_array, 'descend');
    selected_idx = idx(1:min(top_n, length(idx)));

    asset_names = mean_returns.Properties.RowNames(selected_idx);
    mu = mean_returns.MeanReturn(selected_idx);
    Sigma = table2array(cov_matrix(asset_names, asset_names));
    ESG = esg_array(selected_idx);

    NumAssets = length(mu);

    % === Step 2: Create Portfolio object ===
    p = Portfolio("NumAssets", NumAssets);
    p = setBudget(p, 1, 1);           % fully invested
    p = setAssetMoments(p, mu, Sigma);

    % === Step 3: ESG constraint ===
    % (portfolio ESG weighted average >= threshold)
    Aesg = -ESG(:)';
    besg = -threshold;
    p = addInequality(p, Aesg, besg);

    % === Step 4: Best-in-Class constraints ===
    [~, high_idx] = maxk(ESG, top_n);         % top ESG
    [~, low_idx]  = mink(ESG, num_low_esg);   % bottom ESG

    lower_bounds = -ones(NumAssets,1);  % default: free to short/long
    upper_bounds =  ones(NumAssets,1);

    % High ESG → long-only (≥ 0)
    lower_bounds(high_idx) = 0;
    upper_bounds(high_idx) = short_bound_high;

    % Low ESG → short-only (≤ short_bound_low, e.g., -0.1)
    lower_bounds(low_idx) = short_bound_low;
    upper_bounds(low_idx) = 0;

    % Apply bounds
    p = setBounds(p, lower_bounds, upper_bounds);

    % === Step 5: Compute frontier ===
    pwgt = estimateFrontier(p, 100);
    [FrontierVola, FrontierRet] = estimatePortMoments(p, pwgt);

end
