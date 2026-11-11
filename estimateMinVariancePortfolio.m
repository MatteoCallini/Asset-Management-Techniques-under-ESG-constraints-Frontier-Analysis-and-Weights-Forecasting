function [weights, risk] = estimateMinVariancePortfolio(Portfolio)
    % This function estimates the minimum variance portfolio.
    % Portfolio: a matrix where each column is the return of an asset in each period.

    % Calculate the covariance matrix of the portfolio returns
    covariance_matrix = cov(Portfolio);

    % Calculate the inverse of the covariance matrix
    inv_cov_matrix = inv(covariance_matrix);

    % Calculate the number of assets
    n_assets = size(Portfolio, 2);

    % Create a vector of ones
    ones_vector = ones(n_assets, 1);

    % Calculate the minimum variance portfolio weights
    weights = inv_cov_matrix * ones_vector / (ones_vector' * inv_cov_matrix * ones_vector);

    % Calculate the risk (standard deviation) of the minimum variance portfolio
    risk = sqrt(weights' * covariance_matrix * weights);
end
