function [mean_returns, cov_matrix] = compute_mean_and_covariance(prices_returns_df)

    % Computes mean return per asset and the covariance matrix
    % Assumes:
    % - First two columns are metadata (e.g., 'Company Common Name', ticker)
    % - Remaining columns are percentage returns

    % Step 1: Extract return columns
    returns_table = prices_returns_df(:, 3:end);
    returns_matrix = table2array(returns_table);

    % Step 2: Compute mean returns across time (per row/asset)
    mean_values = mean(returns_matrix, 2);  % Mean across columns (time)
    
    % Step 3: Extract company names
    company_names = prices_returns_df.CompanyCommonName;

    % Step 3.1: Make names unique
    unique_names = matlab.lang.makeUniqueStrings(cellstr(company_names));

    % Step 4: Build mean_returns as a table with row names
    mean_returns = array2table(mean_values, ...
        'VariableNames', {'MeanReturn'}, ...
        'RowNames', unique_names);

    % Step 5: Compute covariance matrix across assets (transpose matrix)
    cov_matrix = cov(returns_matrix');

    % Assign row and column names
    cov_matrix = array2table(cov_matrix, ...
        'VariableNames', unique_names, ...
        'RowNames', unique_names);
end
