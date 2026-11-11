function prices_returns = compute_price_returns(prices_df)

    % Compute percentage returns from a Prices_data table
    % Assumes:
    % - First two columns are metadata (e.g., company name and ticker)
    % - Remaining columns are price values from most recent to oldest (left to right)
    % Output:
    % - prices_returns: it contains the monthly returns, ordered from the 
    %                   oldest ones (left) to the closest ones (right)

    % Step 1: Identify metadata and price columns
    meta_data = prices_df(:, 1:2);
    price_data = prices_df(:, 3:end);

    % Step 2: Reverse price columns (oldest to newest)
    reversed_prices = price_data(:, end:-1:1);

    % Step 2.1: Ensure all values are numeric (convert to array)
    price_array = table2array(reversed_prices);
    price_array = double(price_array);  % Convert to double if needed

    % Step 3: Compute percentage returns across columns (time axis)
    returns_array = diff(price_array, 1, 2) ./ price_array(:, 1:end-1);

    % Step 4: Rename return columns
    return_columns = cell(1, size(returns_array, 2));
    for i = 1:length(return_columns)
        return_columns{i} = sprintf('R%d', i);
    end

    % Step 5: Combine metadata and returns into one table
    returns_table = array2table(returns_array, 'VariableNames', return_columns);
    prices_returns = [meta_data, returns_table];

end
