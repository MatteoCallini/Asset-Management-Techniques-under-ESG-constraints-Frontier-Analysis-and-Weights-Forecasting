function [filtered_ESG, filtered_Prices] = filter_by_market_cap(excelFile,N,sheet_name)

    % This function removes NaN, fix data format and returns the top N 
    % companies with respect to their capitalization

    % Load the Capitalization sheet
    cap_data = readtable(excelFile, 'Sheet', 'Capitalization');

    % Try to find useful column names
    cap_col_name = 'CompanyMarketCap';
    name_col_name = 'CompanyCommonName';

    % Convert to string in case it's not text
    cap_strings = string(cap_data.(cap_col_name));

    % Remove rows with "Unable to collect data"
    valid_rows = ~contains(cap_strings, "Unable");
    cap_data = cap_data(valid_rows, :);
    cap_strings = cap_strings(valid_rows);

    % Clean: remove commas, convert to numbers
    cap_cleaned = strrep(cap_strings, ',', '');
    cap_numeric = str2double(cap_cleaned);

    % Remove rows with NaNs
    valid = ~isnan(cap_numeric);
    cap_data = cap_data(valid, :);
    cap_numeric = cap_numeric(valid);

    % Add numeric market cap values to table
    cap_data.CapNumeric = cap_numeric;

    % Sort by market cap descending
    cap_data = sortrows(cap_data, 'CapNumeric', 'descend');

    % Select top N companies
    top_companies = cap_data.(name_col_name)(1:min(N, height(cap_data)));

    % Load ESG and Prices data
    ESG = readtable(excelFile, 'Sheet', 'ESG');
    Prices = readtable(excelFile, 'Sheet', sheet_name);

    % Filter ESG and Prices by top company names
    filtered_ESG = ESG(ismember(ESG.('CompanyCommonName'), top_companies), :);
    filtered_Prices = Prices(ismember(Prices.('CompanyCommonName'), top_companies), :);
end
