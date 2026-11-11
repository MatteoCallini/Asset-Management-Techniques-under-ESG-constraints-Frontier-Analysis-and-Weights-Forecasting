function [ESG_data, Prices_data] = filter_and_align_dataframe(dataStruct, common_column)
    % Filters and aligns ESG and Prices tables based on a common column
    % Removes rows where all time-series values are constant
    % Also removes rows where any value is constant for at least 12 consecutive periods
    % Additionally, normalizes ESG scores by removing the cross-sectional mean

    if nargin < 2
        common_column = 'CompanyCommonName';
    end

    sheets = fieldnames(dataStruct);
    if length(sheets) ~= 2
        error('Expected exactly 2 sheets in the dataStruct.');
    end

    % Extract tables
    sheet1 = dataStruct.(sheets{1});
    sheet2 = dataStruct.(sheets{2});

    % Step 1: Find common company names
    common_names = intersect(sheet1.(common_column), sheet2.(common_column));

    % Step 2: Filter both tables by common company names
    ESG_data_raw = sheet1(ismember(sheet1.(common_column), common_names), :);
    Prices_data_raw = sheet2(ismember(sheet2.(common_column), common_names), :);

    % Step 3: Sort tables by company name
    ESG_data_raw = sortrows(ESG_data_raw, common_column);
    Prices_data_raw = sortrows(Prices_data_raw, common_column);

    % Step 4: Identify time-series columns (assuming the first two columns are metadata)
    time_columns = 3:width(Prices_data_raw);

    % Step 5: Remove rows with constant values or 12+ repeated identical values
    mask_valid = false(height(Prices_data_raw), 1);
    for i = 1:height(Prices_data_raw)
        row_vals = table2array(Prices_data_raw(i, time_columns));
        row_vals = row_vals(~isnan(row_vals));  % remove NaNs

        % Check if entire row is constant
        if numel(unique(row_vals)) <= 1
            continue;
        end

        % Check for 12+ consecutive repeated values
        repeated_count = 1;
        max_repeat = 1;
        for j = 2:length(row_vals)
            if row_vals(j) == row_vals(j-1)
                repeated_count = repeated_count + 1;
                max_repeat = max(max_repeat, repeated_count);
            else
                repeated_count = 1;
            end
        end

        % Keep only if max_repeat < 12
        if max_repeat < 12
            mask_valid(i) = true;
        end
    end

    % Step 6: Apply the final mask to both tables
    ESG_data = ESG_data_raw(mask_valid, :);
    Prices_data = Prices_data_raw(mask_valid, :);

    % Step 7: Normalize ESG values (columns 3:end) by removing cross-sectional mean
    ESG_values = table2array(ESG_data(:, 3:end));
    mean_vector = mean(ESG_values, 1, 'omitnan'); % mean for each column
    ESG_values_normalized = ESG_values - repmat(mean_vector, size(ESG_values, 1), 1);
    ESG_data(:, 3:end) = array2table(ESG_values_normalized, 'VariableNames', ESG_data.Properties.VariableNames(3:end));

    fprintf('Filtered %d aligned rows with common "%s" and clean price time series.\n', ...
        height(ESG_data), common_column);
end
