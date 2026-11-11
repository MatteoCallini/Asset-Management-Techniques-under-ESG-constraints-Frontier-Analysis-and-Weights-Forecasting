% Main
% Callini Matteo - Thesis
clc
clear all
tic

% Load Excel file
excelFile = 'C:/Users/matte/OneDrive/Desktop/Polimi/5/TESI/Matlab/Dati_Tesi.xlsx';

% Read sheet names from Excel file
[~, sheetNames] = xlsfinfo(excelFile);

% Keep only relevant sheets
expectedSheets = {'ESG', 'Prevision_Prices', 'Prices', 'Capitalization', 'Sectors'};
sheetNames = intersect(sheetNames, expectedSheets, 'stable');

% Load data from each sheet into a struct
dataStruct = struct();
for i = 1:length(sheetNames)
    sheet = sheetNames{i};
    dataStruct.(sheet) = readtable(excelFile, 'Sheet', sheet);
end

% Display loaded sheets
disp('Sheets loaded from Excel file:');
disp(fieldnames(dataStruct));

%% Select top N assets by Market Capitalization
top_assets_num = 100;
[filtered_ESG, filtered_Prices] = filter_by_market_cap(excelFile, top_assets_num, 'Prices');

% Create a struct with only ESG and Prices
filteredStruct = struct();
filteredStruct.ESG = filtered_ESG;
filteredStruct.Prices = filtered_Prices;

% Filter and align ESG and Prices data
[ESG_data, Prices_data] = filter_and_align_dataframe(filteredStruct);

%% Compute monthly returns
Prices_returns = compute_price_returns(Prices_data);

% Returns plot
plot_returns(Prices_returns);

%% Winsorization
% Winsorization of order 95% on the returns

alpha = 0.05;
numeric_returns = Prices_returns{:, varfun(@isnumeric, Prices_returns, 'OutputFormat', 'uniform')};
numeric_returns = winsorization(numeric_returns, alpha);
Prices_returns{:, varfun(@isnumeric, Prices_returns, 'OutputFormat', 'uniform')} = numeric_returns;

% Returns plot
plot_returns(Prices_returns);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Select the maximum number of assets to consider for the ESG part
top_n = 50;

%% Compute mean and covariance
[mean_returns, cov_matrix] = compute_mean_and_covariance(Prices_returns);

%% 0) Compute standard efficient frontier 
[topVols_std, topRets_std] = compute_portfolio_frontier(mean_returns,...
                             cov_matrix, size(numeric_returns,1));

%% Plot
figure
plot(topVols_std, topRets_std)
xlabel('Volatility (Std. Dev.)');
ylabel('Expected Return');
title('Efficient Frontier');

%% 0.1) Short Selling
max_short = -0.1; % Maximum quantity that I can short sell

[topVols_std_short, topRets_std_short] = compute_portfolio_frontier(mean_returns,...
                             cov_matrix, size(numeric_returns,1), true, max_short);

%% Plot
figure
plot(topVols_std_short, topRets_std_short)
xlabel('Volatility (Std. Dev.)');
ylabel('Expected Return');
title('Efficient Frontier Short');

% Comparison
figure
plot(topVols_std, topRets_std)
hold on
plot(topVols_std_short, topRets_std_short)
xlabel('Volatility (Std. Dev.)');
ylabel('Expected Return');
legend('Standard','Short')
title('Efficient Frontier Comparison');

%% Set Portfolio type
portfolio_type = 'MinVariance';

%% Set Sheet Name
sheetName = 'Prices';

%% Prepare ESG scores
esg_scores = ESG_data.('ESGScore');  % Adjust if column name differs
company_names = Prices_data.('CompanyCommonName');
esg_scores_table = table(company_names, esg_scores, 'VariableNames', {'Company', 'Score'});
esg_scores_table = rmmissing(esg_scores_table);

% Align with mean_returns (assuming same ordering by company names)
[~, idx] = ismember(mean_returns.Properties.RowNames, esg_scores_table.Company);
esg_scores_aligned = esg_scores_table(:,2);

%% 1) Compute standard efficient frontier considering stocks with the highest ESG score changing the number of stocks to consider
top_n_esg = 10; % Number of stocks with highest ESG score
top_vec = floor(linspace(size(numeric_returns,1),10,top_n_esg));

figure
hold on
for i=1:top_n_esg
    [topVols_std_top_esg_vec(:,i), topRets_std_top_esg_vec(:,i), Portfolio_std_top_esg_vec(i)] = ...
                                compute_portfolio_frontier(mean_returns, cov_matrix, top_vec(i));
    run_performance(Prices_returns, Portfolio_std_top_esg_vec, mean_returns, i, portfolio_type, true, esg_scores_aligned, top_vec(i));
end

%% Plot
figure;
hold on;
legendEntries = cell(1, length(top_vec)); % Preallocate labels
plotHandles = gobjects(1, length(top_vec)); % Preallocate plot handles

for i = 1:length(top_vec)
    plotHandles(i) = plot(topVols_std_top_esg_vec(:, i), topRets_std_top_esg_vec(:, i));
    legendEntries{i} = ['Top ' num2str(top_vec(i)) ' ESG stocks'];
end

legend(plotHandles, legendEntries);
xlabel('Volatility (Std. Dev.)');
ylabel('Expected Return');
title('Efficient Frontier - Top ESG Stocks');
grid on;

%% 1.1) Short selling
figure
hold on
for i=1:length(top_vec)
    [topVols_std_top_esg_vec_short(:,i), topRets_std_top_esg_vec_short(:,i), Portfolio_std_top_esg_vec_short(i)] = compute_portfolio_frontier(...
                                mean_returns, cov_matrix, top_vec(i), true, max_short);
    run_performance(Prices_returns, Portfolio_std_top_esg_vec_short, mean_returns, i, portfolio_type, true,esg_scores_aligned, top_vec(i));
end

%% Plot
figure;
hold on;
for i = 1:length(top_vec)
    plotHandles(i) = plot(topVols_std_top_esg_vec_short(:, i), topRets_std_top_esg_vec_short(:, i));
    legendEntries{i} = ['Top ' num2str(top_vec(i)) ' ESG stocks'];
end
legend(plotHandles, legendEntries);
xlabel('Volatility (Std. Dev.)');
ylabel('Expected Return');
title('Efficient Frontier Short - Top ESG Stocks');
grid on;

%% 2) ESG Integration via Constrained Optimization considering the Ones with Higher ESG scores

% Select different thresholds
% IMPORTANT: empirically observed that there are no differences between the
% one with 0 threshold and the ones with negative threshold

thresholds = linspace(0,max(floor(esg_scores)-1),10);
thresholds = sort(thresholds);

figure
hold on
for i=1:length(thresholds)
    [topVols_esg_constr_vec(:,i), topRets_esg_constr_vec(:,i), Portfolio_esg(i)] = compute_esg_constrained_frontier( ...
                             mean_returns, cov_matrix, esg_scores_aligned,...
                             thresholds(i), top_n, max_short, top_n);
    run_performance(Prices_returns, Portfolio_esg, mean_returns, i, portfolio_type, true, esg_scores_aligned, top_n);
end

%% Plot
figure
hold on
legendEntries = cell(1, length(thresholds)); % Preallocate labels
plotHandles = gobjects(1, length(thresholds)); % Preallocate plot handles
title('Changing threshold (top ESG scores)')
for i = 1:length(thresholds)
    plotHandles(i) = plot(topVols_esg_constr_vec(:,i), topRets_esg_constr_vec(:,i));
    legendEntries{i} = ['Threshold = ' num2str(thresholds(i))]; 
end
legend(plotHandles, legendEntries)

%% 2.2) Short selling
figure
hold on
for i=1:length(thresholds)
    [topVols_esg_constr_vec_short(:,i), topRets_esg_constr_vec_short(:,i), Portfolio_esg_short(i)] = compute_esg_constrained_frontier( ...
                             mean_returns, cov_matrix, esg_scores_aligned,...
                             thresholds(i), top_n, max_short, top_n, true, max_short);
    run_performance(Prices_returns, Portfolio_esg_short, mean_returns, i, portfolio_type, true, esg_scores_aligned, top_n);
end

%% Plot
figure
hold on
title('Changing threshold (top ESG scores)')
for i = 1:length(thresholds)
    plotHandles(i) = plot(topVols_esg_constr_vec_short(:,i), topRets_esg_constr_vec_short(:,i));
    legendEntries{i} = ['Threshold = ' num2str(thresholds(i))]; 
end
legend(plotHandles, legendEntries)

%% 2_alternative) Frontier with ESG constraints that does not consider the top ESG
figure
hold on
for i=1:length(thresholds)
    [topVols_constr_vec(:,i), topRets_constr_vec(:,i), Portfolio(i)] = compute_esg_constrained_frontier( ...
                                 mean_returns, cov_matrix, esg_scores_aligned,...
                                 thresholds(i),size(numeric_returns,1),max_short,size(numeric_returns,1));
    run_performance(Prices_returns, Portfolio, mean_returns, i, portfolio_type, true);
end

%% Plot
figure
hold on
title('Changing threshold')
for i = 1:length(thresholds)
    plotHandles(i) = plot(topVols_constr_vec(:,i), topRets_constr_vec(:,i));
    legendEntries{i} = ['Threshold = ' num2str(thresholds(i))];
end
legend(plotHandles, legendEntries)

%% 2.2_alternative) Short selling
figure
hold on
for i=1:length(thresholds)
    [topVols_constr_vec_short(:,i), topRets_constr_vec_short(:,i), Portfolio_short(i)] = compute_esg_constrained_frontier( ...
                             mean_returns, cov_matrix, esg_scores_aligned,...
                             thresholds(i),size(numeric_returns,1), max_short,...
                             size(numeric_returns,1), true, max_short);
    run_performance(Prices_returns, Portfolio_short, mean_returns, i, portfolio_type, true);
end

%% Plot
figure
hold on
title('Changing threshold')
for i = 1:length(thresholds)
    plotHandles(i) = plot(topVols_constr_vec_short(:,i), topRets_constr_vec_short(:,i));
    legendEntries{i} = ['Threshold = ' num2str(thresholds(i))]; 
end
legend(plotHandles, legendEntries);

%% 6) Sectorial Analysis
% Sector table with only relevant columns
sector_table = dataStruct.Sectors(:, {'CompanyCommonName', 'TRBCBusinessSectorName'});
sector_table = rmmissing(sector_table);

% Initialize empty sector vector
sector_names = strings(height(ESG_data), 1);

% Align sectors based on the names present in ESG_data
[~, idx_sector] = ismember(ESG_data.CompanyCommonName, sector_table.CompanyCommonName);

% Assign the corresponding sectors
sector_names(~isnan(idx_sector)) = sector_table.TRBCBusinessSectorName(idx_sector(~isnan(idx_sector)));

% Create final table aligned with company names
Sector_data = table(ESG_data.CompanyCommonName, sector_names, ...
                    'VariableNames', {'Company', 'Sector'});

%% Count the number of companies for each sector
% Alternatively, if you also want to associate the sector names:
sector_summary = groupsummary(Sector_data, 'Sector');

% Merge ESG and Sector data into a single table
Sector_ESG_table = table(ESG_data.CompanyCommonName, ESG_data.ESGScore, Sector_data.Sector, ...
                         'VariableNames', {'Company', 'ESGScore', 'Sector'});

% Remove rows with missing sectors
Sector_ESG_table = rmmissing(Sector_ESG_table, 'DataVariables', 'Sector');

% Calculate the average ESG by sector
sector_esg_mean = groupsummary(Sector_ESG_table, 'Sector', 'mean', 'ESGScore');

% Rename the column for clarity
sector_esg_mean.Properties.VariableNames{'mean_ESGScore'} = 'Mean_ESGScore';

% Display
disp(sector_esg_mean);

%% Number of sectors to remove
N_worst = 1:10;

% Sort the table of average ESG per sector in ascending order
sorted_sector_mean = sortrows(sector_esg_mean, 'Mean_ESGScore', 'ascend');

figure
hold on
for i=1:length(N_worst)
    % Extract the worst N sectors
    worst_sectors = sorted_sector_mean.Sector(1:N_worst(i));
    
    % Remove companies belonging to the worst sectors from the table
    Sector_ESG_filtered = Sector_ESG_table(~ismember(Sector_ESG_table.Sector, worst_sectors), :);
    
    % (Optional) Display removed sectors
    disp('Sectors removed due to low average ESG score:');
    disp(worst_sectors);
    
    % Get the names of the companies selected after the sector filter
    selected_companies = Sector_ESG_filtered.Company;
    
    % Filter the rows and columns of mean_returns and cov_matrix
    filtered_mean_returns = mean_returns(selected_companies, :);
    filtered_cov_matrix = cov_matrix(selected_companies, selected_companies);
    
    % Number of selected securities
    n_filtered = length(selected_companies);
    
    % Calculate the standard efficient frontier with the filtered data
    [topVols_sector_filtered(:,i), topRets_sector_filtered(:,i), Portfolio_sector(:,i)] = compute_portfolio_frontier(...
                filtered_mean_returns, filtered_cov_matrix, n_filtered);

    run_performance(Prices_returns, Portfolio_sector, filtered_mean_returns, i, portfolio_type, true);
end

%% Plot
figure
hold on
legendEntries = cell(1, length(N_worst)); % Preallocate labels
plotHandles = gobjects(1, length(N_worst)); % Preallocate plot handles
title('Changing threshold')
for i = 1:length(N_worst)
    plotHandles(i) = plot(topVols_sector_filtered(:,i), topRets_sector_filtered(:,i));
    legendEntries{i} = ['N° excluded sectors = ' num2str(N_worst(i))];
end
legend(plotHandles, legendEntries)

%% Short Selling
figure
hold on
for i=1:length(N_worst)
    % Extract the worst N sectors
    worst_sectors = sorted_sector_mean.Sector(1:N_worst(i));
    
    % Remove companies belonging to the worst sectors from the table
    Sector_ESG_filtered = Sector_ESG_table(~ismember(Sector_ESG_table.Sector, worst_sectors), :);
    
    % (Optional) Display removed sectors
    disp('Sectors removed due to low average ESG score:');
    disp(worst_sectors);
    
    % Get the names of the companies selected after the sector filter
    selected_companies = Sector_ESG_filtered.Company;
    
    % Filter the rows and columns of mean_returns and cov_matrix
    filtered_mean_returns = mean_returns(selected_companies, :);
    filtered_cov_matrix = cov_matrix(selected_companies, selected_companies);
    
    % Number of selected securities
    n_filtered = length(selected_companies);
    
    % Calculate the standard efficient frontier with the filtered data
    [topVols_sector_filtered_short(:,i), topRets_sector_filtered_short(:,i), Portfolio_sector_short(:,i)] = compute_portfolio_frontier(...
            filtered_mean_returns, filtered_cov_matrix, n_filtered, true, max_short);

    run_performance(Prices_returns, Portfolio_sector_short, filtered_mean_returns, i, portfolio_type, true);
end

%% Plot
figure
hold on
legendEntries = cell(1, length(N_worst)); % Preallocate labels
plotHandles = gobjects(1, length(N_worst)); % Preallocate plot handles
title('Changing threshold')
for i = 1:length(N_worst)
    plotHandles(i) = plot(topVols_sector_filtered_short(:,i), topRets_sector_filtered_short(:,i));
    legendEntries{i} = ['N° excluded sectors = ' num2str(N_worst(i))];
end
legend(plotHandles, legendEntries)

%% 7) Frontiers with linear constraints considering smaller weights for the worst ones (Top ESG scores)
% with smaller ESG scores
max_low_weight = 0.1;                    % e.g., max 10% of the portfolio in worst ESG assets
num_low_esg = linspace(10,40,4);         % e.g., restrict exposure to worst ESG stocks

threshold = 0;

figure
hold on
for i=1:length(num_low_esg)
    [topVols_std_minsum_esg(:,i), topRets_std_minsum_esg(:,i), Portfolio_minsum_esg(i)]  = compute_esg_constrained_frontier(mean_returns, cov_matrix,...
                          esg_scores_aligned, threshold, top_n, max_low_weight, num_low_esg(i));
    run_performance(Prices_returns, Portfolio_minsum_esg, mean_returns, i, portfolio_type, true, esg_scores_aligned, top_n);
end

%% 7.1) Short selling
figure
hold on
for i=1:length(num_low_esg)
    [topVols_std_minsum_esg_short(:,i), topRets_std_minsum_esg_short(:,i), Portfolio_minsum_esg_short(i)] = compute_esg_constrained_frontier(mean_returns, cov_matrix, ...
                              esg_scores_aligned, threshold, top_n, max_low_weight, num_low_esg(i), true, max_short);
    run_performance(Prices_returns, Portfolio_minsum_esg_short, mean_returns, i, portfolio_type, true, esg_scores_aligned, top_n);
end

%% 7_alternative) Frontiers with linear constraints considering smaller weights for the ones with smaller ESG scores
figure
hold on
for i=1:length(num_low_esg)
    [topVols_std_minsum(:,i), topRets_std_minsum(:,i), Portfolio_minsum(i)] = compute_esg_constrained_frontier(mean_returns, cov_matrix, ...
                              esg_scores_aligned, threshold, size(numeric_returns,1), max_low_weight, num_low_esg(i));
    run_performance(Prices_returns, Portfolio_minsum, mean_returns, i, portfolio_type, true);
end

%% 7.1_alternative) Short selling
figure
hold on
for i=1:length(num_low_esg)
    [topVols_std_minsum_short(:,i), topRets_std_minsum_short(:,i), Portfolio_minsum_short(i)] = compute_esg_constrained_frontier(mean_returns, cov_matrix, ...
                          esg_scores_aligned, threshold, size(numeric_returns,1), max_low_weight, num_low_esg(i), true, max_short);
    run_performance(Prices_returns, Portfolio_minsum_short, mean_returns, i, portfolio_type, true);
end

%% Plot
figure
hold on
title('Low Score - Top 50 ESG')
for i = 1:length(num_low_esg)
    plotHandles(i) = plot(topVols_std_minsum_esg(:,i), topRets_std_minsum_esg(:,i));
    legendEntries{i} = ['N° assets with smaller weights = ' num2str(num_low_esg(i))]; 
end
legend(plotHandles, legendEntries)

%% Plot
figure
hold on
title('Low Score - Top 50 ESG - Short Selling')
for i = 1:length(num_low_esg)
    plotHandles(i) = plot(topVols_std_minsum_esg_short(:,i), topRets_std_minsum_esg_short(:,i));
    legendEntries{i} = ['N° assets with smaller weights = ' num2str(num_low_esg(i))];
end
legend(plotHandles, legendEntries)

%% Plot
figure
hold on
title('Low Score')
for i = 1:length(num_low_esg)
    plotHandles(i) = plot(topVols_std_minsum(:,i), topRets_std_minsum(:,i));
    legendEntries{i} = ['N° assets with smaller weights = ' num2str(num_low_esg(i))];
end
legend(plotHandles, legendEntries)

%% Plot
figure
hold on
legendEntries = cell(1, length(num_low_esg)); % Preallocate labels
plotHandles = gobjects(1, length(num_low_esg)); % Preallocate plot handles
title('Low Score - Short selling')
for i = 1:length(num_low_esg)
    plotHandles(i) = plot(topVols_std_minsum_short(:,i), topRets_std_minsum_short(:,i));
    legendEntries{i} = ['N° assets with smaller weights = ' num2str(num_low_esg(i))];
end
legend(plotHandles, legendEntries)

%% 8) Best in Class ESG Frontier
short_bound_high = 1;    % high ESG assets: no short allowed
short_bound_low  = -0.1; % low ESG assets: forced short up to -10%

figure
hold on
for i=1:length(num_low_esg)
    [topVols_best(:,i), topRets_best(:,i), Portfolio_best(i)] = compute_best_in_class_frontier( ...
                                            mean_returns, cov_matrix, esg_scores_aligned, threshold, ...
                                            size(numeric_returns,1), num_low_esg(i), short_bound_high, short_bound_low);
    run_performance(Prices_returns, Portfolio_best, mean_returns, i, portfolio_type, true);
end

%% 8_alternative) Best in Class ESG Frontier
figure
hold on
for i=1:length(num_low_esg)
    [topVols_best_esg(:,i), topRets_best_esg(:,i), Portfolio_best_esg(i)] = compute_best_in_class_frontier( ...
                                            mean_returns, cov_matrix, esg_scores_aligned, threshold, ...
                                            top_n, num_low_esg(i), short_bound_high, short_bound_low);
    run_performance(Prices_returns, Portfolio_best_esg, mean_returns, i, portfolio_type, true, esg_scores_aligned, top_n);
end

%% Plot
figure
hold on
legendEntries = cell(1, length(num_low_esg)); % Preallocate labels
plotHandles = gobjects(1, length(num_low_esg)); % Preallocate plot handles
title('Best in Class')
for i = 1:length(num_low_esg)
    plotHandles(i) = plot(topVols_best(:,i), topRets_best(:,i));
    legendEntries{i} = ['N° assets with forced short sell = ' num2str(num_low_esg(i))];
end
legend(plotHandles, legendEntries)

%% Plot
figure
hold on
title('Best in Class - Top 50 ESG')
for i = 1:length(num_low_esg)
    plotHandles(i) = plot(topVols_best_esg(:,i), topRets_best_esg(:,i));
    legendEntries{i} = ['N° assets with forced short sell = ' num2str(num_low_esg(i))]; % Cambia stringa secondo necessità
end
legend(plotHandles, legendEntries)

%% Plot
figure;
plot(topVols_esg_constr_vec(:,8), topRets_esg_constr_vec(:,8), 'LineWidth', 1.5); hold on
plot(topVols_esg_constr_vec_short(:,8), topRets_esg_constr_vec_short(:,8), 'LineWidth', 1.5);
plot(topVols_std, topRets_std, 'LineWidth', 1.5);
plot(topVols_std_short, topRets_std_short, 'LineWidth', 1.5);
plot(topVols_constr_vec(:,8), topRets_constr_vec(:,8), 'LineWidth', 1.5);
plot(topVols_constr_vec_short(:,8), topRets_constr_vec_short(:,8), 'LineWidth', 1.5);
plot(topVols_std_minsum(:,1), topRets_std_minsum(:,1), 'LineWidth', 1.5);
plot(topVols_std_minsum_short(:,1), topRets_std_minsum_short(:,1), 'LineWidth', 1.5);
plot(topVols_std_minsum_esg(:,1), topRets_std_minsum_esg(:,1), 'LineWidth', 1.5);
plot(topVols_std_minsum_esg_short(:,1), topRets_std_minsum_esg_short(:,1), 'LineWidth', 1.5);
plot(topVols_sector_filtered, topRets_sector_filtered, 'LineWidth', 1.5);
plot(topVols_sector_filtered_short, topRets_sector_filtered_short, 'LineWidth', 1.5);
plot(topVols_best(:,1), topRets_best(:,1), 'LineWidth', 1.5);
plot(topVols_best_esg(:,1), topRets_best_esg(:,1), 'LineWidth', 1.5);

xlabel('Volatility (Std. Dev.)');
ylabel('Expected Return');
title('Total Comparison');
legend('ESG top frontier - ESG Constraint', ...
       'ESG top frontier - ESG Constraint short', ...
       'Standard Frontier', ...
       'Standard Frontier short', ...
       'Standard frontier - ESG Constraint', ...
       'Standard frontier - ESG Constraint short', ...
       'Standard frontier - ESG Constraint + Sum constr', ...
       'Standard frontier - ESG Constrint + Sum constr short', ...
       'ESG top frontier - ESG Constraint + Sum constr', ...
       'ESG top frontier - ESG Constraint + Sum constr short', ...
       'Standard Frontier - Sector Filtration', ...
       'Standard Frontier - Sector Filtration short', ...
       'Standard Frontier - Best in Class', ...       
       'ESG top frontier - Best in Class');                  
grid on;

%% Top ESG Plot
figure;
plot(topVols_esg_constr_vec(:,8), topRets_esg_constr_vec(:,8), 'LineWidth', 1.5);
hold on
plot(topVols_esg_constr_vec_short(:,8), topRets_esg_constr_vec_short(:,8), 'LineWidth', 1.5);
plot(topVols_std_minsum_esg(:,1), topRets_std_minsum_esg(:,1), 'LineWidth', 1.5);
plot(topVols_std_minsum_esg_short(:,1), topRets_std_minsum_esg_short(:,1), 'LineWidth', 1.5);
plot(topVols_best_esg(:,1), topRets_best_esg(:,1), 'LineWidth', 1.5);

xlabel('Volatility (Std. Dev.)');
ylabel('Expected Return');
title('Total Comparison - Top ESG');
legend('ESG top frontier - ESG Constraint', ...
       'ESG top frontier - ESG Constraint short', ...
       'ESG top frontier - ESG Constrint + Sum constr',...
       'ESG top frontier - ESG Constrint + Sum constr short',...
       'ESG top frontier - Best in Class');
grid on;

%% Standard Frontier Plot
figure;
plot(topVols_std, topRets_std, 'LineWidth', 1.5);
hold on
plot(topVols_std_short, topRets_std_short, 'LineWidth', 1.5);
plot(topVols_constr_vec(:,8), topRets_constr_vec(:,8), 'LineWidth', 1.5);
plot(topVols_constr_vec_short(:,8), topRets_constr_vec_short(:,8), 'LineWidth', 1.5);
plot(topVols_std_minsum(:,1), topRets_std_minsum(:,1), 'LineWidth', 1.5);
plot(topVols_std_minsum_short(:,1), topRets_std_minsum_short(:,1), 'LineWidth', 1.5);
plot(topVols_best(:,1), topRets_best(:,1), 'LineWidth', 1.5);

xlabel('Volatility (Std. Dev.)');
ylabel('Expected Return');
title('Total Comparison - Standard');
legend('Standard Frontier',...
       'Standard Frontier short',...
       'Standard frontier - ESG Constraint',...
       'Standard frontier - ESG Constraint short',...
       'Standard frontier - ESG Constraint + Sum constr',...
       'Standard frontier - ESG Constraint + Sum constr short',...
       'Standard frontier - Best in Class');
grid on;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Kalman Filter Prevision Part
years = 3; % years to consider as "training part" of the Kalman filter

%% Select top N assets by Market Capitalization
top_assets_num = 100;
[Prevision_filtered_ESG, Prevision_filtered_Prices] = filter_by_market_cap(excelFile, top_assets_num, 'Prevision_Prices');

% Create a struct with only ESG and Prices
Prevision_filteredStruct = struct();
Prevision_filteredStruct.ESG = Prevision_filtered_ESG;
Prevision_filteredStruct.Prices = Prevision_filtered_Prices;

% Filter and align ESG and Prices data
[Prevision_ESG_data, Prevision_Prices_data] = filter_and_align_dataframe(Prevision_filteredStruct);

%% Compute monthly returns
Prevision_Prices_returns = compute_price_returns(Prevision_Prices_data);

%% Best in Class Frontier
disp('\bf Best in Class Frontier Metrics')
for i=1:length(num_low_esg)
    [prev_returns_best, prev_cum_best, metrics_best, kalman_weights_best] = run_prevision_rolling_best_mv( ...
                        Prevision_Prices_returns, i, years, num_low_esg, ...
                        esg_scores_aligned, short_bound_high, short_bound_low, threshold, true);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Smaller Weights for worst ESG Scores Frontier
disp('\bf Lower weights Frontier Metrics')
for i=1:length(num_low_esg)
    [prev_returns_minsum, prev_cum_minsum, metrics_minsum, kalman_weights_minsum] = run_prevision_rolling_minsum_mv( ...
                        Prevision_Prices_returns, i, years, num_low_esg, ...
                        esg_scores_aligned, max_low_weight, max_short, threshold, true, false);
end

% %% Smaller Weights for worst ESG Scores Frontier - Short selling
% disp('\bf Lower weights Frontier - Short selling Metrics')
% for i=1:length(num_low_esg)
%     [prev_returns_minsum_short, prev_cum_minsum_short, metrics_minsum_short, kalman_weights_minsum_short]  = run_prevision_rolling_minsum_mv( ...
%                         Prevision_Prices_returns, i, years, num_low_esg, ...
%                         esg_scores_aligned, max_low_weight, max_short, threshold, true, true);
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Screening Frontier
disp('\bf Sector Screening Metrics')
for i=1:length(N_worst)
    [prev_returns_sector, prev_cum_sector, metrics_sector, kalman_weights_sector] = run_prevision_rolling_sector_mv( ...
                        Prevision_Prices_returns, i, years, esg_scores_aligned, sorted_sector_mean, Sector_ESG_table, N_worst, max_short, true, false);
end

% %% Screening Frontier - Short selling
% disp('\bf Sector Screening - Short selling Metrics')
% for i=1:length(N_worst)
%     [prev_returns_sector_short, prev_cum_sector_short, metrics_sector_short, kalman_weights_sector_short] = run_prevision_rolling_sector_mv( ...
%                         Prevision_Prices_returns, i, years, esg_scores_aligned, sorted_sector_mean, Sector_ESG_table, N_worst, max_short, true, true);
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Constrained Frontier
disp('\bf Constrained Frontier Metrics')
for i=1:length(thresholds)
    [prev_returns_const, prev_cum_const, metrics_const, kalman_weights_const] = run_prevision_rolling_const_mv( ...
                        Prevision_Prices_returns, i, years, thresholds, ...
                        esg_scores_aligned, max_short, true, false);
end

% %% Constrained Frontier - Short selling
% disp('\bf Constrained Frontier - Short selling Metrics')
% for i=1:length(thresholds)
%     [prev_returns_const_short, prev_cum_const_short, metrics_const_short, kalman_weights_const_short] = run_prevision_rolling_const_mv( ...
%                         Prevision_Prices_returns, i, years, thresholds, ...
%                         esg_scores_aligned, max_short, true, true);
% end

%%
disp("Total computational time: ");
toc
