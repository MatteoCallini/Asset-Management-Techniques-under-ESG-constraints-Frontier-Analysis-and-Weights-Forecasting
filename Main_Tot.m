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
expectedSheets = {'ESG', 'Backtesting_Prices', 'Prices', 'Capitalization', 'Sectors'};
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
[filtered_ESG, filtered_Prices] = filter_by_market_cap(excelFile, top_assets_num);

% Ricrea una struct con solo ESG e Prices per la funzione di allineamento
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
                             cov_matrix,size(numeric_returns,1));

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

%% 1) Compute standard efficient frontier considering stocks with the highest ESG score changing the number of stocks to consider
top_n_esg = 10; % Number of stocks with highest ESG score
top_vec = floor(linspace(size(numeric_returns,1),10,top_n_esg));
for i=1:top_n_esg
    [topVols_std_top_esg_vec(:,i), topRets_std_top_esg_vec(:,i), Portfolio_std_top_esg_vec(i)] = ...
                                compute_portfolio_frontier(mean_returns, cov_matrix, top_vec(i));
end

%% Plot
figure;
hold on;
legendEntries = cell(1, length(top_vec)); % Prealloca etichette
plotHandles = gobjects(1, length(top_vec)); % Prealloca handle grafici

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
for i=1:length(top_vec)
    [topVols_std_top_esg_vec_short(:,i), topRets_std_top_esg_vec_short(:,i), Portfolio_std_top_esg_vec_short(i)] = compute_portfolio_frontier(...
                                mean_returns, cov_matrix, top_vec(i), true, max_short);
end

%% Plot
figure;
hold on;
legendEntries = cell(1, length(top_vec)); % Prealloca etichette
plotHandles = gobjects(1, length(top_vec)); % Prealloca handle grafici

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
% Prepare ESG scores
esg_scores = ESG_data.('ESGScore');  % Adjust if column name differs
company_names = Prices_data.('CompanyCommonName');
esg_scores_table = table(company_names, esg_scores, 'VariableNames', {'Company', 'Score'});
esg_scores_table = rmmissing(esg_scores_table);

% Align with mean_returns (assuming same ordering by company names)
[~, idx] = ismember(mean_returns.Properties.RowNames, esg_scores_table.Company);
esg_scores_aligned = esg_scores_table(:,2);

%% 2.1) Select different thresholds
% thresholds = linspace(min(floor(esg_scores)+1),max(floor(esg_scores)-1),10);
% thresholds(end) = 0;

% IMPORTANT: empirically observed that there are no differences between the
% one with 0 threshold and the ones with negative threshold

thresholds = linspace(0,max(floor(esg_scores)-1),10);
thresholds = sort(thresholds);

for i=1:length(thresholds)
[topVols_esg_constr_vec(:,i), topRets_esg_constr_vec(:,i), Portfolio_esg(i)] = compute_esg_constrained_frontier( ...
                             mean_returns, cov_matrix, esg_scores_aligned,...
                             thresholds(i), top_n, max_short, top_n);
end

%% Plot
figure
hold on
legendEntries = cell(1, length(thresholds)); % Prealloca cella per le etichette
plotHandles = gobjects(1, length(thresholds)); % Prealloca array di handle
title('Changing threshold (top ESG scores)')
for i = 1:length(thresholds)
    plotHandles(i) = plot(topVols_esg_constr_vec(:,i), topRets_esg_constr_vec(:,i));
    legendEntries{i} = ['Soglia = ' num2str(thresholds(i))]; % Cambia stringa secondo necessità
end
legend(plotHandles, legendEntries)

%% 2.2) Short selling
% Some small differences can be observed if I take negative values in the
% short seeling situation

for i=1:length(thresholds)
[topVols_esg_constr_vec_short(:,i), topRets_esg_constr_vec_short(:,i), Portfolio_esg_short(i)] = compute_esg_constrained_frontier( ...
                             mean_returns, cov_matrix, esg_scores_aligned,...
                             thresholds(i), top_n, max_short, top_n, true, max_short);
end

%% Plot
figure
hold on
legendEntries = cell(1, length(thresholds)); % Prealloca cella per le etichette
plotHandles = gobjects(1, length(thresholds)); % Prealloca array di handle
title('Changing threshold (top ESG scores)')
for i = 1:length(thresholds)
    plotHandles(i) = plot(topVols_esg_constr_vec_short(:,i), topRets_esg_constr_vec_short(:,i));
    legendEntries{i} = ['Soglia = ' num2str(thresholds(i))]; % Cambia stringa secondo necessità
end
legend(plotHandles, legendEntries)

%% 2_alternative) Frontier with ESG constraints that does not consider the top ESG
for i=1:length(thresholds)
[topVols_constr_vec(:,i), topRets_constr_vec(:,i), Portfolio(i)] = compute_esg_constrained_frontier( ...
                             mean_returns, cov_matrix, esg_scores_aligned,...
                             thresholds(i),size(numeric_returns,1),max_short,size(numeric_returns,1));
end

%% Plot
figure
hold on
legendEntries = cell(1, length(thresholds)); % Prealloca cella per le etichette
plotHandles = gobjects(1, length(thresholds)); % Prealloca array di handle
title('Changing threshold')
for i = 1:length(thresholds)
    plotHandles(i) = plot(topVols_constr_vec(:,i), topRets_constr_vec(:,i));
    legendEntries{i} = ['Soglia = ' num2str(thresholds(i))]; % Cambia stringa secondo necessità
end
legend(plotHandles, legendEntries)

%% 2.2_alternative) Short selling
for i=1:length(thresholds)
[topVols_constr_vec_short(:,i), topRets_constr_vec_short(:,i), Portfolio_short(i)] = compute_esg_constrained_frontier( ...
                             mean_returns, cov_matrix, esg_scores_aligned,...
                             thresholds(i),size(numeric_returns,1), max_short,...
                             size(numeric_returns,1), true, max_short);
end

%% Plot
figure
hold on
legendEntries = cell(1, length(thresholds)); % Prealloca cella per le etichette
plotHandles = gobjects(1, length(thresholds)); % Prealloca array di handle
title('Changing threshold')
for i = 1:length(thresholds)
    plotHandles(i) = plot(topVols_constr_vec_short(:,i), topRets_constr_vec_short(:,i));
    legendEntries{i} = ['Soglia = ' num2str(thresholds(i))]; % Cambia stringa secondo necessità
end
legend(plotHandles, legendEntries)

%% 3) ESG score in the minimization function (Top ESG scores) (NON CONSIDERO, ESCE MALE)
% Select different weights
esg_weights = linspace(-1,1,15);

for i=1:length(esg_weights)
[topVols_esg_uti_vec(:,i), topRets_esg_uti_vec(:,i), Portfolio_esg_uti(i)] = compute_esg_utility_frontier(mean_returns,...
                                    cov_matrix, esg_scores_aligned, top_n, esg_weights(i));
end

%% Plot
figure
hold on
legendEntries = cell(1, length(esg_weights)); % Prealloca cella per le etichette
plotHandles = gobjects(1, length(esg_weights)); % Prealloca array di handle
title('Changing weights (top ESG scores)')
for i = 1:length(esg_weights)
    plotHandles(i) = plot(topVols_esg_uti_vec(:,i), topRets_esg_uti_vec(:,i));
    legendEntries{i} = ['Soglia = ' num2str(esg_weights(i))]; % Cambia stringa secondo necessità
end

legend(plotHandles, legendEntries)

%% 3_alternative) ESG score in the minimization function (Top ESG scores)
% Select different weights (select symmetric intervals and odds number in order to consider 0 in the analysis)
esg_weights = linspace(-1,1,15);

for i=1:length(esg_weights)
[topVols_uti_vec(:,i), topRets_uti_vec(:,i), Portfolio_uti(i)] = compute_esg_utility_frontier(mean_returns,...
                                    cov_matrix, esg_scores_aligned, size(numeric_returns,1), esg_weights(i));
end

%% Plot
figure
hold on
legendEntries = cell(1, length(esg_weights)); % Prealloca cella per le etichette
plotHandles = gobjects(1, length(esg_weights)); % Prealloca array di handle
title('Changing weights')
for i = 1:length(esg_weights)
    plotHandles(i) = plot(topVols_uti_vec(:,i), topRets_uti_vec(:,i));
    legendEntries{i} = ['Soglia = ' num2str(esg_weights(i))]; % Cambia stringa secondo necessità
end

legend(plotHandles, legendEntries)

%% 4) Tracking error minimization (Top ESG scores)
[benchmark_weights, ~] = estimateFrontierLimits(Portfolio_esg(8)); % >= 0 constraint
% [benchmark_weights, ~] = estimateMaxSharpeRatio(Portfolio_esg(8), 0);
threshold = 0;

[~, FrontierVols_track_esg, FrontierRet_track_esg, Portfolio_tracking_esg] = compute_tracking_error_esg_frontier(mean_returns,...
                    cov_matrix, esg_scores_aligned, benchmark_weights, threshold, top_n);

%% 4.1) Short selling
[~, FrontierVols_track_esg_short, FrontierRet_track_esg_short, Portfolio_tracking_esg_short] = compute_tracking_error_esg_frontier(mean_returns,...
                    cov_matrix, esg_scores_aligned, benchmark_weights, threshold, top_n,...
                    true, max_short);

%% Plot
figure
plot(FrontierVols_track_esg_short, FrontierRet_track_esg_short, 'LineWidth', 1.5);
hold on
plot(FrontierVols_track_esg, FrontierRet_track_esg, 'LineWidth', 1.5);
xlabel('Volatility (Std. Dev.)');
ylabel('Expected Return');
title('Tracking error - Top 50 ESG');
legend('Short','Standard')
grid on;

%% 4_alternative) Tracking error minimization
[benchmark_weights, ~] = estimateFrontierLimits(Portfolio(8)); % >= 0 constraint
% [benchmark_weights, ~] = estimateMaxSharpeRatio(Portfolio(8), 0);

[~, FrontierVols_track, FrontierRet_track, Portfolio_tracking] = compute_tracking_error_esg_frontier(mean_returns,...
                    cov_matrix, esg_scores_aligned, benchmark_weights, threshold, size(numeric_returns,1));

%% 4.1_alternative) Short selling
[~, FrontierVols_track_short, FrontierRet_track_short, Portfolio_tracking_short] = compute_tracking_error_esg_frontier(mean_returns,...
                    cov_matrix, esg_scores_aligned, benchmark_weights, threshold, size(numeric_returns,1),...
                    true, max_short);

%% Plot
figure
plot(FrontierVols_track_short, FrontierRet_track_short, 'LineWidth', 1.5);
hold on
plot(FrontierVols_track, FrontierRet_track, 'LineWidth', 1.5);
xlabel('Volatility (Std. Dev.)');
ylabel('Expected Return');
title('Tracking error');
legend('Short','Standard')
grid on;

%% 5) Frontier using Regression (FUNZIONA MALE)
for i = 1:length(FrontierRet_track)
    [Portfolio_reg(i,:), FrontierRet_reg(i), ~, FrontierVols_reg(i)] = compute_varmaz_weights(mean_returns,...
                                    cov_matrix, esg_scores_aligned, FrontierRet_track(i), threshold);
end

%%
figure
plot(FrontierVols_reg, FrontierRet_reg)
title('Regression Frontier')

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
N_worst = 7;

% Sort the table of average ESG per sector in ascending order
sorted_sector_mean = sortrows(sector_esg_mean, 'Mean_ESGScore', 'ascend');

% Extract the worst N sectors
worst_sectors = sorted_sector_mean.Sector(1:N_worst);

% Remove companies belonging to the worst sectors from the table
Sector_ESG_filtered = Sector_ESG_table(~ismember(Sector_ESG_table.Sector, worst_sectors), :);

% (Optional) Display removed sectors
disp('Sectors removed due to low average ESG score:');
disp(worst_sectors);

%% Get the names of the companies selected after the sector filter
selected_companies = Sector_ESG_filtered.Company;

% Filter the rows and columns of mean_returns and cov_matrix
filtered_mean_returns = mean_returns(selected_companies, :);
filtered_cov_matrix = cov_matrix(selected_companies, selected_companies);

% Number of selected securities
n_filtered = length(selected_companies);

%% Calculate the standard efficient frontier with the filtered data
[topVols_sector_filtered, topRets_sector_filtered, Portfolio_sector] = compute_portfolio_frontier(...
            filtered_mean_returns, filtered_cov_matrix, n_filtered);

%% Plot
figure
plot(topVols_sector_filtered, topRets_sector_filtered, 'LineWidth', 1.5);
xlabel('Volatility (Std. Dev.)');
ylabel('Expected Return');
title('Sectorial Analysis');
grid on;

%% Short Selling
[topVols_sector_filtered_short, topRets_sector_filtered_short, Portfolio_sector_short] = compute_portfolio_frontier(...
            filtered_mean_returns, filtered_cov_matrix, n_filtered, true, max_short);

%% Plot
figure
plot(topVols_sector_filtered_short, topRets_sector_filtered_short, 'LineWidth', 1.5);
xlabel('Volatility (Std. Dev.)');
ylabel('Expected Return');
title('Sectorial Analysis - Short selling');
grid on;

%% 7) Frontiers with linear constraints considering smaller weights for the worst ones (Top ESG scores)
% with smaller ESG scores
max_low_weight = 0.1;     % e.g., max 10% of the portfolio in worst ESG assets
num_low_esg = 20;         % e.g., restrict exposure to 20 worst ESG stocks

[topVols_std_esg_minsum, topRets_std_esg_minsum, Portfolio_minsum_esg]  = compute_esg_constrained_frontier(mean_returns, cov_matrix, ...
                          esg_scores_aligned, threshold, top_n, max_low_weight, num_low_esg);

%% 7.1) Short selling
[topVols_std_esg_minsum_short, topRets_std_esg_minsum_short, Portfolio_minsum_esg_short] = compute_esg_constrained_frontier(mean_returns, cov_matrix, ...
                          esg_scores_aligned, threshold, top_n, max_low_weight, num_low_esg, true, max_short);

%% 7_alternative) Frontiers with linear constraints considering smaller weights for the ones with smaller ESG scores
[topVols_std_minsum, topRets_std_minsum, Portfolio_minsum] = compute_esg_constrained_frontier(mean_returns, cov_matrix, ...
                          esg_scores_aligned, threshold, size(numeric_returns,1), max_low_weight, num_low_esg);

%% 7.1_alternative) Short selling
[topVols_std_minsum_short, topRets_std_minsum_short, Portfolio_minsum_short] = compute_esg_constrained_frontier(mean_returns, cov_matrix, ...
                          esg_scores_aligned, threshold, size(numeric_returns,1), max_low_weight, num_low_esg, true, max_short);

%% 8) Best in Class ESG Frontier
short_bound_high = 1;    % high ESG assets: no short allowed
short_bound_low  = -0.1; % low ESG assets: forced short up to -10%

[topVols_best, topRets_best, Portfolio_best] = compute_best_in_class_frontier( ...
                                            mean_returns, cov_matrix, esg_scores_aligned, threshold, ...
                                            size(numeric_returns,1), num_low_esg, short_bound_high, short_bound_low);

%% 8_alternative) Best in Class ESG Frontier
[topVols_best_esg, topRets_best_esg, Portfolio_best_esg] = compute_best_in_class_frontier( ...
                                            mean_returns, cov_matrix, esg_scores_aligned, threshold, ...
                                            top_n, num_low_esg, short_bound_high, short_bound_low);

%% Plot Best in Class (8) vs Weight Reduction Frontier (7)
figure;
hold on;
plot(topVols_best, topRets_best, 'LineWidth', 2, 'Color', 'g');
plot(topVols_std_esg_minsum, topRets_std_esg_minsum, '--r', 'LineWidth', 1.5);
xlabel('Volatility');
ylabel('Expected Return');
legend('Best in Class', 'Min-sum ESG constraint', 'Location', 'best');
title('Efficient Frontier - Best in Class vs Min-sum ESG');
grid on;
hold off;

%% Plot
figure;
plot(topVols_esg_constr_vec(:,8), topRets_esg_constr_vec(:,8), 'LineWidth', 1.5); hold on
plot(topVols_esg_constr_vec_short(:,8), topRets_esg_constr_vec_short(:,8), 'LineWidth', 1.5);
plot(topVols_std, topRets_std, 'LineWidth', 1.5);
plot(topVols_std_short, topRets_std_short, 'LineWidth', 1.5);
plot(FrontierVols_track_esg, FrontierRet_track_esg, 'LineWidth', 1.5);
plot(FrontierVols_track_esg_short, FrontierRet_track_esg_short, 'LineWidth', 1.5);
plot(FrontierVols_track, FrontierRet_track, 'LineWidth', 1.5);
plot(FrontierVols_track_short, FrontierRet_track_short, 'LineWidth', 1.5);
plot(topVols_constr_vec(:,8), topRets_constr_vec(:,8), 'LineWidth', 1.5);
plot(topVols_constr_vec_short(:,8), topRets_constr_vec_short(:,8), 'LineWidth', 1.5);
plot(topVols_std_minsum, topRets_std_minsum, 'LineWidth', 1.5);
plot(topVols_std_minsum_short, topRets_std_minsum_short, 'LineWidth', 1.5);
plot(topVols_std_esg_minsum, topRets_std_esg_minsum, 'LineWidth', 1.5);
plot(topVols_std_esg_minsum_short, topRets_std_esg_minsum_short, 'LineWidth', 1.5);
plot(topVols_sector_filtered, topRets_sector_filtered, 'LineWidth', 1.5);
plot(topVols_sector_filtered_short, topRets_sector_filtered_short, 'LineWidth', 1.5);
plot(topVols_best, topRets_best, 'LineWidth', 1.5);
plot(topVols_best_esg, topRets_best_esg, 'LineWidth', 1.5);

xlabel('Volatility (Std. Dev.)');
ylabel('Expected Return');
title('Total Comparison');
legend('ESG top frontier - ESG Constraint', ...
       'ESG top frontier - ESG Constraint short', ...
       'Standard Frontier', ...
       'Standard Frontier short', ...
       'ESG top frontier - Tracking error', ...
       'ESG top frontier - Tracking error short', ...
       'Standard frontier - Tracking error', ...
       'Standard frontier - Tracking error short', ...
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
plot(FrontierVols_track_esg, FrontierRet_track_esg, 'LineWidth', 1.5);
plot(FrontierVols_track_esg_short, FrontierRet_track_esg_short, 'LineWidth', 1.5);
plot(topVols_std_esg_minsum, topRets_std_esg_minsum, 'LineWidth', 1.5);
plot(topVols_std_esg_minsum_short, topRets_std_esg_minsum_short, 'LineWidth', 1.5);
plot(topVols_best_esg, topRets_best_esg, 'LineWidth', 1.5);

xlabel('Volatility (Std. Dev.)');
ylabel('Expected Return');
title('Total Comparison - Top ESG');
legend('ESG top frontier - ESG Constraint', ...
       'ESG top frontier - ESG Constraint short', ...
       'ESG top frontier - Tracking error',...
       'ESG top frontier - Tracking error short',...
       'ESG top frontier - ESG Constrint + Sum constr',...
       'ESG top frontier - ESG Constrint + Sum constr short',...
       'ESG top frontier - Best in Class');
grid on;

%% Standard Frontier Plot
figure;
plot(topVols_std, topRets_std, 'LineWidth', 1.5);
hold on
plot(topVols_std_short, topRets_std_short, 'LineWidth', 1.5);
plot(FrontierVols_track, FrontierRet_track, 'LineWidth', 1.5);
plot(FrontierVols_track_short, FrontierRet_track_short, 'LineWidth', 1.5); % FUNZIONA MALE
plot(topVols_constr_vec(:,8), topRets_constr_vec(:,8), 'LineWidth', 1.5);
plot(topVols_constr_vec_short(:,8), topRets_constr_vec_short(:,8), 'LineWidth', 1.5);
plot(topVols_std_minsum, topRets_std_minsum, 'LineWidth', 1.5);
plot(topVols_std_minsum_short, topRets_std_minsum_short, 'LineWidth', 1.5);
plot(topVols_best, topRets_best, 'LineWidth', 1.5);

xlabel('Volatility (Std. Dev.)');
ylabel('Expected Return');
title('Total Comparison - Standard');
legend('Standard Frontier',...
       'Standard Frontier short',...
       'Standard frontier - Tracking error',...
       'Standard frontier - Tracking error short',...
       'Standard frontier - ESG Constraint',...
       'Standard frontier - ESG Constraint short',...
       'Standard frontier - ESG Constraint + Sum constr',...
       'Standard frontier - ESG Constraint + Sum constr short',...
       'Standard frontier - Best in Class');
grid on;

%% 9) Backtesting
sheetName = 'Backtesting_Prices';

% Standard Portfolio
figure
hold on
for i=1:length(Portfolio)
    [portfolio_returns, portfolio_cum_returns, metrics] = run_backtest(excelFile, sheetName, Portfolio, mean_returns, i);
end

% Standard Portfolio with short selling
figure
hold on
for i=1:length(Portfolio_short)
    [portfolio_returns_short, portfolio_cum_returns_short, metrics_short] = run_backtest(excelFile, sheetName, Portfolio_short, mean_returns, i);
end

%% Portfolio with penalties for worst ESG Scores
figure
hold on
for i=1:length(Portfolio_minsum)
    [portfolio_returns_minsum, portfolio_cum_returns_minsum, metrics_minsum] = run_backtest(excelFile, sheetName, Portfolio_minsum, mean_returns, i);
end

% Portfolio with penalties for worst ESG Scores and short selling
figure
hold on
for i=1:length(Portfolio_minsum_short)
    [portfolio_returns_minsum_short, portfolio_cum_returns_minsum_short, metrics_minsum_short] = run_backtest(excelFile, sheetName, Portfolio_minsum_short, mean_returns, i);
end

%% Portfolio with penalties for worst ESG Scores - Top 50 ESG Scores
figure
hold on
for i=1:length(Portfolio_minsum_esg)
    [portfolio_returns_minsum_esg, portfolio_cum_returns_minsum_esg, metrics_minsum_esg] = run_backtest(excelFile, sheetName, Portfolio_minsum_esg, mean_returns, i, esg_scores_aligned, top_n);
end

% Portfolio with penalties for worst ESG Scores and short selling
figure
hold on
for i=1:length(Portfolio_minsum_esg_short)
    [portfolio_returns_minsum_esg_short, portfolio_cum_returns_minsum_esg_short, metrics_minsum_esg_short] = run_backtest(excelFile, sheetName, Portfolio_minsum_esg_short, mean_returns, i, esg_scores_aligned, top_n);
end

%% Standard Portfolio with ESG constraint
figure
hold on
for i=1:length(Portfolio_esg)
    [portfolio_returns_esg, portfolio_cum_returns_esg, metrics_esg] = run_backtest(excelFile, sheetName, Portfolio_esg, mean_returns, i, esg_scores_aligned, top_n);
end

% Standard Portfolio with ESG constraint and short selling
figure
hold on
for i=1:length(Portfolio_esg_short)
    [portfolio_returns_esg_short, portfolio_cum_returns_esg_short, metrics_esg_short] = ...
                                                            run_backtest(excelFile, sheetName, Portfolio_esg_short, mean_returns, i, esg_scores_aligned, top_n);
end

%% Standard Portfolio - Sectorial Analysis
figure
hold on
for i=1:length(Portfolio_sector)
    [portfolio_returns_sector, portfolio_cum_returns_sector, metrics_sector] =...
                                                    run_backtest(excelFile, sheetName, Portfolio_sector, mean_returns, i, esg_scores_aligned, top_n);
end

% Standard Portfolio with short selling - Sectorial Analysis
figure
hold on
for i=1:length(Portfolio_sector_short)
    [portfolio_returns_sector_short, portfolio_cum_returns_sector_short, metrics_sector_short] =...
                                            run_backtest(excelFile, sheetName, Portfolio_sector_short, mean_returns, i, esg_scores_aligned, top_n);
end

%% Compute standard efficient frontier considering stocks with the highest ESG score
figure
hold on
for i=1:length(Portfolio_std_top_esg_vec)
    [portfolio_returns_top_esg, portfolio_cum_returns_top_esg, metrics_top_esg] =...
                                                    run_backtest(excelFile, sheetName, Portfolio_std_top_esg_vec, mean_returns, i, esg_scores_aligned, top_vec(i));
end

% Standard Portfolio with short selling - Sectorial Analysis
figure
hold on
for i=1:length(Portfolio_std_top_esg_vec_short)
    [portfolio_returns_top_esg_short, portfolio_cum_returns_top_esg_short, metrics_top_esg_short] =...
                                            run_backtest(excelFile, sheetName, Portfolio_std_top_esg_vec_short, mean_returns, i, esg_scores_aligned, top_vec(i));
end

%% Compute standard efficient frontier considering the one that maximizes sharpe ratio
[portfolio_returns_tracking, portfolio_cum_returns_tracking, metrics_tracking] =...
                                                    run_backtest_weights(excelFile, sheetName, Portfolio_tracking, mean_returns, esg_scores_aligned, size(numeric_returns,1));

[portfolio_returns_tracking_short, portfolio_cum_returns_tracking_short, metrics_tracking_short] =...
                                                    run_backtest_weights(excelFile, sheetName, Portfolio_tracking_short, mean_returns, esg_scores_aligned, size(numeric_returns,1));

[portfolio_returns_tracking_esg, portfolio_cum_returns_tracking_esg, metrics_tracking_esg] =...
                                                    run_backtest_weights(excelFile, sheetName, Portfolio_tracking_esg,...
                                                    mean_returns, esg_scores_aligned, top_n);

[portfolio_returns_tracking_esg_short, portfolio_cum_returns_tracking_esg_short, metrics_tracking_esg_short] =...
                                                    run_backtest_weights(excelFile, sheetName, Portfolio_tracking_esg_short,...
                                                    mean_returns, esg_scores_aligned, top_n);

%% Compute standard efficient frontier considering Best in Class ESG Frontier
figure
[portfolio_returns_best, portfolio_cum_returns_best, metrics_best] = ...
                                                    run_backtest(excelFile, sheetName, Portfolio_best, mean_returns, 1, esg_scores_aligned, size(numeric_returns,1));

figure
[portfolio_returns_best_esg, portfolio_cum_returns_best_esg, metrics_best_esg] = ...
                                                    run_backtest(excelFile, sheetName, Portfolio_best_esg, mean_returns, 1, esg_scores_aligned, top_n);

%%
disp("Total computational time: ");
toc
