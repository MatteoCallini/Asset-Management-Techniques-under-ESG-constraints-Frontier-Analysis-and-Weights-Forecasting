function plot_returns(Prices_returns)
    % Plots the time evolution of returns
    % Input:
    %   Prices_returns - table containing monthly returns for the securities

    figure;

    % Identify numeric columns
    numeric_vars = varfun(@isnumeric, Prices_returns, 'OutputFormat', 'uniform');
    returns_matrix = Prices_returns{:, numeric_vars};  % Extract only numeric data

    % Handle company names
    if ismember('CompanyCommonName', Prices_returns.Properties.VariableNames)
        names = Prices_returns.CompanyCommonName;
    else
        % If names are not available, use generic labels
        names = strcat("Stock_", string(1:height(Prices_returns)))';
    end

    % Build the time axis
    num_stocks = size(returns_matrix, 1);
    x = 1:size(returns_matrix, 2);  % Time index (e.g., months)

    hold on;
    for i = 1:num_stocks
        plot(x, returns_matrix(i, :), 'DisplayName', names{i});
    end
    hold off;

    xlabel('Time Index');
    ylabel('Return');
    title('Return Evolution of All Stocks');
    grid on;
end
