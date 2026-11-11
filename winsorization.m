function winsorized_data = winsorization(data,alpha)

    % This function applies a winsorization of order alpha on the returns
    % 
    % Input:
    % - data: matrix (NxM) containing data (row = asset, column = date
    %
    % Output:
    % - winsorized_data: matrix (NxM) containing winsorized data 

    winsorized_data = data;     % output inizialization
    lower_p = alpha/2*100;      % lower percentile
    upper_p = (1-alpha/2)*100;  % upper percentile

    for i = 1:size(data, 1)
        row = data(i, :);
        p_low = prctile(row, lower_p);
        p_high = prctile(row, upper_p);
        
        % Winsorization
        row(row < p_low) = p_low;
        row(row > p_high) = p_high;
        winsorized_data(i, :) = row;
    end
end
