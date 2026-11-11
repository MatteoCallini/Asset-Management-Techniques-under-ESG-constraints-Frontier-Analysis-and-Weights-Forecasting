function [w_opt, realized_mu, realized_esg, realized_std] = compute_varmaz_weights(mean_returns, cov_matrix, esg_scores, target_return, target_esg)
% Computes minimum-norm ESG-constrained portfolio and returns variance

    % Step 1: Extract data
    mu = mean_returns.MeanReturn;
    theta = table2array(esg_scores);
    Sigma = table2array(cov_matrix);
    asset_names = mean_returns.Properties.RowNames;
    N = length(mu);

    % Step 2: Build constraint matrix X and target vector b
    X = [ones(N,1), mu, theta];         % Columns: budget, return, ESG
    b = [1; target_return; target_esg]; % Investor's specified targets

    % Step 3: Solve for minimum-norm weights
    w_opt = (X * X') \ (X * b);

    % Step 4: Normalize weights to ensure fully invested
    w_opt = w_opt / sum(w_opt);

    % Step 5: Outputs
    realized_mu = mu'*w_opt;                 % Realized expected return
    realized_esg = theta'*w_opt;              % Realized ESG score
    realized_std = sqrt(w_opt'*Sigma*w_opt);      % Realized standard deviation

    % % Optional: Display
    % disp('Optimal Weights (Varmaz method):');
    % disp(table(asset_names, w_opt, 'VariableNames', {'Asset', 'Weight'}));
    % fprintf('Realized Return: %.4f\n', realized_mu);
    % fprintf('Realized ESG Score: %.4f\n', realized_esg);
    % fprintf('Realized Variance: %.6f (Std Dev: %.4f)\n', realized_std^2, realized_std);
end