function [TrackingErrorVals, FrontierVols, FrontierRet, Portfolio] = ...
    compute_tracking_error_esg_frontier(mean_returns, cov_matrix, esg_scores,...
                                        benchmark_weights, threshold, top_n, allow_short_selling, short_bound)

    % Optional parameters
    if nargin < 8
        short_bound = -0.1;
    end
    if nargin < 7
        allow_short_selling = false;
    end

    % Seleziona top N asset per ESG
    esg_array = table2array(esg_scores);
    [~, idx] = sort(esg_array, 'descend');
    selected_idx = idx(1:min(top_n, length(idx)));

    asset_names = mean_returns.Properties.RowNames(selected_idx);
    mu = mean_returns.MeanReturn(selected_idx);
    Sigma = table2array(cov_matrix(asset_names, asset_names));
    ESG = table2array(esg_scores(selected_idx, :));

    % Benchmark
    benchmark_weights = benchmark_weights(:,1);
    benchmark_weights = benchmark_weights / sum(benchmark_weights);

    NumAssets = length(mu);
    TrackingErrorVals = NaN(100, 1);
    FrontierVols = NaN(100, 1);
    FrontierRet = NaN(100, 1);

    % Prepara intervallo di ritorni target
    r_min = min(mu);
    r_max = max(mu);
    target_returns = linspace(r_min, r_max, 100);

    % Create the Portfolio with all weights
    Portfolio = zeros(NumAssets,100);

    Aeq = ones(1, NumAssets);
    beq = 1;

    if allow_short_selling
        lb = short_bound * ones(NumAssets, 1);
    else
        lb = zeros(NumAssets, 1);
    end
    ub = ones(NumAssets, 1);

    options = optimoptions('quadprog', 'Display', 'off');

    for i = 1:100
        r_target = target_returns(i);

        % Vincoli: ESG + ritorno atteso
        A = [-ESG(:)'; -mu'];
        b = [-threshold; -r_target];

        % Funzione obiettivo: tracking error
        H = 2 * Sigma;
        f = -2 * Sigma * benchmark_weights;

        [w_opt, ~, exitflag] = quadprog(H, f, A, b, Aeq, beq, lb, ub, [], options);

        if exitflag == 1
            TrackingErrorVals(i) = sqrt((w_opt - benchmark_weights)' * Sigma * (w_opt - benchmark_weights));
            FrontierVols(i) = sqrt(w_opt' * Sigma * w_opt);
            FrontierRet(i) = mu' * w_opt;
            Portfolio(:,i) = w_opt;
        end
    end
end