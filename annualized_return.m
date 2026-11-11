function ar = annualized_return(returns, freq)
% Computes the annualized return
% returns: vector of returns (e.g., monthly)
% freq: data frequency (12 for monthly, 252 for daily)
    mean_ret = mean(returns);
    ar = (1 + mean_ret)^freq - 1;
end
