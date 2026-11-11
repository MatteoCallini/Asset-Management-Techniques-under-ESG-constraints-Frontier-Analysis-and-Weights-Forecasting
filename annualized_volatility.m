function vol = annualized_volatility(returns, freq)
% Computes the annualized volatility
% returns: vector of returns
% freq: sampling frequency (e.g., 12 = monthly)
    vol = std(returns) * sqrt(freq);
end
