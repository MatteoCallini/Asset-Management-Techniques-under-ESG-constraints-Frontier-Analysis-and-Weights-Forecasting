function sr = sharpe_ratio(returns, freq, rf)
% Computes the annualized Sharpe Ratio
% returns: portfolio returns
% freq: sampling frequency (e.g., 12 = monthly)
% rf: annual risk-free rate (e.g., 0.02)

    ann_ret = annualized_return(returns, freq);
    ann_vol = annualized_volatility(returns, freq);
    
    rf_monthly = (1 + rf)^(1/freq) - 1;
    excess_return = mean(returns) - rf_monthly;
    sr = excess_return / std(returns) * sqrt(freq);
end
