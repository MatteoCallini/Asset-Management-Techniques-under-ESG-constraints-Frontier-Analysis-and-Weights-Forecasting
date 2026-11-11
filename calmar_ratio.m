function cr = calmar_ratio(returns, freq)
% Computes the Calmar Ratio
% returns: portfolio returns
% freq: 12 = monthly, 252 = daily

    ann_ret = annualized_return(returns, freq);
    mdd = abs(max_drawdown(returns));
    cr = ann_ret / mdd;
end
