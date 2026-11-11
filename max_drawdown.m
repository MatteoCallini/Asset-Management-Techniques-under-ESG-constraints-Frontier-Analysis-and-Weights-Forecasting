function mdd = max_drawdown(returns)
% Computes the Max Drawdown from a series of cumulative returns

    cum_returns = cumprod(1 + returns);
    peak = cummax(cum_returns);
    drawdowns = (cum_returns - peak) ./ peak;
    mdd = min(drawdowns);
end
