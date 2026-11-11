# Asset Management Techniques under ESG constraints: Efficient Frontier Analysis and Portfolio Weights Forecasting with a Kalman Filter

This thesis analyses the integration of Environmental, Social and Governance (ESG)
factors into portfolio optimization, combining traditional asset management frameworks
with sustainability constraints. The aim is to provide a quantitative assessment of how
different ESG integration methods influence portfolio efficiency and robustness over time.
The empirical study is based on the S&P 500 constituents from 2015 to 2024, with ESG
data from Refinitiv applied across several frameworks: Constrained optimization, where
ESG scores act as portfolio restrictions; Negative screening, excluding low-rated sectors;
Best-in-Class selection, favoring top ESG performers within each sector and shorting the
lowest; and Linear limits on low rated assets, a softer constraint imposing upper caps
on low-score assets. Each configuration is analyzed under long-only and limited shortselling
regimes to evaluate diversification trade-offs. A forecasting extension introduces
a Kalman Filter (KF) to model the dynamic evolution of minimum-variance portfolio
weights, assessing predictive accuracy and out-of-sample performance over 2023–2024.
Results from the first chapter show that short-selling enhances frontier efficiency and
diversification, especially in higher-return regions, but increases computational instability.
Across all models, stronger ESG constraints generate a negative efficiency premium, with
tighter rules reducing Sharpe and Calmar ratios and raising volatility. The Best-in-Class
with short-selling and Constrained optimization with mild ESG limits achieve the best
balance between return and stability.
The second chapter shows that the Kalman Filter captures real portfolio dynamics, producing
smoother and more coherent weight trajectories with lower volatility but slightly
weaker returns. Its adaptive smoothing mitigates short-term shocks but lags in fastchanging
markets. Overall, the research finds that rigid ESG mandates entail measurable
efficiency losses, while adaptive, data-driven and moderately flexible approaches, especially
those combining dynamic filtering and selective short exposure, can align sustainability
with robust portfolio behavior.

# Project Materials

- [Main Code](./Main.m)
- [Full Thesis](./2025_12_Callini_Tesi.pdf)
- [Executive Summary](./2025_12_Callini_ExecutiveSummary.pdf)
- [Presentation](./2025_12_Callini_Presentation.pptx)

# Keywords
ESG, Sustainable Finance, Portfolio Optimization, Efficient Frontier, Short-Selling, Kalman Filter, Linear Constraints.
