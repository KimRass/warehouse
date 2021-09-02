# Trend
- Trend is a pattern in data that shows the movement of a series to relatively higher or lower values over a long period of time.
- A trend is observed when there is an increasing or decreasing slope in the time series.
- Trend usually happens for some time and then disappears, it does not repeat.

# Seasonality
- The presence of variations that occur at specific regular intervals less than a year, such as weekly, monthly, or quarterly.
- Seasonality may be caused by various factors, such as weather, vacation, and holidays and consists of periodic, repetitive, and generally regular and predictable patterns in the levels of a time series.
- Seasonal fluctuations in a time series can be contrasted with cyclical patterns.
- if the period is unchanging and associated with some aspect of the calendar, then the pattern is seasonal.

# Cycle
- If the fluctuations are not of fixed period then they are cyclic.
- In general, the average length of cycles is longer than the length of a seasonal pattern, and the magnitude of cycles tends to be more variable than the magnitude of seasonal patterns.

# Stationarity
- A stationary process has the property that the mean, variance and autocorrelation structure do not change over time. Stationarity can be defined in precise mathematical terms, but for our purpose we mean a flat looking series, without trend, constant variance over time, a constant autocorrelation structure over time and no periodic fluctuations.

## Stationarity Test

### Augmented Dickey-Fuller(ADF) Test, ADF-GLS Test, Phillips-Perron(PP) Test
- Null Hypothesis $H_0$: The residuals are non-stationary.(p-value >= Criterion)
- Alternative Hypothesis $H_1$: The residuals are stationary.(p-value < Criterion)
    
### Kwiatkowski Phillips Schmidt Shin(KPSS) Test 
- Null Hypothesis $H_0$: The residuals are non-stationary.(p-value < Criterion)
- Alternative Hypothesis $H_1$: The residuals are stationary.(p-value >= Criterion)

## Normality Test

### Shapiro–Wilk Test, Kolmogorov–Smirnov Test, Lilliefors Test, Anderson-Darling Test, Jarque-Bera Test, Pearsons's chi-squared Test, D'Agostino's K-squared Test
- Null Hypothesis $H_0$: The residuals are normally distributed.(p-value >= Criterion)
- Alternative Hypothesis $H_1$: The residualts are not normaly distributed.(p-value < Criterion)

## Autocorrelation Test

### Ljung–Box Test, Portmanteau Test, Breusch–Godfrey Test
- Null Hypothesis $H_0$: Autocorrelation is absent.(p-value >= Criterion)
- Alternative Hypothesis $H_1$: Autocorrelation is present.(p-value < Criterion)
  
### Durbin–Watson Statistic
- Null Hypothesis $H_0$: Autocorrelation is absent.(Test Statistic is near at 2)
- Alternative Hypothesis $H_1$: Autocorrelation is present.(Test Statistic is near at 0(Positive) or 4(Negative))

## Homoscedasticity Test

### Goldfeld–Quandt Test, Breusch–Pagan Test, Bartlett's Test
- Null Hypothesis $H_0$: The residuals are homoscedasticity.(p-value >= Criterion)
- Alternative Hypothesis $H_1$: The residuals are heteroscedasticity.(p-value < Criterion)

# Feature Scaling
## Standard Scaler
## Min-Max Scaler
## Robust Scaler
## Normalizer