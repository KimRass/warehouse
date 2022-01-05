# Classification of Data
## Cross-Sectional Data
- Source: https://en.wikipedia.org/wiki/Cross-sectional_data
- ***A type of data collected by observing many subjects (such as individuals, firms, countries, or regions) at the one point or period of time. The analysis might also have no regard to differences in time. *Analysis of cross-sectional data usually consists of comparing the differences among selected subjects.***
## Time Series Data
- Source: https://en.wikipedia.org/wiki/Time_series
- A time series is a series of data points indexed (or listed or graphed) in time order. Most commonly, a time series is a sequence taken at successive equally spaced points in time. Thus it is a sequence of discrete-time data.
## Panel Data
- Source: https://en.wikipedia.org/wiki/Panel_data
- panel data is both multi-dimensional data involving measurements over time. *Time series and cross-sectional data can be thought of as special cases of panel data that are in one dimension only (one panel member or individual for the former, one time point for the latter).*

# White Noise

# Characteristics of Time Series Data
## Trend
- Trend is a pattern in data that shows the movement of a series to relatively higher or lower values over a long period of time.
- A trend is observed when there is an increasing or decreasing slope in the time series.
- Trend usually happens for some time and then disappears, it does not repeat.
## Seasonality
- The presence of variations that occur at specific regular intervals less than a year, such as weekly, monthly, or quarterly.
- Seasonality may be caused by various factors, such as weather, vacation, and holidays and consists of periodic, repetitive, and generally regular and predictable patterns in the levels of a time series.
- Seasonal fluctuations in a time series can be contrasted with cyclical patterns.
- If the period is unchanging and associated with some aspect of the calendar, then the pattern is seasonal.
## Cycle
- If the fluctuations are not of fixed period then they are cyclic.
- In general, the average length of cycles is longer than the length of a seasonal pattern, and the magnitude of cycles tends to be more variable than the magnitude of seasonal patterns.
## Stationarity
- A stationary process has the property that the mean, variance and autocorrelation structure do not change over time. Stationarity can be defined in precise mathematical terms, but for our purpose we mean a flat looking series, without trend, constant variance over time, a constant autocorrelation structure over time and no periodic fluctuations.

# Test
## Stationarity Test
### ADF (Augmented Dickey-Fuller) Test, ADF-GLS Test, PP (Phillips-Perron) Test
- Null hypothesis: The residuals are non-stationary.(p-value >= Criterion)
- Alternative hypothesis: The residuals are stationary.(p-value < Criterion)
### KPSS (Kwiatkowski Phillips Schmidt Shin) Test 
- Null hypothesis: The residuals are non-stationary.(p-value < Criterion)
- Alternative hypothesis: The residuals are stationary.(p-value >= Criterion)
## Normality Test
### Shapiro–Wilk Test, Kolmogorov–Smirnov Test, Lilliefors Test, Anderson-Darling Test, Jarque-Bera Test, Pearsons's chi-squared Test, D'Agostino's K-squared Test
- Null hypothesis: The residuals are normally distributed.(p-value >= Criterion)
- Alternative hypothesis: The residualts are not normaly distributed.(p-value < Criterion)
## Autocorrelation Test
### Ljung–Box Test, Portmanteau Test, Breusch–Godfrey Test
- Null hypothesis: Autocorrelation is absent.(p-value >= Criterion)
- Alternative hypothesis: Autocorrelation is present.(p-value < Criterion)
### Durbin–Watson Statistic
- Null hypothesis: Autocorrelation is absent.(Test Statistic is near at 2)
- Alternative hypothesis: Autocorrelation is present.(Test Statistic is near at 0(Positive) or 4(Negative))
## Homoscedasticity Test
### Goldfeld–Quandt Test, Breusch–Pagan Test, Bartlett's Test
- Null hypothesis: The residuals are homoscedasticity.(p-value >= Criterion)
- Alternative hypothesis: The residuals are heteroscedasticity.(p-value < Criterion)

# Feature Scaling
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
```
```python
sc = StandardScaler()
```
#### `sc.fit()`, `sc.transform()`, `sc.fit_transform()`
## Standard Scaler
## Min-Max Scaler
## Robust Scaler
## Normalizer