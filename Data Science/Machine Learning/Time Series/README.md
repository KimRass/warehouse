# Time Series
- Source: https://www.geeksforgeeks.org/what-is-a-trend-in-time-series/
- Time series data is a sequence of data points that measure some variable over ordered period of time.

# Classification of Time Series Data
## Cross-Sectional Data
- Source: https://en.wikipedia.org/wiki/Cross-sectional_data
- ***A type of data collected by observing many subjects (such as individuals, firms, countries, or regions) at the one point or period of time. The analysis might also have no regard to differences in time. *Analysis of cross-sectional data usually consists of comparing the differences among selected subjects.***
## Time Series Data
- Source: https://en.wikipedia.org/wiki/Time_series
- A time series is a series of data points indexed (or listed or graphed) in time order. Most commonly, a time series is a sequence taken at successive equally spaced points in time. Thus it is a sequence of discrete-time data.
## Panel Data
- Source: https://en.wikipedia.org/wiki/Panel_data
- panel data is both multi-dimensional data involving measurements over time. *Time series and cross-sectional data can be thought of as special cases of panel data that are in one dimension only (one panel member or individual for the former, one time point for the latter).*

# Characteristics of Time Series Data
## Trend
- Source: https://www.geeksforgeeks.org/what-is-a-trend-in-time-series/
- Trend is a pattern in data that shows the movement of a series to relatively higher or lower values over a long period of time. In other words, *a trend is observed when there is an increasing or decreasing slope in the time series. Trend usually happens for some time and then disappears, it does not repeat.* For example, some new song comes, it goes trending for a while, and then disappears. There is fairly any chance that it would be trending again.
## Seasonality
- Source: https://en.wikipedia.org/wiki/Seasonality
- In time series data, *seasonality is the presence of variations that occur at specific regular intervals less than a year, such as weekly, monthly, or quarterly.* Seasonality may be caused by various factors, such as weather, vacation, and holidays and consists of periodic, repetitive, and generally regular and predictable patterns in the levels of a time series.*
- *Seasonal fluctuations in a time series can be contrasted with cyclical patterns. The latter occur when the data exhibits rises and falls that are not of a fixed period.* Such non-seasonal fluctuations are usually due to economic conditions and are often related to the "business cycle"; *their period usually extends beyond a single year, and the fluctuations are usually of at least two years.*
## Cycle
- Source: http://www.idc-online.com/technical_references/pdfs/civil_engineering/Time_Series_Patterns.pdf
- *If the fluctuations are not of fixed period then they are cyclic. In general, the average length of cycles is longer than the length of a seasonal pattern, and the magnitude of cycles tends to be more variable than the magnitude of seasonal patterns.*
## Stationarity
- Source: https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc442.htm#:~:text=A%20common%20assumption%20in%20many,do%20not%20change%20over%20time.&text=If%20the%20data%20contain%20a,the%20residuals%20from%20that%20fit.
- ***A stationary process has the property that the mean, variance and autocorrelation structure do not change over time. Stationarity can be defined in precise mathematical terms, but for our purpose we mean a flat looking series, without trend, constant variance over time, a constant autocorrelation structure over time and no periodic fluctuations.***

# Preprocessing
## Set Frequency
```
data = data.asfreq()
```
## Variable Transformation
### Log Transformation
```python
data["var_log"] = data["var"].map(np.log())
```
### Difference Transformation
```python
data["var_diff"] = data["var"].diff(n)
```
### Lagged Variable
## Moving Average
```python
data[["var"]].rolling(window).mean()
```
## Decomposition
```python
import statsmodels.api as sm

decomp = sm.tsa.seasonal_decompose(data["var"], model="additive")
# decomp = sm.tsa.seasonal_decompose(data["var"], model="multiplicative")

y_observ = decomp.observed
y_trend = decomp.trend
y_season = decomp.seasonal
y_resid = decomp.resid
```
# Autocorrelation
```python
import statsmodels.api as sm

fig = sm.graphics.tsa.plot_acf(ax=axes[0], x=resid_tr["resid"].iloc[1:], lags=100, use_vlines=True)
```

# Tests (Not important)
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
- 일반적인 데이터와 전혀 다른 특성
- 일반적인 데이터를 다루는 모델에 적용할 경우 원하는 결과 얻지 못할 가능성이 높다.

# Horizon
# Stage
# Robust? 강건하다

# Seq2Seq-Based
## MQ-RNN
## DeepAR

# RCGAN 시계열 생성

# Time Series Decomposition
- 정상화: Trend 제거 -> Statinary -> 평균이 일정 -> 일반적인 모델 적용 가능
- 변환 시 로그 -> 차분 순서로 해서 음수 값이 나오지 않도록 함.
- Seasonality는 정의에 의해 자동으로 Statinary.

`Log Likelihood`: 높을수록 좋다.
`AIC`: 낮을수록 좋다