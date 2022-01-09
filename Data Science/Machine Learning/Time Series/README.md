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
## Detrend
```python
detrend = data["passengers"] - decomp.trend
```
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
### Wide-Sense Stationarity
- Source: https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average
- *For a wide-sense stationary time series, the mean and the variance/autocovariance keep constant over time.*

# Datasets
## `Air Passengers`
- Source: https://www.kaggle.com/rakannimer/air-passengers
## `Bike Sharing Demand`
- Source: https://www.kaggle.com/c/bike-sharing-demand

# Preprocessing
## Set Frequency
- `freq`: (`"YS"` (year start), `"Y"` (year end), `"QS"` (quarter start), `"Q"` (quarter end), `"MS"` (month start), `"M"` (month end), "W"` (week), `"D"` (day), `"H"` (hour), `"T"` (minute), `"S"` (second))
- `method="ffill"`: Forawd fill.
- `method="bfill"`: Backward fill.
```python
data = data.asfreq()
```
## Data Transformation
### Log Transformation
- Source: https://cadmus.eui.eu/handle/1814/11150
- ***In time series analysis log transformation is often considered to stabilize the variance of a series.***
```python
import numpy as np

data["var_log"] = np.log(data["var"])
```
### Differencing
- Source: https://machinelearningmastery.com/remove-trends-seasonality-difference-transform-python/
- ***Differencing can help stabilize the mean of the time series by removing changes in the level of a time series, and so eliminating (or reducing) trend and seasonality.***
- Some temporal structure may still exist after performing a differencing operation, such as in the case of a nonlinear trend. As such, *the process of differencing can be repeated more than once until all temporal dependence has been removed. The number of times that differencing is performed is called the difference order.*
- Source: https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average
- ***Differencing in statistics is a transformation applied to a non-stationary time-series in order to make it stationary in the mean sense (viz., to remove the non-constant trend), but having nothing to do with the non-stationarity of the variance or autocovariance. Likewise, the seasonal differencing is applied to a seasonal time-series to remove the seasonal component.***
```python
data["var_diff"] = data["var"].diff(n)
```
- Sometimes it may be necessary to difference the data a second time to obtain a stationary time series, which is referred to as second-order differencing.
### Lagged Variable
### Moving Average
```python
data[["var"]].rolling(window).mean()
```
## Time Series Decomposition
```python
import statsmodels.api as sm

decomp = sm.tsa.seasonal_decompose(data["var"], model="additive")
# decomp = sm.tsa.seasonal_decompose(data["var"], model="multiplicative")

y_observ = decomp.observed # `pandas.Series`
y_trend = decomp.trend # `pandas.Series`
y_season = decomp.seasonal # `pandas.Series`
y_resid = decomp.resid # `pandas.Series`
```

# Splitting Dataset
- Overfitting would be a major concern since your training data could contain information from the future. ***It is important that all your training data happens before your test data.*** One way of validating time series data is by using k-fold CV and making sure that ***in each fold the training data takes place before the test data.***
- Using `sklearn.model_selection.train_test_split(shuffle=False)`
```python
from sklearn.model_selection import train_test_split

data_tr, data_te = train_test_split(data, test_size=0.2, shuffle=False)
```

# Autocorrelation
- Source: https://statisticsbyjim.com/time-series/autocorrelation-partial-autocorrelation/
- Autocorrelation is the correlation between two observations at different points in a time series. For example, values that are separated by an interval might have a strong positive or negative correlation. ***When these correlations are present, they indicate that past values influence the current value.***
- *Analysts record time-series data by measuring a characteristic at evenly spaced intervals—such as daily, monthly, or yearly. *The number of intervals between the two observations is the lag. This lag can be days, quarters, or years depending on the nature of the data.
## ACF (AutoCorrelation Function)
- The autocorrelation function (ACF) assesses the correlation between observations in a time series for a set of lags.
- In an ACF plot, *each bar represents the size and direction of the correlation. Bars that extend across the red line are statistically significant.*
- ***Stationarity: The autocorrelation function declines to near zero rapidly for a stationary time series. In contrast, the ACF drops slowly for a non-stationary time series.***
- ***Trend: When trends are present in a time series, shorter lags typically have large positive correlations because observations closer in time tend to have similar values. The correlations taper off slowly as the lags increase.***
- ***Seasonality: When seasonal patterns are present, the autocorrelations are larger for lags at multiples of the seasonal frequency than for other lags.***
- When a time series has both a trend and seasonality, the ACF plot displays a mixture of both effects.
```python
import statsmodels.api as sm

sm.graphics.tsa.plot_acf(x=data["var"], lags=50);
```
### White Noise
- ***For random data, autocorrelations should be near zero for all lags. Analysts also refer to this condition as white noise. Non-random data have at least one significant lag.***
## PACF (Partial AutoCorrelation Function)
- *The partial autocorrelation function is similar to the ACF except that it displays only the correlation between two observations that the shorter lags between those observations do not explain. For example, the partial autocorrelation for lag 3 is only the correlation that lags 1 and 2 do not explain. In other words, the partial correlation for each lag is the unique correlation between those two observations after partialling out the intervening correlations.*
- As you saw, the autocorrelation function helps assess the properties of a time series. In contrast, *the partial autocorrelation function (PACF) is more useful during the specification process for an autoregressive model. Analysts use partial autocorrelation plots to specify regression models with time series data and Auto Regressive Integrated Moving Average (ARIMA) models.*
- Typically, you will use the ACF to determine whether an autoregressive model is appropriate. If it is, you then use the PACF to help you choose the model terms.
- ***On the graph, the partial autocorrelations for lags 1 and 2 are statistically significant. The subsequent lags are nearly significant. Consequently, this PACF suggests fitting either a second or third-order autoregressive model.***
```python
import statsmodels.api as sm

sm.graphics.tsa.plot_pacf(x=data["var"], lags=50);
```

# ARIMA (AutoRegressive Integrated Moving Average)
- Source: https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average
- *An autoregressive integrated moving average (ARIMA) model is a generalization of an autoregressive moving average (ARMA) model.* Both of these models are fitted to time series data either to better understand the data or to predict future points in the series (forecasting). ***ARIMA models are applied in some cases where data show evidence of non-stationarity in the sense of mean (but not variance/autocovariance), where an initial differencing step (corresponding to the "integrated" part of the model) can be applied one or more times to eliminate the non-stationarity of the mean function (i.e., the trend).*** When the seasonality shows in a time series, the seasonal-differencing could be applied to eliminate the seasonal component.
- ***The AR part of ARIMA indicates that the evolving variable of interest is regressed on its own lagged (i.e., prior) values. The MA part indicates that the regression error is actually a linear combination of error terms whose values occurred contemporaneously and at various times in the past. The I (for "integrated") indicates that the data values have been replaced with the difference between their values and the previous values (and this differencing process may have been performed more than once).***
- ***Non-seasonal ARIMA models are generally denoted ARIMA(p,d,q) where parameters p, d, and q are non-negative integers, p is the order (number of time lags) of the autoregressive model, d is the degree of differencing (the number of times the data have had past values subtracted), and q is the order of the moving-average model.***
- When two out of the three terms are zeros, the model may be referred to based on the non-zero parameter, dropping "AR", "I" or "MA" from the acronym describing the model. For example, ARIMA(1, 0, 0) is AR(1), ARIMA(0, 1, 0) is I(1), and ARIMA(0, 0, 1) is MA(1).
- ARIMA models can be estimated following the Box–Jenkins approach.
```python
from pmdarima.arima import auto_arima

model = auto_arima(diff_train_data, start_p=1, start_q=1, max_p=3, max_q=3, m=12, seasonal=True, d=1, D=1, max_P=3, max_Q=3, trace=True, error_action="ignore")
```
## Seasonal ARIMA
- ***Seasonal ARIMA models are usually denoted ARIMA(p,d,q)(P,D,Q)s, where m refers to the number of periods in each season, and the uppercase P,D,Q refer to the autoregressive, differencing, and moving average terms for the seasonal part of the ARIMA model.***
```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(y, order=(1, 0, 0), seasonal_order=(1, 0, 0, 12))
fit = model.fit()
fit.summary()

pred = fit.get_forecast()
pred.predicted_mean
# Lower bound
pred.conf_int()[:, 0]
# Upper bound
pred.conf_int()[:, 1]
```

# Evaluation Metrics
## AIC (Akaike Information Criterion)
- Source: https://en.wikipedia.org/wiki/Akaike_information_criterion
- The Akaike information criterion (AIC) is an estimator of prediction error and thereby relative quality of statistical models for a given set of data.[1][2][3] Given a collection of models for the data, AIC estimates the quality of each model, relative to each of the other models. Thus, AIC provides a means for model selection.

AIC is founded on information theory. When a statistical model is used to represent the process that generated the data, the representation will almost never be exact; so some information will be lost by using the model to represent the process. AIC estimates the relative amount of information lost by a given model: the less information a model loses, the higher the quality of that model.

In estimating the amount of information lost by a model, AIC deals with the trade-off between the goodness of fit of the model and the simplicity of the model. In other words, AIC deals with both the risk of overfitting and the risk of underfitting.

The Akaike information criterion is named after the Japanese statistician Hirotugu Akaike, who formulated it. It now forms the basis of a paradigm for the foundations of statistics and is also widely used for statistical inference.

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