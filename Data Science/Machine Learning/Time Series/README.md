# Datasets
## `Air Passengers`
- Source: https://www.kaggle.com/rakannimer/air-passengers
## `Bike Sharing Demand`
- Source: https://www.kaggle.com/c/bike-sharing-demand
## `pmdarima.datasets.load_wineind()`

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
### Strict-Sense Stationarity

# Forecast Horizon
- Source: https://ec.europa.eu/eurostat/statistics-explained/index.php?title=Glossary:Forecast_horizon#:~:text=The%20forecast%20horizon%20is%20the,(more%20than%20two%20years).
- *The forecast horizon is the length of time into the future for which forecasts are to be prepared.* These generally vary from short-term forecasting horizons (less than three months) to long-term horizons (more than two years).

# Random Walk
- Source: https://www.investopedia.com/terms/r/randomwalktheory.asp
- Random walk theory suggests that changes in stock prices have the same distribution and are independent of each other. Therefore, *it assumes the past movement or trend of a stock price or market cannot be used to predict its future movement. In short, random walk theory proclaims that stocks take a random and unpredictable path that makes all methods of predicting stock prices futile in the long run.*
```python
np.cumsum(np.random.normal(size=200))
```

# Preprocessing
## Set Frequency
```python
data = data.asfreq()
```
- `freq`: (`"YS"` (year start), `"Y"` (year end), `"QS"` (quarter start), `"Q"` (quarter end), `"MS"` (month start), `"M"` (month end), `"W"` (week), `"D"` (day), `"H"` (hour), `"T"` (minute), `"S"` (second))
- `method="ffill"`: Forawd fill.
- `method="bfill"`: Backward fill.
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
- `min_periods`: Minimum number of observations in window required to have a value (otherwise result is NA).
## Treating Missing Values
### Imputation
- Imputation with Mean
	```python
	data["mean_imputation"] = data["var"].fillna(data["var"].mean())
	```
- Median Imputation with 
	```python
	data["median_imputation"] = data["var"].fillna(data["var"].median())
	```
- LOCF (Last Observation Carried Forward)
	```python
	data["LOCF"] = data["var"].fillna(data["var"].shift(1))
	```
- NOCB (Next Observation Carried Backward)
	```python
	data["LOCF"] = data["var"].fillna(data["var"].shift(-1))
	```
- Imputation with Moving Average
	```python
	data["moving_avg"] = data["tar"].fillna(data["tar"].rolling(24, min_periods=1).mean())
	```
### Interpolation
```python
data["interpolation"] = data["var"].interpolate(method)
```
- `method`
	- "linear", "time", "quadratic", "cubic", "slinear", "akima"
	- "polynomial", "spline" (Require that you specify `order`)
		- Source: https://en.wikipedia.org/wiki/Spline_(mathematics)
		- In mathematics, *a spline is a special function defined piecewise by polynomials.* In interpolating problems, spline interpolation is often preferred to polynomial interpolation because it yields similar results, even when using low degree polynomials, while avoiding Runge's phenomenon for higher degrees.
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
## Denoising
### Denoising with Moving Average
- 노이즈가 간혹 발생하는 경우 효과적입니다.
### Gaussian Filter
### Bilateral Filter
### Kalman Filter
- 노이즈가 매우 많은 경우 효과적입니다.
```python
kf = simdkalman.KalmanFilter(state_transition=np.array([[1, 1], [0, 1]]), process_noise = np.diag([0.1, 0.01]), observation_model=np.array([[1, 0]]), observation_noise=1.0)
kf = kf.em(data, n_iter=10)

smoothed = kf.smooth(data)
pred = kf.predict(data, n_test=15)

smoothed_mean = smoothed.observations.mean
smoothed_std = np.sqrt(smoothed.observations.cov)

pred_mean = pred.observations.mean
pred_std = np.sqrt(pred.observations.cov)

trend = smoothed.states.mean[:, 1]
trend_std = np.sqrt(smoothed.states.cov[:, 1, 1])

trend_pred = pred.states.mean[:, 1]
trend_pred_std = np.sqrt(pred.states.cov[:, 1, 1])
```

<<<<<<< HEAD
=======
# Splitting Dataset
- Overfitting would be a major concern since your training data could contain information from the future. ***It is important that all your training data happens before your test data.*** One way of validating time series data is by using k-fold CV and making sure that ***in each fold the training data takes place before the test data.***
- Using `sklearn.model_selection.train_test_split(shuffle=False)`
	```python
	from sklearn.model_selection import train_test_split

	data_tr, data_te = train_test_split(data, test_size=0.2, shuffle=False)
	```
- Using `sklearn.model_selection.TimeSeriesSplit()`
	```python
	from sklearn.model_selection import TimeSeriesSplit
	
	tscv = TimeSeriesSplit(n_splits=3)
	```
## Cross Validation (CV)
### Time Series Splits CV
- Source: https://hub.packtpub.com/cross-validation-strategies-for-time-series-forecasting-tutorial/
- The idea for time series splits is to divide the training set into two folds at each iteration on condition that the validation set is always ahead of the training split. At the first iteration, one trains the candidate model on the closing prices from January to March and validates on April’s data, and for the next iteration, train on data from January to April, and validate on May’s data, and so on to the end of the training set. This way dependence is respected.

Blocking Time Series Split Cross-Validation

However, this may introduce leakage from future data to the model. The model will observe future patterns to forecast and try to memorize them. That’s why blocked cross-validation was introduced.  It works by adding margins at two positions. The first is between the training and validation folds in order to prevent the model from observing lag values which are used twice, once as a regressor and another as a response. The second is between the folds used at each iteration in order to prevent the model from memorizing patterns from an iteration to the next.
- `from sklearn.model_selection import TimeSeriesSplit`
### Blocked CV
- Source: https://hub.packtpub.com/cross-validation-strategies-for-time-series-forecasting-tutorial/
- Using `pmdarima.model_selection.SlidingWindowForecastCV()`
	- References: https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.model_selection.SlidingWindowForecastCV.html, https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.model_selection.cross_val_score.html
	```python
	import pmdarima as pm
	from pmdarima import model_selection
	from pmdarima.model_selection import SlidingWindowForecastCV
	from pmdarima.model_selection import cross_val_score

	tr, te = model_selection.train_test_split(data, train_size=165)

	# This approach to CV slides a window over the training samples while using several future samples as a test set. While similar to the `RollingForecastCV()`, it differs in that the train set does not grow, but rather shifts.
	# `h`: The forecasting horizon, or the number of steps into the future after the last training sample for the test set.
	# `step`: The size of step taken to slide both training samples and test samples.
	# `window_size`: The size of the rolling window to use. If `None`, a rolling window of size `n_samples//5` will be used.
	cv = SlidingWindowForecastCV(h, stemp, window_size)
	# Generate indices to split data into training and test sets.
	# cv_gen = cv.split()
	# next(cv_gen)
	
	# `scoring`: (`"smape"`, `"mean_absolute_error"`, `"mean_squared_error"`)
	scores = cross_val_score(estimator=model, y=tr, scoring="smape", cv=cv, verbose=2)
	```

>>>>>>> 76a0b763042046d2adf2907edda9a09f64656bc6
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
- *For random data, autocorrelations should be near zero for all lags. Analysts also refer to this condition as white noise. Non-random data have at least one significant lag.*
- Source: https://medium.com/@kothawadegs/noise-in-time-series-data-63c5450e10f9
- The expectation of each element is 0.
- The variance of each element is ﬁnite.
- The elements are uncorrelated.
## PACF (Partial AutoCorrelation Function)
- *The partial autocorrelation function is similar to the ACF except that it displays only the correlation between two observations that the shorter lags between those observations do not explain. For example, the partial autocorrelation for lag 3 is only the correlation that lags 1 and 2 do not explain. In other words, the partial correlation for each lag is the unique correlation between those two observations after partialling out the intervening correlations.*
- As you saw, the autocorrelation function helps assess the properties of a time series. In contrast, *the partial autocorrelation function (PACF) is more useful during the specification process for an autoregressive model. Analysts use partial autocorrelation plots to specify regression models with time series data and Auto Regressive Integrated Moving Average (ARIMA) models.*
- Typically, you will use the ACF to determine whether an autoregressive model is appropriate. If it is, you then use the PACF to help you choose the model terms.
- ***On the graph, the partial autocorrelations for lags 1 and 2 are statistically significant. The subsequent lags are nearly significant. Consequently, this PACF suggests fitting either a second or third-order autoregressive model.***
```python
import statsmodels.api as sm

sm.graphics.tsa.plot_pacf(x=data["var"], lags=50);
```

# Splitting Dataset
- ***Overfitting would be a major concern since your training data could contain information from the future. It is important that all your training data happens before your test data.***
- Using `sklearn.model_selection.train_test_split(shuffle=False)`
	```python
	from sklearn.model_selection import train_test_split

	data_tr, data_te = train_test_split(data, test_size=0.2, shuffle=False)
	```
## Cross Validation (CV)
- Source: https://hub.packtpub.com/cross-validation-strategies-for-time-series-forecasting-tutorial/
### Time Series Splits CV
[!tss](https://hub.packtpub.com/wp-content/uploads/2019/05/TimeSeries-Split.png)
- The idea for time series splits is to divide the training set into two folds at each iteration on condition that the validation set is always ahead of the training split. At the first iteration, one trains the candidate model on the closing prices from January to March and validates on April’s data, and for the next iteration, train on data from January to April, and validate on May’s data, and so on to the end of the training set. This way dependence is respected.
- ***However, this may introduce leakage from future data to the model. The model will observe future patterns to forecast and try to memorize them. That’s why blocked cross-validation was introduced.***
- Using `sklearn.model_selection.TimeSeriesSplit()`
	```python
	from sklearn.model_selection import TimeSeriesSplit
	
	cv = TimeSeriesSplit(n_splits, max_train_size, test_size, gap)
### Blocked CV
[!blocked](https://hub.packtpub.com/wp-content/uploads/2019/05/Blocking-Time-Series-Split.png)
- Using `pmdarima.model_selection.SlidingWindowForecastCV()` for Seasonal ARIMA
	- References: https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.model_selection.SlidingWindowForecastCV.html, https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.model_selection.cross_val_score.html
	```python
	import pmdarima as pm
	from pmdarima import model_selection
	from pmdarima.model_selection import SlidingWindowForecastCV
	from pmdarima.model_selection import cross_val_score

	tr, te = model_selection.train_test_split(data, train_size=165)

	# This approach to CV slides a window over the training samples while using several future samples as a test set. While similar to the `RollingForecastCV()`, it differs in that the train set does not grow, but rather shifts.
	# `h`: The forecasting horizon, or the number of steps into the future after the last training sample for the test set.
	# `step`: The size of step taken to slide both training samples and test samples.
	# `window_size`: The size of the rolling window to use. If `None`, a rolling window of size `n_samples//5` will be used.
	cv = SlidingWindowForecastCV(h, stemp, window_size)
	# Generate indices to split data into training and test sets.
	# cv_gen = cv.split()
	# next(cv_gen)
	
	# `scoring`: (`"smape"`, `"mean_absolute_error"`, `"mean_squared_error"`)
	scores = cross_val_score(estimator=model, y=tr, scoring="smape", cv=cv, verbose=2)
	```

# ARIMA (AutoRegressive Integrated Moving Average)
- Source: https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average
- *An autoregressive integrated moving average (ARIMA) model is a generalization of an autoregressive moving average (ARMA) model.* Both of these models are fitted to time series data either to better understand the data or to predict future points in the series (forecasting). ***ARIMA models are applied in some cases where data show evidence of non-stationarity in the sense of mean (but not variance/autocovariance), where an initial differencing step (corresponding to the "integrated" part of the model) can be applied one or more times to eliminate the non-stationarity of the mean function (i.e., the trend).*** When the seasonality shows in a time series, the seasonal-differencing could be applied to eliminate the seasonal component.
- ***The AR part of ARIMA indicates that the evolving variable of interest is regressed on its own lagged (i.e., prior) values. The MA part indicates that the regression error is actually a linear combination of error terms whose values occurred contemporaneously and at various times in the past. The I (for "integrated") indicates that the data values have been replaced with the difference between their values and the previous values (and this differencing process may have been performed more than once).***
- ***Non-seasonal ARIMA models are generally denoted ARIMA(p,d,q) where parameters p, d, and q are non-negative integers, p is the order (number of time lags) of the autoregressive model, d is the degree of differencing (the number of times the data have had past values subtracted), and q is the order of the moving-average model.***
- When two out of the three terms are zeros, the model may be referred to based on the non-zero parameter, dropping "AR", "I" or "MA" from the acronym describing the model. For example, ARIMA(1, 0, 0) is AR(1), ARIMA(0, 1, 0) is I(1), and ARIMA(0, 0, 1) is MA(1).
- ARIMA models can be estimated following the Box–Jenkins approach.
## Seasonal ARIMA
- ***Seasonal ARIMA models are usually denoted ARIMA(p,d,q)(P,D,Q)[m], where m refers to the number of periods in each season, and the uppercase P, D, Q refer to the autoregressive, differencing, and moving average terms for the seasonal part of the ARIMA model.***
- Hyperparameter Tunning
	- Implementation
		- Reference: https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html
		```python
		import math
		import itertools
		from statsmodels.tsa.statespace.sarimax import SARIMAX
		
		minim = math.inf
		for params in itertools.product(range(0, max_p + 1), range(0, max_d + 1), range(0, max_q + 1)):
			model = SARIMAX(endog=data_tr["passengers_log"], order=params)
			hist = model.fit()
			aic = hist.aic
			if aic < minim:
				minim = aic
				best_params = params
		print(best_params)

		model = SARIMAX(data_tr["passengers_log"], order=best_params)
		hist = model.fit()
		```
	- Using `pmdarima.arima.auto_arima()`
		```python
		model = auto_arima(data_tr["passengers_log"], start_p=1, max_p=3, d=1, start_q=1, max_q=3, seasonal=True, max_P=3, D=1, max_Q=3, m=12, trace=True, error_action="ignore")

		pred_res = model.predict(len(data_te), return_conf_int=True)
		preds = pd.Series(np.exp(pred_res[0]), index=data_te.index)
		preds_lb = pd.Series(np.exp(pred_res[1])[:, 0], index=data_te.index)
		preds_ub = pd.Series(np.exp(pred_res[1])[:, 1], index=data_te.index)
		```
- Reference: https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html?highlight=sarimax#statsmodels.tsa.statespace.sarimax.SARIMAX
- Modeling & Prediction
	- Using `statsmodels.tsa.statespace.sarimax.SARIMAX()`
		```python
		from statsmodels.tsa.statespace.sarimax import SARIMAX

		model = SARIMAX(endog=data["var"], order=(p, d, q), seasonal_order=(P, D, Q, m))
		hist = model.fit()
		hist.summary()
		# hist.aic

		pred_res = hist.get_forecast(len(data_te))
		preds = np.exp(pred_res.predicted_mean)
		preds_lb = np.exp(pred.conf_int().iloc[:, 0])
		preds_ub = np.exp(pred.conf_int().iloc[:, 1])
		```
	- Using
		```python
		import pmdarima as pm
		
		model = pm.ARIMA(order=(p, d, q), seasonal_order=(P, D, Q, m), suppress_warnings=True)
		```
- Visualization
	```python
	fig = plt.figure(figsize=(12, 6))
	data["var"].plot.line();
	preds.plot.line(label="Prediction");
	plt.axvline(x=forecast_boundary, linestyle="--", color="r", label="forecast boundary");
	plt.fill_between(x=data_te.index, y1=preds_lb, y2=preds_ub, color="b", alpha=0.3, label="95% Confidence Interval");
	plt.legend(loc="upper left");
	print(f"r2_score: {r2_score(data_te['var'], preds)}")
	```

# Evaluation Metrics
## AIC (Akaike Information Criterion)
- Source: https://en.wikipedia.org/wiki/Akaike_information_criterion
- *The Akaike information criterion (AIC) is an estimator of prediction error and thereby relative quality of statistical models for a given set of data. Given a collection of models for the data, AIC estimates the quality of each model, relative to each of the other models. Thus, AIC provides a means for model selection.*
- *AIC estimates the relative amount of information lost by a given model: the less information a model loses, the higher the quality of that model.*
- In estimating the amount of information lost by a model, AIC deals with the trade-off between the goodness of fit of the model and the simplicity of the model. In other words, AIC deals with both the risk of overfitting and the risk of underfitting.
- Source: https://www.scribbr.com/statistics/akaike-information-criterion/
- ***AIC is calculated from:***
	- ***The number of independent variables used to build the model.***
	- ***The maximum likelihood estimate of the model (how well the model reproduces the data).***
- ***The best-fit model according to AIC is the one that explains the greatest amount of variation using the fewest possible independent variables.***
- *In statistics, AIC is most often used for model selection. By calculating and comparing the AIC scores of several possible models, you can choose the one that is the best fit for the data.*
- When testing a hypothesis, you might gather data on variables that you aren’t certain about, especially if you are exploring a new idea. You want to know which of the independent variables you have measured explain the variation in your dependent variable.
- Once you’ve created several possible models, you can use AIC to compare them. *Lower AIC scores are better, and AIC penalizes models that use more parameters. So if two models explain the same amount of variation, the one with fewer parameters will have a lower AIC score and will be the better-fit model.*
```python
import math

aic = 2*K - 2*math.exp(L)
```
- `K` is the number of independent variables used and `L` is the log-likelihood estimate (a.k.a. the likelihood that the model could have produced your observed y-values). The default `K` is always 2, so if your model uses one independent variable your `K` will be 3.
- Using `statsmodels.tsa.statespace.sarimax.SARIMAX().fit().aic`
	```python
	model = SARIMAX()
	hist = model.fit()
	aic = hist.aic
	```

# Tests (not important)
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
- Ljung–Box Test
	```python
	sm.stats.diagnostic.acorr_ljungbox()
	```
### Durbin–Watson Statistic
- Null hypothesis: Autocorrelation is absent.(Test Statistic is near at 2)
- Alternative hypothesis: Autocorrelation is present.(Test Statistic is near at 0(Positive) or 4(Negative))
## Homoscedasticity Test
### Goldfeld–Quandt Test, Breusch–Pagan Test, Bartlett's Test
- Null hypothesis: The residuals are homoscedasticity.(p-value >= Criterion)
- Alternative hypothesis: The residuals are heteroscedasticity.(p-value < Criterion)
- Goldfeld–Quandt Test
	```python
	sm.stats.diagnostic.het_goldfeldquandt()
	```
		- `alternative`: Specity the alternative for the p-value calculation.

# Seq2Seq-Based
## MQ-RNN
## DeepAR

# RCGAN 시계열 생성

# Libararies for Time Series
- `format`:
	- `"%Y"`: Year with century as a decimal number.
	- `"%y"`: Year without century as a zero-padded decimal number.
	- `"%m"`: Month as a zero-padded decimal number.
	- `"%d"`: Day of the month as a zero-padded decimal number.
	- `"%H"`: Hour (24-hour clock) as a zero-padded decimal number.
	- `"%I"`: Hour (12-hour clock) as a zero-padded decimal number.
	- `"%M"`: Minute as a zero-padded decimal number.
	- `"%S"`: Second as a zero-padded decimal number
	- `"%A"`: Weekday as locale’s full name.
	- `"%B"`: Month as locale’s full name.
	- `"%b"`: Month as locale’s abbreviated name.
	- `"%p"`: `"AM"` or `"PM"`
## `pd.to_datetime()`
- `unit`
- `format`: The strftime to parse time.
### `pd.to_datetime().dt`
#### `pd.to_datetime().dt.hour`, `pd.to_datetime().dt.day`,  `pd.to_datetime().dt.week`,  `pd.to_datetime().dt.dayofweek`, `pd.to_datetime().dt.month`, `pd.to_datetime().dt.quarter`, `pd.to_datetime().dt.year`
#### `pd.to_datetime().dt.normalize()`
```python
appo["end"] = appo["end"].dt.normalize()
```
## `pd.date_range()`
```python
raw.time = pd.date_range(start="1974-01-01", periods=len(raw), freq="M")
```
- Return a fixed frequency DatetimeIndex.
## `pd.Grouper()`
```python
n_tasks_month = tasks.groupby(pd.Grouper(key="task_date", freq="M")).size()
```
## `datetime.today()`
## `datetime.datetime()`, `datetime.date()`
```python
datetime.datetime(year, month, day)
```
- Require three parameters in sequence to create a date; `year`, `month`, `day`
### `datetime.datetime.now()`
### `datetime.datetime.strptime()`
```python
datetime.datetime.strptime(<<Date String>>, format)
```
- Returns a datetime corresponding to <<Date String>>, parsed according to `format`.
- `format`
### `datetime.datetime.strftime()`
```python
datetime.datetime.strftime(format)
```
- Returns a string representing the date and time, controlled by an explicit format string.
```python
gby_month["ym"] = gby_month["date_created"].apply(lambda x:datetime.datetime.strftime(x, "%m"))
```
## `datetime.timedelta`
```python
day = start + datetime.timedelta(days=1)
```
### `datetime.timedelta.days`
### `datetime.timedelta.total_seconds()`
```python
(t2 - t1).total_seconds()
```
## `relativedelta`
```python
from dateutil.relativedelta import relativedelta

data["년-월"] = data["년-월"].apply(lambda x:x + relativedelta(months=1) - datetime.timedelta(days=1))
```
## `time.time()`
## `time.localtime()`
## `time.strftime()`
```python
time.strftime("%Y%m%d", time.localtime(time.time()))
```