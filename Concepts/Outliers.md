# types of outliers
- 출처 : https://adataanalyst.com/machine-learning/comprehensive-guide-feature-engineering/
- Univariate outliers can be found when we look at distribution of a single variable.
Multi-variate outliers are outliers in an n-dimensional space. In order to find them, you have to look at distributions in multi-dimensions.
## data entry errors
- Human errors such as errors caused during data collection, recording, or entry can cause outliers in data.
## data processing errors
- It is possible that some manipulation or extraction errors may lead to outliers in the dataset.
## measurement errors
- It is the most common source of outliers. This is caused when the measurement instrument used turns out to be faulty. 
## experimental errors
- For example: In a 100m sprint of 7 runners, one runner missed out on concentrating on the ‘Go’ call which caused him to start late. Hence, this caused the runner’s run time to be more than other runners.
## intentional outliers
- For example: Teens would typically under report the amount of alcohol that they consume. Only a fraction of them would report actual value. Here actual values might look like outliers because rest of the teens are under reporting the consumption.
## sampling errors
- For instance, we have to measure the height of athletes. By mistake, we include a few basketball players in the sample.
## natural outliers
# outliers detection
- 출처 : https://adataanalyst.com/machine-learning/comprehensive-guide-feature-engineering/
# outliers treatment
## deleting
## transforming
- Natural log of a value reduces the variation caused by extreme values.
- Decision Tree algorithm allows to deal with outliers well due to binning of variable. We can also use the process of assigning weights to different observations.
## binning
- Decision Tree algorithm allows to deal with outliers well due to binning of variable. We can also use the process of assigning weights to different observations.
## imputing
- We can use mean, median, mode imputation methods. Before imputing values, we should analyse if it is natural outlier or artificial. If it is artificial, we can go with imputing values. We can also use statistical model to predict values of outlier.
## treating separately
- If there are significant number of outliers, we should treat them separately in the statistical model. One of the approach is to treat both groups as two different groups and build individual model for both groups and then combine the output.
## Turkey Fences
- https://cyan91.tistory.com/40
- https://lsjsj92.tistory.com/556?category=853217
## Z-score
- https://cyan91.tistory.com/40
- https://soo-jjeong.tistory.com/121
