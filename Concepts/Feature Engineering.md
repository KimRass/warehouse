- 출처 : https://adataanalyst.com/machine-learning/comprehensive-guide-feature-engineering/
## time-related variables
- Time-stamp attributes are usually denoted by the EPOCH time or split up into multiple dimensions such as (Year, Month, Date, Hours, Minutes, Seconds). But in many applications, a lot of that information is unnecessary. Consider for example a supervised system that tries to predict traffic levels in a city as a function of Location+Time. In this case, trying to learn trends that vary by seconds would mostly be misleading. The year wouldn’t add much value to the model as well. Hours, day and month are probably the only dimensions you need. So when representing the time, try to ensure that your model does require all the numbers you are providing it.
- Here is an example hypothesis: An applicant who takes days to fill in an application form is likely to be less interested / motivated in the product compared to some one who fills in the same application with in 30 minutes. Similarly, for a bank, time elapsed between dispatch of login details for Online portal and customer logging in might show customers’ willingness to use Online portal. Another example is that a customer living closer to a bank branch is more likely to have a higher engagement than a customer living far off.
## creating new ratios and proportions
- For example, in order to predict future performance of credit card sales of a branch, ratios like credit card sales / Sales person or Credit card Sales / Marketing spend would be more powerful than just using absolute number of card sold in the branch
## creating weights
- There may be domain knowledge that items with a weight above 4 incur a higher taxation rate. That magic domain number could be used to create a new binary feature Item_Above_4kg with a value of “1” for our example of 6289 grams.
## creating aggregated values
- You may also have a quantity stored as a rate or an aggregate quantity for an interval. For example, Num_Customer_Purchases aggregated over a year.
- For example, the following new binary features could be created: Purchases_Summer, Purchases_Fall, Purchases_Winter and Purchases_Spring.
## splitting features
- The Item_Weight could be split into two features: Item_Weight_Kilograms and Item_Weight_Remainder_Grams, with example values of 6 and 289 respectively.
## bucketing/binning/discretization
- Sometimes, it makes more sense to represent a numerical attribute as a categorical one.
- Consider the problem of predicting whether a person owns a certain item of clothing or not. Age might definitely be a factor here. What is actually more pertinent, is the Age Group. So what you could do, is have ranges such as 1-10, 11-18, 19-25, 26-40, etc.
- It reduces overfitting in certain applications, where you don’t want your model to try and distinguish between values that are too close by – for example, you could club together all latitude values that fall in a city, if your property of interest is a function of the city as a whole.
- Binning also reduces the effect of tiny errors, by ’rounding off’ a given value to the nearest representative. Binning does not make sense if the number of your ranges is comparable to the total possible values, or if precision is very important to you.
- For example, you may have Item_Weight in grams, with a value like 6289. You could create a new feature with this quantity in kilograms as 6.289 or rounded kilograms like 6. If the domain is shipping data, perhaps kilograms is sufficient or more useful (less noisy) a precision for Item_Weight.
## variables transformation
- Transform complex non-linear relationships into linear relationships.Existence of a linear relationship between variables is easier to comprehend compared to a non-linear or curved relation. Transformation helps us to convert a non-linear relation into linear relation. Scatter plot can be used to find the relationship between two continuous variables. These transformations also improve the prediction. Log transformation is one of the commonly used transformation technique used in these situations
- For right skewed distribution, we take square / cube root or logarithm of variable and for left skewed, we take square / cube or exponential of variables.
- Cube root can be applied to negative values including zero. Square root can be applied to positive values including zero.
## feature crosses
- Feature crosses are a unique way to combine two or more categorical attributes into a single one. This is extremely useful a technique, when certain features together denote a property better than individually by themselves. Mathematically speaking, you are doing a cross product between all possible values of the categorical features.
Consider a feature A, with two possible values {A1, A2}. Let B be a feature with possibilities {B1, B2}. Then, a feature-cross between A & B (lets call it AB) would take one of the following values: {(A1, B1), (A1, B2), (A2, B1), (A2, B2)}. You can basically give these ‘combinations’ any names you like. Just remember that every combination denotes a synergy between the information contained by the corresponding values of A and B.
