- 출처 : https://dailyheumsi.tistory.com/113?category=815369
# Decision Tree
## Feature Importance
- 출처 : https://stats.stackexchange.com/questions/162162/relative-variable-importance-for-boosting
- sum up the feature importances of the individual trees, then divide by the total number of trees.
- return the fmap, which has the counts of each time a  variable was split on
- How many times was this variable split on?
## data skewness
- Parametric methods are mainly based on the assumptions on the distribution of the data. They estimate a parameter (usually mean , sd ) from the sample data and is used in the modelling framework.
Point to ponder - Mean for a normal distribution will be different than mean for a right skewed distribution hence affecting how your model performs.
In Non Parametric methods no such feature of distribution is used for modelling. Primarily in Decision trees (say CART) it takes into account which variable/split brings in maximum difference in the two branches(eg - Gini) . In such a case , the distribution does not really matter.
- A positive aspect of using the error ratio instead of the error difference is that the feature importance measurements are comparable across different problems.
# classification tree
# regression tree
### intro
- 출처 : https://www.youtube.com/watch?v=7VeUPuFGJHk&list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF&index=35
### feature selection and missing values
- 출처 : https://www.youtube.com/watch?v=wpNl-JwwplA&list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF&index=36
### algorithm
- 출처 : https://www.youtube.com/watch?v=g9c66TUylZ4&list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF&index=37
### pruning
- 출처 : https://www.youtube.com/watch?v=D0efHEJsfHo&list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF&index=38
## gradient boost
- 출처 : https://www.youtube.com/watch?v=3CC4N4z3GJc&list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF&index=45
- 출처 : https://www.youtube.com/watch?v=2xudPOBz-vs&list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF&index=46
# ensemble
- [https://lsjsj92.tistory.com/544?category=853217](https://lsjsj92.tistory.com/544?category=853217)
- https://lsjsj92.tistory.com/543?category=853217
