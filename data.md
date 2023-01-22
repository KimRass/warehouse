# Header
- Reference: https://www.webopedia.com/definitions/header/
- In many disciplines of computer science, a header is a unit of information that precedes a data object.

# Dataset & Data Set
- Reference: https://english.stackexchange.com/questions/2120/which-is-correct-dataset-or-data-set
- Dataset for certain datasets
- Data set for any set for data in general.

# Data Density (or Sparsity)
- Reference: https://datascience.foundation/discussion/data-science/data-sparsity
- In a database, sparsity and density describe the number of cells in a table that are empty (sparsity) and that contain information (density), though sparse cells are not always technically empty—they often contain a "0” digit.
## Sparse Matrix & Dense Matrix
- Reference: https://en.wikipedia.org/wiki/Sparse_matrix
- ***A sparse matrix or sparse array is a matrix in which most of the elements are zero.*** *There is no strict definition regarding the proportion of zero-value elements for a matrix to qualify as sparse but a common criterion is that the number of non-zero elements is roughly equal to the number of rows or columns.* ***By contrast, if most of the elements are non-zero, the matrix is considered dense. The number of zero-valued elements divided by the total number of elements is sometimes referred to as the sparsity of the matrix.***
### Compressed Sparse Row (CSR)
- *The compressed sparse row (CSR) or compressed row storage (CRS) or Yale format represents a matrix M by three (one-dimensional) arrays, that respectively contain nonzero values, the extents of rows, and column indices.*
```python
from scipy.sparse import csr_matrix

vals = [2, 4, 3, 4, 1, 1, 2]
rows = [0, 1, 2, 2, 3, 4, 4]
cols = [0, 2, 5, 6, 14, 0, 1]
sparse_mat = csr_matrix((vals,  (rows,  cols)))
dense_mat = sparse_mat.todense()
```

# Categories of Variables
- Continuous
- Categorical

# Metadata
- Reference: https://en.wikipedia.org/wiki/Metadata
- Metadata is "data that provides information about other data", but not the content of the data, such as the text of a message or the image itself.

# Outliers
## Types of Outliers
- Reference: https://adataanalyst.com/machine-learning/comprehensive-guide-feature-engineering/
- Univariate outliers can be found when we look at distribution of a single variable.
Multi-variate outliers are outliers in an n-dimensional space. In order to find them, you have to look at distributions in multi-dimensions.
### Data Entry Errors
- Human errors such as errors caused during data collection, recording, or entry can cause outliers in data.
### Data Processing Errors
- It is possible that some manipulation or extraction errors may lead to outliers in the dataset.
### Measurement Errors
- It is the most common Reference of outliers. This is caused when the measurement instrument used turns out to be faulty. 
E## Experimental Errors
- For example: In a 100m sprint of 7 runners, one runner missed out on concentrating on the ‘Go’ call which caused him to start late. Hence, this caused the runner’s run time to be more than other runners.
### Intentional Outliers
- For example: Teens would typically under report the amount of alcohol that they consume. Only a fraction of them would report actual value. Here actual values might look like outliers because rest of the teens are under reporting the consumption.
### Sampling Errors
- For instance, we have to measure the height of athletes. By mistake, we include a few basketball players in the sample.
### Natural Outliers
## Outliers Treatment
### Deleting
### Transforming
- Natural log of a value reduces the variation caused by extreme values.
- Decision Tree algorithm allows to deal with outliers well due to binning of variable. We can also use the process of assigning weights to different observations.
### Binning
- Decision Tree algorithm allows to deal with outliers well due to binning of variable. We can also use the process of assigning weights to different observations.
### Imputing
- We can use mean, median, mode imputation methods. Before imputing values, we should analyse if it is natural outlier or artificial. If it is artificial, we can go with imputing values. We can also use statistical model to predict values of outlier.
### Treating Separately
- If there are significant number of outliers, we should treat them separately in the statistical model. One of the approach is to treat both groups as two different groups and build individual model for both groups and then combine the output.