# Seed
```python
random.seed()

np.random.seed()

tf.random.set_seed()
```

# Sample
## Sample from [0, 1)
```python
# Returns a random number in [0, 1).
random.random()

# Create an array of the given shape and populate it with random samples from a uniform distribution over [0, 1).
np.random.rand()
```
## Sample from Integers
```python
# Return a random integer N such that `a` <= N <= `b`.
# Alias for `randrange(a, b + 1)`.
random.randint(a, b)

# Return random integers from `low` (inclusive) to `high` (exclusive).
# Return random integers from the "discrete uniform" distribution of the specified dtype in the "half-open" interval [`low`, `high`). If `high` is `None` (the default), then results are from [0, `low`).
np.random.randint(low, [high], size)
```
## Sample from Sequence
```python
random.sample(sequence, k)

random.choices(sequence, k, [weights])

# Generates a random sample from a given 1-D array.
# `replace`: (bool)
# `p`: The probabilities associated with each entry in a. If not given, the sample assumes a uniform distribution over all entries in a
np.random.choice(size, [replace], [p])
```

# Shuffle
```python
# In-place function
random.shuffle()
```

# Normal Distribution
```python
tf.random.normal()
```