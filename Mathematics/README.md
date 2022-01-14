# Fermat's little Theorem(FlT)
- If `p` is a prime number, for any integer `a` the number `a**p – a` is an integer multiple of p. (a**p ≡ a (mod p))
	```python
	(a**p)%p == a%p
	```
- If `a` is not divisible by `p`, `a**p - 1` is an integer multiple of `p`. (a**(p - 1) ≡ 1 (mod p))
	```python
	(a**(p - 1))%p == 1
	```
	
# Sequence (= 수열)
## Arithmetic Progression (AP) (= 등차수열)
- Common Difference

# Gemometry
## Euclidean Geometry
- Given a line and a point not on the line, there exist(s) exactly one line through the given point and parallel to the given line.
## Non-Euclidean Geometry
- Given a line and a point not on the line, there exist(s) no lines (spherical) or infinitely many lines (hyperbolic) through the given point and parallel to the given line.
### Taxicab Geometry
- Source: https://en.wikipedia.org/wiki/Taxicab_geometry
- A taxicab geometry is a form of geometry in which the usual distance function or metric of Euclidean geometry is replaced by a new metric in which the distance between two points is the sum of the absolute differences of their Cartesian coordinates.
## Manhattan Distance
## Triangle
### Equilateral Triangle: 정삼각형
### Isosceles Triangle: 이등변삼각형
### Scalene Triangle
- In geometry, a scalene triangle has three sides that are all different lengths.
### Right Triangle: 직각삼각형
### Obtuse Triangle: 둔각삼각형
### Acute Triangle: 예각삼각형

# Prime Number
## Composition Number: 합성수
## Sieve of Eratosthenes
- Source: https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes
- In mathematics, the sieve of Eratosthenes is an ancient algorithm for finding all prime numbers up to any given limit. It does so by iteratively marking as composite (i.e., not prime) the multiples of each prime, starting with the first prime number, 2.
##  Bertrand's Postulate
- Source: https://en.wikipedia.org/wiki/Bertrand%27s_postulate
- In number theory, Bertrand's postulate is a theorem stating that for any integer `n > 3`, there always exists at least one prime number `p` with `n < p < 2n - 2`.
- A less restrictive formulation is: for every `n > 1` there is always at least one prime `p` such that `n < p < 2n`.
## Prime Number Theorem(PNT)
- The prime number theorem (PNT) implies that the number of primes up to `x` is roughly `x/ln(x)`, so if we replace `x` with `2x` then we see the number of primes up to `2x` is asymptotically twice the number of primes up to `x` (the terms `ln(2x)` and `ln(x)` are asymptotically equivalent). Therefore, the number of primes between `n` and `2n` is roughly `n/ln(n)` when `n` is large, and so in particular there are many more primes in this interval than are guaranteed by Bertrand's Postulate. So Bertrand's postulate is comparatively weaker than the PNT. But PNT is a deep theorem, while Bertrand's Postulate can be stated more memorably and proved more easily, and also makes precise claims about what happens for small values of `n`.
## Goldbach's Conjecture
- Source: https://en.wikipedia.org/wiki/Goldbach%27s_conjecture
- Goldbach's conjecture is one of the oldest and best-known unsolved problems in number theory and all of mathematics. It states that every even whole number greater than `2` is the sum of two prime numbers.
- The conjecture has been shown to hold for all integers less than `4*(10**18)`, but remains unproven despite considerable effort.
## Prime Factorization

# Number Theory
## Factor: 약수(인수)
### Common Factor: 공약수
## Multiple: 배수
### Common Multiple: 공배수
## Square Number
- Source: https://en.wikipedia.org/wiki/Square_number
- In mathematics, a square number or perfect square is an integer that is the square of an integer.
## Euclidean Algorithm
- Source: https://en.wikipedia.org/wiki/Euclidean_algorithm
- The Euclidean algorithm is based on the principle that **the greatest common divisor of two numbers does not change if the larger number is replaced by its difference with the smaller number.** For example, 21 is the GCD of 252 and 105 (as 252 = 21 × 12 and 105 = 21 × 5), and the same number 21 is also the GCD of 105 and 252 − 105 = 147. **Since this replacement reduces the larger of the two numbers, repeating this process gives successively smaller pairs of numbers until the two numbers become equal. When that occurs, they are the GCD of the original two numbers.**

# Fraction
- Numerator: 분자, Denominator: 분모
## Coprime Integers
- Source: https://en.wikipedia.org/wiki/Coprime_integers
- In mathematics, two integers `a` and `b` are coprime, relatively prime or mutually prime if the only positive integer that is a divisor of both of them is `1`.
## Irreducible Fraction(= 기약분수)
- Source: https://en.wikipedia.org/wiki/Irreducible_fraction
- An irreducible fraction is a fraction in which the numerator and denominator are integers that have no other common divisors than `1` (and `−1`, when negative numbers are considered). In other words, a fraction `a/b` is irreducible if and only if `a` and `b` are coprime, that is, if `a` and `b` have a greatest common divisor of `1`.

## Whole Number: 0과 자연수
## Natural Number: 자연수
## Rational Number: 유리수
## Irrational Number: 무리수
## Real Number: 실수
## Imaginary Number: 허수
## Complex Number: 복소수

# Function
## Monotonic Function
- Source: https://en.wikipedia.org/wiki/Monotonic_function
- In mathematics, a monotonic function (or monotone function) is a function between ordered sets that preserves or reverses the given order.
### Increasing Function
### Decreasing Function

# Multiset
- Source: https://en.wikipedia.org/wiki/Multiset
- In mathematics, ***a multiset (or bag, or mset) is a modification of the concept of a set that, unlike a set, allows for multiple instances for each of its elements. The number of instances given for each element is called the multiplicity of that element in the multiset.***
- In the multiset {a, a, b}, the element a has multiplicity 2, and b has multiplicity 1.
- *As with sets, and in contrast to tuples, order does not matter in discriminating multisets, so {a, a, b} and {a, b, a} denote the same multiset.*
- *The cardinality of a multiset is constructed by summing up the multiplicities of all its elements. For example, in the multiset {a, a, b, b, b, c} the multiplicities of the members a, b, and c are respectively 2, 3, and 1, and therefore the cardinality of this multiset is 6.*

# Probability Distribution
## Normal Distribution
```python
import numpy as np

np.random.normal(mean, std, size)
```
	- `loc`: Mean.
	- `scale`: Standard deviation.
	- `size`: Output shape.

## Pareto Distribution
- Source: https://en.wikipedia.org/wiki/Pareto_distribution
## Beta Distribution
- Source: https://en.wikipedia.org/wiki/Beta_distribution
- In probability theory and statistics, the beta distribution is a family of continuous probability distributions defined on the interval [0, 1] parameterized by two positive shape parameters, denoted by α and β, that appear as exponents of the random variable and control the shape of the distribution. The generalization to multiple variables is called a Dirichlet distribution.
- The formulation of the beta distribution discussed here is also known as the beta distribution of the first kind, whereas beta distribution of the second kind is an alternative name for the beta prime distribution.
- Probability Density Function (PDF)
	- ![formula](https://render.githubusercontent.com/render/math?math=\f(x; α, β)=\frac{1}{B(α, β)}x^{(α−1)}(1−x)^{(β−1)})
- Mean
	- ![formula](https://render.githubusercontent.com/render/math?math=\color{white}\large\E(X)=\frac{α}{α+β})
	- ![formula](https://render.githubusercontent.com/render/math?math=\color{white}\large\E(X)=\frac{\alpha}{\alpha+\beta})
- Variance
	- ![formula](https://render.githubusercontent.com/render/math?math=\color{white}\large\var(x)=\frac{αβ}{(α+β)^{2}(α+β+1)})
- Mode
	- ![formula](https://render.githubusercontent.com/render/math?math=\color{white}\large\\frac{α−1}{α+β−2})

## Dirichlet Distribution

# Similarities
## Cosine Similarity
```python
import numpy as np

cos_sim = np.dot(vec1, vec2)/(np.linalg.norm(vec1, ord=2)*np.linalg.norm(vec2, ord=2))
```
## Euclidean Similarity
```python
import numpy as np

euc_dist = np.linalg.norm(vec1 - vec2, ord=2)
euc_sim = 1/(1 + euc_dist)
```

# Statistics
## Correlation
## Casuality (= Cause and Effect)
- Source: https://en.wikipedia.org/wiki/Causality
- *Causality is influence by which one event, process, state or object (a cause) contributes to the production of another event, process, state or object (an effect) where the cause is partly responsible for the effect, and the effect is partly dependent on the cause.*
- ***In general, a process has many causes, which are also said to be causal factors for it, and all lie in its past. An effect can in turn be a cause of, or causal factor for, many other effects, which all lie in its future.***
## Mutually Exclusive
## Population (= 모집단) and Sample (= 표본)
- Source: https://www.scribbr.com/methodology/population-vs-sample/
- A population is the entire group that you want to draw conclusions about.
- *A sample is the specific group that you will collect data from.* The size of the sample is always less than the total size of the population.
### Sampling
- ***Ideally, a sample should be randomly selected and representative of the population. Using  probability sampling methods (such as simple random sampling or stratified sampling) reduces the risk of sampling bias and enhances both internal and external validity.***
- For practical reasons, researchers often use non-probability sampling methods. Non-probability samples are chosen for specific criteria; they may be more convenient or cheaper to access. Because of non-random selection methods, any statistical inferences about the broader population will be weaker than with a probability sample.
- Reasons for sampling
	- Sometimes it’s simply not possible to study the whole population due to its size or inaccessibility.
	- It’s easier and more efficient to collect data from a sample.
	- There are fewer participant, laboratory, equipment, and researcher costs involved.
	- Storing and running statistical analyses on smaller datasets is easier and reliable.
#### Probability Sampling
- It involves random selection, allowing you to make strong statistical inferences about the whole group.
#### Non-Probability Sampling
- It involves non-random selection based on convenience or other criteria, allowing you to easily collect data.
#### Sampling Error (= 표본 오차)
- A sampling error is the difference between a population parameter and a sample statistic. In your study, the sampling error is the difference between the mean political attitude rating of your sample and the true mean political attitude rating of all undergraduate students in the Netherlands.
- *Sampling errors happen even when you use a randomly selected sample. This is because random samples are not identical to the population in terms of numerical measures like means and standard deviations.*
- *Because the aim of scientific research is to generalize findings from the sample to the population, you want the sampling error to be low. You can reduce sampling error by increasing the sample size.*
### Parameter (= 모수) and Statistic (= 통계량)
- When you collect data from a population or a sample, there are various measurements and numbers you can calculate from the data. ***A parameter is a measure that describes the whole population. A statistic is a measure that describes the sample.***
- You can use estimation or hypothesis testing to estimate how likely it is that a sample statistic differs from the population parameter.
#### Test Statistic (= 검정 통계량)
- Source: https://en.wikipedia.org/wiki/Test_statistic
- *A test statistic is a statistic (a quantity derived from the sample) used in statistical hypothesis testing. A hypothesis test is typically specified in terms of a test statistic, considered as a numerical summary of a data-set that reduces the data to one value that can be used to perform the hypothesis test. In general, a test statistic is selected or defined in such a way as to quantify, within observed data, behaviors that would distinguish the null from the alternative hypothesis, where such an alternative is prescribed, or that would characterize the null hypothesis if there is no explicitly stated alternative hypothesis.*
## Descriptive Statistics (= 기술 통계학) and Inferential Statistics (= 추론 통계학)
- Descriptive statistics are typically distinguished from inferential statistics. ***With descriptive statistics you are simply describing what the data shows. With inferential statistics, you are trying to reach conclusions that extend beyond the immediate data alone. For instance, we use inferential statistics to try to infer from the sample data what the population might think. Or, we use inferential statistics to make judgments of the probability that an observed difference between groups is a dependable one or one that might have happened by chance in this study. Thus, we use inferential statistics to make inferences from our data to more general conditions; we use descriptive statistics simply to describe what’s going on in our data.***
## Hypothesis
### Null Hypothesis H0 (= 귀무 가설)
- Source: https://en.wikipedia.org/wiki/Null_hypothesis
- The null hypothesis is that the observed difference is due to chance alone.
- Examples: Are boys taller than girls at age eight? The null hypothesis is "they are the same average height."
- The null hypothesis is a characteristic arithmetic theory suggesting that no statistical relationship and significance exists in a set of given, single, observed variables between two sets of observed data and measured phenomena.
### Alternative Hypothesis H1 (= 대립 가설)
- *An alternative hypothesis is an opposing theory in relation to the null hypothesis.* When you create a null hypothesis, you make an educated guess whether something is true, or whether there is any relation between two phenomena.
- The alternative hypothesis always takes the opposite stance of a null hypothesis. If the null hypothesis estimates something to be true, then the alternative hypothesis estimates it to be false. The alternative hypothesis is usually the statement that you are testing when attempting to disprove the null hypothesis. *If you can gather enough data to support the alternative hypothesis, then it will replace the null hypothesis.*
## Confidence Interval (CI) (= 신뢰 구간) and Confidence Level (= 신뢰 수준)
- Source: https://en.wikipedia.org/wiki/Confidence_interval
- ***In statistics, a confidence interval (CI) is a range of estimates for an unknown parameter, defined as an interval with a lower bound and an upper bound. The interval is computed at a designated confidence level. The 95% confidence level is most common, but other levels (such as 90% or 99%) are sometimes used. The confidence level represents the long-run frequency of confidence intervals that contain the true value of the parameter. In other words, 95% of confidence intervals computed at the 95% confidence level contain the parameter, and likewise for other confidence levels.***
- ***The factors affecting the width of the CI include the confidence level, the sample size, and the variability in the sample. Larger samples produce narrower confidence intervals when all other factors are equal. Greater variability in the sample produces wider confidence intervals when all other factors are equal. A higher confidence level produces wider confidence intervals when all other factors are equal.***
- Source: https://www.scribbr.com/statistics/confidence-interval/
## Normalization
- Source: https://en.wikipedia.org/wiki/Normalization_(statistics)
- In statistics and applications of statistics, normalization can have a range of meanings. ***In the simplest cases, normalization of ratings means adjusting values measured on different scales to a notionally common scale, often prior to averaging.***
## Probability & Likelihood