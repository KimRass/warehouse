# Fermat's little Theorem(FlT)
-  If `p` is a prime number, for any integer `a` the number `a**p – a` is an integer multiple of p.
	- a**p ≡ a (mod p)
-  If `a` is not divisible by `p`, `a**p - 1` is an integer multiple of `p`.
	- a**(p - 1) ≡ 1 (mod p)
	
# Sequence
## Arithmetic Progression(AP)
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
## Coprime Integers
- Source: https://en.wikipedia.org/wiki/Coprime_integers
- In mathematics, two integers `a` and `b` are coprime, relatively prime or mutually prime if the only positive integer that is a divisor of both of them is `1`.
## Irreducible Fraction(= 기약분수)
- Source: https://en.wikipedia.org/wiki/Irreducible_fraction
- An irreducible fraction is a fraction in which the numerator and denominator are integers that have no other common divisors than `1` (and `−1`, when negative numbers are considered). In other words, a fraction `a/b` is irreducible if and only if `a` and `b` are coprime, that is, if `a` and `b` have a greatest common divisor of `1`.
## Numerator: 분자
## Denominator: 분모

## Whole Number: 0과 자연수
## Natural Number: 자연수
## Rational Number: 유리수
## Irrational Number: 무리수
## Real Number: 실수
## Imaginary Number: 허수
## Complex Number: 복소수