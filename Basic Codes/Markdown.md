- source : https://ko.wikipedia.org/wiki/%EC%9C%84%ED%82%A4%EB%B0%B1%EA%B3%BC:TeX_%EB%AC%B8%EB%B2%95

# Number of Prameters = 0
## {\cdot}
## \infty
## \in
## \le, \ge
## \forall
## \leftarrow, \rightarrow, \leftrightarrow
## \sim
## \overset{\underset{\mathrm{def}}{}}{=}


# Number of Prameters = 1
## \mathbb{}
### \mathbb{E}
## \vec{}
## \hat{}
## \left, \right
## \big, \bigg, \Big, \Bigg
## \begin{} \end{}
### \begin{array} \end{array}
```
\pi^{*}(a|s)=\left\{
\begin{array}{c l}	
    1, & if\ a = \underset{a \in A}{\operatorname{argmax}}Q^{*}(s, a)\\
    0, & otherwise
\end{array}\right.
```
### \begin{align} \end{align}
```
\begin{align}
V_{t+1}^{\pi}(s) &= \sum_{a \in A} \pi(a|s) \left( R_{s}^{a} + \gamma \sum_{s' \in S} P_{ss'}^{\pi}V_{t}^{\pi}(s') \right)\\
&= R_{s}^{\pi} + \gamma \sum_{s' \in S} P_{ss'}^{\pi}V_{t}^{\pi}(s')
\end{align}
```
```
$$\pi(a|s) =
\left\{
\begin{align}
&\frac{\epsilon}{|A|} + 1 - \epsilon
&&if\ a = \underset{a \in A}{\operatorname{argmax}}Q^{\pi}(s, a)\\
&\frac{\epsilon}{|A|}
&&otherwise
\end{align}
\right.$$
```



# Number of Prameters = 2
## \sum_{}^{}
## \prod_{}^{}
## \frac{}{}
```
\frac{1}{2}
```
## \underset{}{\operatorname{}}
```
\underset{a \in A}{\operatorname{argmax}}
```
