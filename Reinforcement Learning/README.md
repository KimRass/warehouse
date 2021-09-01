# MDP; Markov Decision Process

## 1. Markov Property
- 미래의 State는 과거와 무관하게 현재의 State만으로 결정된다.
$$
P(s_{t+1}|s_{t}) = P(s_{t+1}|s_{t}, s_{t-1}, ... , s_{0})
$$

## 2. MP; Markov Process(= Markov Chain)
- Time Interval이 discrete하고 Markov Property를 나타내는 확률 과정

### (1) State Transition Matrix $P$ 
$$
P_{ss'} \overset{\underset{\mathrm{def}}{}}{=} P(S_{t+1}=s'|S_{t}=s)
$$

## 3. MRP; Markov Reward Process
- MP에 Reward 개념을 추가한 확률 과정
- MP에서는 각 state별 transition 확률이 주어져 있다할 뿐이지, 이 state에서 다음 state로 가는 것이 얼마나 가치가 있는지는 알 수 없습니다.

### (1) Return $G_{t}$
- 시점 t에서의 전체 미래에 대한 sum of discounted reward.
$$
G_{t} \overset{\underset{\mathrm{def}}{}}{=} R_{t+1} + \gamma R_{t+2} + ... = \sum_{k=0}^\infty \gamma^{k}R_{t+k+1}
$$

### (2) State Value Function $V(s)$
- Expectation of Return $G_{t}$
$$
V(s) \overset{\underset{\mathrm{def}}{}}{=} E[G_{t}|S_{t}=s]
$$

## 4. MDP; Markov Decision Process
- MRP에 Action 개념을 추가한 확률 과정
- 이전에는 State가 Transition Probability에 따라 변했다면, 이제는 Action에 따라 변함.

### (1) State Trainsition Matrix P
$$
P_{ss'}^{a} \overset{\underset{\mathrm{def}}{}}{=} P(S_{t+1}=s'|S_{t}=s, A_{t}=a)
$$

### (2) Policy $\pi$
$$\pi(a|s) \overset{\underset{\mathrm{def}}{}}{=} P(A_{t}=a|S_{t}=s)$$

$$P_{ss'}^{\pi} = \sum_{a \in A} \pi(a|s) P_{ss'}^a$$

$$R_{s}^{\pi} = \sum_{a \in A} \pi(a|s) R_{s}^a$$

### (3) State Value Function $V_{\pi}(s)$
- The expected return starting from state $s$, and then following policy $\pi$
$$
V_{\pi}(s) \overset{\underset{\mathrm{def}}{}}{=} E_{\pi}[G_{t}|S_{t}=s]
$$

#### Bellman (Expectation) Equation
$$
V_{\pi}(s) = E_{\pi}[R_{t+1} + \gamma V(S_{t+1})|S_{t}=s]
$$

### (4) State-Action Value Function $Q_{\pi}$
- The expected return starting from state $s$, taking action $a$, and then following policy $\pi$
$$
Q_{\pi}(s, a) \overset{\underset{\mathrm{def}}{}}{=} E_{\pi}[G_{t}|S_{t}=s, A_{t}=a]
$$

#### Bellman (Expectation) Equation
$$
Q_{\pi}(s, a) = E_{\pi}[R_{t+1} + \gamma Q_{\pi}(S_{t+1}, A_{t+1})|S_{t}=s, A_{t}=a]
$$

### (5) 관계식
#### State-Action Value Function to State Value Function
$$
\begin{align}
V_{\pi}(s) &= \sum_{a \in A} \pi(a|s) Q_{\pi}(s, a)\\
&= \sum_{a \in A} \pi(a|s) \left( R_{s}^{a} + \gamma \sum_{s' \in S} P_{ss'}^{a}V_{\pi}(s') \right)\\
&= R_{s}^{\pi} + \gamma \sum_{s' \in S} P_{ss'}^{\pi}V_{\pi}(s')
\end{align}
$$

#### State Value Function to State-Action Value Function
$$\begin{align}
Q_{\pi}(s, a) &= R_{s}^{a} + \gamma \sum_{s' \in S} P_{ss'}^{a}V_{\pi}(s')\\
&= R_{s}^{a} + \gamma \sum_{s' \in S} P_{ss'}^{a} \sum_{a' \in A} \pi(a'|s')Q_{\pi}(s',a')
\end{align}$$

### (6) Optimal Value Function
- "MDP를 풀었다" = "Optimal Value Function 혹은 그것을 만드는 $\pi$를 찾았다"
#### Optimal State Value Function $V^{*}$
$$
V^{*}(s) = \max_{\pi} V_{\pi}(s)
$$
#### Optimal State-Action Value Function $Q^{*}$
$$
Q^{*}(s, a) = \max_{\pi} Q_{\pi}(s, a)
$$

### (7) Optimal Policy Threom
- 어떤 MDP에서도 다음이 성립한다.
$$
\pi' \ge \pi \leftrightarrow V_{\pi'}(s) \ge V_{\pi}(s), \forall s \in S
$$

$$
\pi^{*} \ge \pi, \forall \pi
$$

$$
V_{\pi^{*}}(s) = V^{*}(s),\ Q_{\pi^{*}}(s, a) = Q^{*}(s, a)
$$
- $Q^{*}$를 알고 있다면 Optimal Policy $\pi^{*}$를 다음 방법으로 찾을 수 있다.
$$\pi^{*}(a|s)=\left\{
\begin{array}{c l}	
    1, & if\ a = \underset{{a \in A}}{\operatorname{argmax}}Q^{*}(s, a)\\
    0, & otherwise
\end{array}\right.$$

### (8) BOE; Bellman Optimality Equation
- Linear Equation이 아니므로 일반해가 존재하지 않는다.
$$
V^{*}(s) = \sum_{a \in A} \pi^{*}(a|s) Q^{*}(s, a) = \max_{a \in A}Q^{*}(s, a)
$$
$$
Q^{*}(s, a) = R_{s}^{a} + \gamma \sum_{s' \in S} P_{ss'}^{a}V^{*}(s')
$$
$$
V_{\pi}(s) = \max_{a \in A} \left( R_{s}^{a} + \gamma \sum_{s' \in S} P_{ss'}^{a}V^{*}(s') \right)
$$
$$
Q_{\pi}(s, a) = R_{s}^{a} + \gamma \sum_{s' \in S} P_{ss'}^{a}\max_{a' \in A}Q^{*}(s', a')
$$