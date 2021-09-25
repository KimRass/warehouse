Written by KimRass
# Data Structures
## Stack
- LIFO(Last-In-First-Out)
```python
List.append()
List.pop()
```
## Queue
- LILO(Last-In-Last-Out)
- 데이터가 들어오는 위치는 가장 뒤(Rear 또는 Back이라고 한다.)에 있고, 데이터가 나가는 위치는 가장 앞(Front라고 한다.)
- `List.pop(0)` 사용 시 첫 번째 element를 pop한 후 나머지 elements의 Index를 1칸씩 당기는 과정에서 O(n)의 계산량이 발생한다.(Source: https://www.acmicpc.net/board/view/47845)
```python
from collections import deque
deque().append()
deque().popleft()
```
### Priority Queue
## Deque(Double Ended Queue)
```python
from collections import deque
deque().append()
deque().appendleft()
deque().pop()
deque().popleft()
```
## Heap
```python
import heqpq as hq
hq.heappush(<<Heap Object>>, <<Element>>)
hq.heqppop(<<Heap Object>>)
```
## Trie
- Source: https://velog.io/@gojaegaebal/210126-%EA%B0%9C%EB%B0%9C%EC%9D%BC%EC%A7%8050%EC%9D%BC%EC%B0%A8-%ED%8A%B8%EB%9D%BC%EC%9D%B4Trie-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98-%EA%B0%9C%EB%85%90-%EB%B0%8F-%ED%8C%8C%EC%9D%B4%EC%8D%AC%EC%97%90%EC%84%9C-%EA%B5%AC%ED%98%84%ED%95%98%EA%B8%B0feat.-Class
```python
class Node():
    def __init__(self, key, flag=None):
        self.key = key
        self.flag = flag
        self.child = dict()
        
class Trie():
    def __init__(self):
        self.head = Node(None)
    def insert(self, word):
        cur_node = self.head
        for char in word:
            if char not in cur_node.child:
                cur_node.child[char] = Node(char)
            cur_node = cur_node.child[char]
        cur_node.flag = word
    def search(self, word):
        cur_node = self.head
        for char in word:
            if char in cur_node.child:
                cur_node = cur_node.child[char]
            else:
                return False
        if cur_node.flag == word:
            return True
        else:
            return False
    def startwith(self, word):
        cur_node = self.head
        for char in word:
            if char in cur_node.child:
                cur_node = cur_node.child[char]
            else:
                return None
        cur_nodes = [cur_node]
        next_nodes = list()
        words = list()
        while cur_nodes:
            for node in cur_nodes:
                if node.flag != None:
                    words.append(node.flag)
                next_nodes.extend(list(node.child.values()))
            cur_nodes = next_nodes
            next_nodes = list()
        return words
```

# Divide and Conquer
- Divide -> Conquer -> Combine

# Brute-Force Attack

# Recursion

# Sorting

# Back Tracking

# Dynamic Programming
## 0-1 Knapsack Problem
- Source: https://gsmesie692.tistory.com/113
- P\[i, w\] 란 i개의 보석이 있고 배낭의 무게 한도가 w일 때 최적의 이익을 의미한다.
- i번째 보석이 배낭의 무게 한도보다 무거우면 넣을 수 없으므로 i번째 보석을 뺀 i-1개의 보석들을 가지고 구한 전 단계의 최적값을 그대로 가져온다.
- 그렇지 않은 경우, i번째 보석을 위해 i번째 보석만큼의 무게를 비웠을 때의 최적값에 i번째 보석의 가격을 더한 값 or i-1개의 보석들을 가지고 구한 전 단계의 최적값 중 큰 것을 선택한다.
## Longest Increasing Subsequence

# Greedy Algorithms
- Source: Source: https://www.geeksforgeeks.org/greedy-algorithms/
- Greedy is an algorithmic paradigm that builds up a solution piece by piece, always choosing the next piece that offers the most obvious and immediate benefit. So the problems where choosing locally optimal also leads to global solution are best fit for Greedy.
## Fractional Knapsack Problem

# Graph Traversal
## Depth First Search
## Breadth First Search

# Two Pointers

# Tree

# Implementation
- 풀이를 떠올리는 것은 쉽지만 소스코드로 옮기기 어려운 문제.
## Exhaustive Search

## Simulation