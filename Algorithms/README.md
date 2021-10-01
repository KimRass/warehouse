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
#### Heap
- Min Heap: Parent < Child
```python
import heqpq as hq

hq.heappush(<<Heap Object>>, <<Element>>)
hq.heqppop(<<Heap Object>>)
```
- Max Heap: Parent > Child
```python
import heqpq as hq

hq.heappush(<<Heap Object>>, -<<Element>>)
hq.heqppop(<<Heap Object>>)
```
##### Binary Heap
## Deque(Double Ended Queue)
```python
from collections import deque

deque().append()
deque().appendleft()
deque().pop()
deque().popleft()
```
## Tree
- Node = Vertex
- Edge = Link
### Binary Tree
- 최대 2개의 자식 노드
- Reference: https://gingerkang.tistory.com/86
#### Binary Search Tree
- Left < Parent < Right
```python
class Node():
    def __init__(self, value):
        self.value = value
        self.left = None
		self.right = None
		
class BinarySearchTree():
    def __init__(self, root):
        self.root = root
    def insert(self, value):
		self.cur_node = self.root
		while True:
			if value < self.cur_node.value:
				if self.cur_node.left == None:
					self.cur_node.left = Node(value)
					break
				else:
					self.cur_node = self.cur_node.left
			else:
				if self.cur_node.right == None:
					self.cur_node.right = Node(value)
					break
				else:
					self.cur_node = self.cur_node.right
					
					
root = Node(1)
bst = BinarySearchTree(root)
```
### Tree Traversal
#### Depth First Traversal
- Preorder Traverse: Parent -> Left -> Right
```python
def preorder(self, node):
    if node == None:
        pass
    else:
        print(node.value)
        preorder(node.left)
        preorder(node.right)
```
- Inorder Traverse: Left -> Parent -> Right
```python
def inorder(self, node):
    if node == None:
        pass
    else:
        inorder(node.left)
        print(node.value)
        inorder(node.right)
```
- Postorder Traverse: Left -> Right -> Parent
```python
def postorder(self, node):
    if node == None:
        pass
    else:
        postorder(node.left)
        postorder(node.right)
        print(node.value)
```
#### Breadth First Traversal
- Level-Order Traverse: Level-order
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

# Brute-Force Attack

# Recursion
- Factorial
```python
def fac(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n*fac(n - 1)
```
- Fibonacci Number
```python
def fibo(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibo(n - 1) + fibo(n - 2)
```

# Sorting

# Back Tracking

# Dynamic Programming
- Divide and Conquer: Divide -> Conquer -> Combine
## 0-1 Knapsack Problem
- Source: https://gsmesie692.tistory.com/113
- P\[i, w\] 란 i개의 보석이 있고 배낭의 무게 한도가 w일 때 최적의 이익을 의미한다.
- i번째 보석이 배낭의 무게 한도보다 무거우면 넣을 수 없으므로 i번째 보석을 뺀 i-1개의 보석들을 가지고 구한 전 단계의 최적값을 그대로 가져온다.
- 그렇지 않은 경우, i번째 보석을 위해 i번째 보석만큼의 무게를 비웠을 때의 최적값에 i번째 보석의 가격을 더한 값 or i-1개의 보석들을 가지고 구한 전 단계의 최적값 중 큰 것을 선택한다.
## Longest Increasing Subsequence
- Time complexity: $O(N^2)$
```python
arr = [0] + arr

mem = {0:0}
for i in A[1:]:
    mem[i] = max([v for k, v in mem.items() if k < i]) + 1
print(max(mem.values()))
```

# Greedy Algorithms
- Source: Source: https://www.geeksforgeeks.org/greedy-algorithms/
- Greedy is an algorithmic paradigm that builds up a solution piece by piece, always choosing the next piece that offers the most obvious and immediate benefit. So the problems where choosing locally optimal also leads to global solution are best fit for Greedy.
## Fractional Knapsack Problem

# Graph Theory
## Dijkstra's
## Floyd–warshall
# Graph Traversal
## Depth First Search
```python
stack = [...]
visited = {...}
while stack:
	start = stack.pop()
	if visited[start] == False:
		visited[start] = True
		stack.append(graph[start])
```
## Breadth First Search
```python
from collections import deque

dq = deque([...])
visited = {...}
while dq:
	start = dq.popleft()
	if visited[start] == False:
		visited[start] = True
		dq.append(graph[start])
```

# Two-Pointers
- Time complexity: $O(N)$
```python
arr = sorted(arr)

left = 0
right = len(arr)
cnt = 0
while left < right:
	if f(left, right) < x:
		left += 1
	elif f(left, right) > x:
		right -= 1
	else:
		left += 1
		cnt += 1
```

# Prefix Sum
```python
pref_sum = [0]
temp = 0
for i in arr:
    temp += i
    pref_sum.append(temp)
```

# Binary Search
```python
arr = sorted(arr)

left = 0
right = N - 1
while left <= right:
    mid = (left + right)//2

    if arr[mid] == tar:
        ...
		break
    elif arr[mid] > tar:
        right = mid - 1
    else:
        left = mid + 1
```
```python
from bisect import bisect_left, bisect_right

arr = sorted(arr)

print(... if bisect_right(arr, tar) == bisect_left(arr, tar) else ...)
```
```python
def bisect_left(arr, tar):
    left = 0
    right = len(arr)
    while left < right:
        mid = (left + right)//2

        if arr[mid] < tar:
            left = mid + 1
        else:
            right = mid
    return left
    
def bisect_right(arr, tar):
    left = 0
    right = len(arr)
    while left < right:
        mid = (left + right)//2

        if arr[mid] > tar:
            right = mid
        else:
            left = mid + 1
    return left
```
## Longest Increasing Subsequence
- Time complexity: $O(N\logN)$

# Implementation
- 풀이를 떠올리는 것은 쉽지만 소스코드로 옮기기 어려운 문제.
## Exhaustive Search

## Simulation

# Hash

# Union-Find