Written by KimRass
# Data Structures
## Stack
- LIFO(Last-In-First-Out)
```python
List.append()
List.pop()
```
### VPS(Valid Parenthesis String)
### NGE(Next Greater Element)
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
- Priority queue can be implemented by heap data structure.
## Deque(Double Ended Queue)
```python
from collections import deque

deque().append()
deque().appendleft()
deque().pop()
deque().popleft()
```
## Graph
- Source: https://en.wikipedia.org/wiki/Graph_theory
- Graph is made up of vertices (also called nodes or points) which are connected by edges (also called links or lines).
- Undirected graph is a graph where edges link two vertices symmetrically, and directed graph is a the one where edges link two vertices asymmetrically.
- Source: https://www.acmicpc.net/problem/15681
- 간선의 집합에 변함이 없는 한, 그래프는 얼마든지 원하는 대로 다시 그릴 수가 있다.
- Source: https://en.wikipedia.org/wiki/Cycle_(graph_theory)
- A cycle in a graph is a non-empty trail in which the only repeated vertices are the first and last vertices. A directed cycle in a directed graph is a non-empty directed trail in which the only repeated vertices are the first and last vertices.
### Dijkstra's
### Floyd–warshall
## Graph Traversal
### Depth First Search
```python
stack = [...]
visited = {...}
while stack:
	start = stack.pop()
	if visited[start] == False:
		visited[start] = True
		for end in graph[start]:
			stack.append(end)
```
### Breadth First Search
```python
from collections import deque

dq = deque([...])
visited = {...}
while dq:
	start = dq.popleft()
	if visited[start] == False:
		visited[start] = True
		for end in graph[start]:
			dq.append(end)
```
- Geodesic Distance: Shortest path between the vertices.
```python
from collections import deque

dq = deque([...])
visited = {...}
while dq:
	start, cur_dist = dq.popleft()
	if visited[start] == False:
		visited[start] = True
		for end, dist in graph[start]:
			dq.append((end, cur_dist + dist))
```
### Tree
- Source: https://en.wikipedia.org/wiki/Cycle_(graph_theory)
- A graph without cycles is called an acyclic graph. A directed graph without directed cycles is called a Directed Acyclic Graph(DAG, tree).
- Source: https://www.acmicpc.net/problem/15681
- 임의의 두 정점 U와 V에 대해, U에서 V로 가는 최단경로는 유일하다.
아무 정점이나 잡고 부모와의 연결을 끊었을 때, 해당 정점과 그 자식들, 그 자식들의 자식들… 로 이루어진 부분그래프는 트리가 된다.
- There are (n - 1) edges in a tree with n nodes.
- Source: https://en.wikipedia.org/wiki/Tree_(data_structure)
- Ancestor: A node reachable by repeated proceeding from child to parent.
- Descendant, Subchild: A node reachable by repeated proceeding from parent to child.
- Degree: For a given node, its number of children. A leaf has necessarily degree zero.
	- Degree of a tree: Maximum degree of a node in the tree.
- Distance: The number of edges along the shortest path between two nodes.
- Level: The level of a node is the number of edges along the unique path between it and the root node.
- Width: The number of nodes in a level.
- Bredth: The number of leaves.
- Diameter of a tree: The longest path between any two nodes (which MAY or MAY NOT PASS through the root Node).
```python
# Choose a node arbitrarily.
node = list(tree)[0]

stack = [(node, 0)]
visited = {i:False for i in tree}

# 지름을 구성하는 2개의 노드 중 하나를 구합니다.
max_dist = 1
while stack:
    start, dist1 = stack.pop()
	if visited[start] == False:
		visited[start] = True
		if dist1 > max_dist:
			max_dist = dist1
			max_node = start
		for end, dist2 in tree[start]:
			stack.append((end, dist1 + dist2))

stack = [(max_node, 0)]
visited = {i:False for i in tree}
visited[node] = True

# Get the diameter.
diam = 1
while stack:
    start, dist1 = stack.pop()
	if visited[start] == False:
		visited[start] = True
		if dist1 > diam:
			diam = dist1
		for end, dist2 in tree[start]:
            stack.append((end, dist1 + dist2))
```
#### Binary Tree
- A tree whose elements have at most 2 children.
##### Complete Binary Tree
- A binary tree in which all the levels are completely filled except possibly the lowest one, which is filled from the left.
##### Heap
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
###### Binary Heap
#### Full Binary Tree
- A binary tree in which every node has 0 or 2 children.
#### Binary Search Tree
- Reference: https://gingerkang.tistory.com/86
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
#### Depth First Search
- Preorder Traverse: Parent -> Left -> Right
```python
def preorder(node):
    if node != None:
        left, right = tree[node]
        print(node)
        preorder(left)
        preorder(right)
```
```python
def preorder(self, node):
    if node != None:
		print(node.value)
        preorder(node.left)
        preorder(node.right)
```
- Inorder Traverse: Left -> Parent -> Right
```python
def inorder(node):
    if node != None:
        left, right = tree[node]
        inorder(left)
        print(node)
        inorder(right)
```
```python
def inorder(self, node):
    if node != None:
        inorder(node.left)
        print(node.value)
        inorder(node.right)
```
- Postorder Traverse: Left -> Right -> Parent
```python
def postorder(node):
    if node != None:
        left, right = tree[node]
        postorder(left)
        postorder(right)
        print(node)
```
```python
def postorder(self, node):
    if node != None:
        postorder(node.left)
        postorder(node.right)
        print(node.value)
```
#### Breadth First search
- Level-Order Traverse: Level-order
### Trie
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

# Exhaustive Search
## Brute-Force Attack
## Backtracking
- Source: https://en.wikipedia.org/wiki/Backtracking
- Backtracking is a general algorithm for finding solutions to some computational problems, notably constraint satisfaction problems, that incrementally builds candidates to the solutions, and abandons a candidate ("backtracks") as soon as it determines that the candidate cannot possibly be completed to a valid solution.
- N-Queen

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

# Sort
- Stable sorting algorithm: Sorting algorithm which maintains the relative order of records with equal keys (i.e. valuies)).
- Comparison sorting algorithm
## Time Complexity of O(n^2)
### Bubble Sort
- One of stable sorting algorithms
- One of comparison sorting algorithms
```python
for i in range(len(arr), 0, -1):
    for j in range(i - 1):
        if arr[j] > arr[j + 1]:
            arr[j], arr[j + 1] = arr[j + 1], arr[j]
```
### Selection Sort
- Not a stable sorting algorithm
- One of comparison sorting algorithms
```python
for i in range(len(arr)):
    min_j = i
    for j in range(i, len(arr)):
        if arr[j] < arr[min_j]:
            min_j = j
    arr[i], arr[min_j] = arr[min_j], arr[i]
```
### Insertion Sort
- One of stable sorting algorithms
- One of comparison sorting algorithms
```python
for i in range(1, len(arr)):
    for j in range(i, 0, -1):
        if arr[j - 1] > arr[j]:
            arr[j], arr[j - 1] = arr[j - 1], arr[j]
		else:
			break
```
## Time Complexity of O(n\*logn)
### Merge Sort
- One of stable sorting algorithms
- One of comparison sorting algorithms
```python
def merge_sort(arr):
    if len(arr) == 1:
        return arr
    else:
        left = merge_sort(arr[:len(arr)//2])
        right = merge_sort(arr[len(arr)//2:])
        
        new_arr = list()
        i = 0
        j = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                new_arr.append(left[i])
                i += 1
            else:
                new_arr.append(right[j])
                j += 1
        new_arr += left[i:]
        new_arr += right[j:]
        
        return new_arr
```
### Heap Sort
### Quick Sort
## ETC
### Counting Sort
- Not a comparison sorting algorithm
- Time complexity of O(n + k)(k is the range of elements of the array)
- It assumes that all the elements of the array are positive integers.
```python
maxim = max(arr)
count = {i:0 for i in range(1, maxim + 1)}

for elm in arr:
    count[elm] += 1

new_arr = list()
for k, v in count.items():
    for _ in range(v):
        new_arr.append(k)
```

# Back Tracking

# Dynamic Programming
- Divide and Conquer: Divide -> Conquer -> Combine
## 0-1 Knapsack Problem
- Source: https://gsmesie692.tistory.com/113
- P\[i, w\] 란 i개의 보석이 있고 배낭의 무게 한도가 w일 때 최적의 이익을 의미한다.
- i번째 보석이 배낭의 무게 한도보다 무거우면 넣을 수 없으므로 i번째 보석을 뺀 i-1개의 보석들을 가지고 구한 전 단계의 최적값을 그대로 가져온다.
- 그렇지 않은 경우, i번째 보석을 위해 i번째 보석만큼의 무게를 비웠을 때의 최적값에 i번째 보석의 가격을 더한 값 or i-1개의 보석들을 가지고 구한 전 단계의 최적값 중 큰 것을 선택한다.
## LIS(Longest Increasing Subsequence)
- Time complexity: O(n^2)
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

# Two-Pointers
- Time complexity: O(n)
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
- Time complexity: O(n\*logn)

# Implementation
- 풀이를 떠올리는 것은 쉽지만 소스코드로 옮기기 어려운 문제.
## Exhaustive Search

## Simulation

# Hash

# Union-Find