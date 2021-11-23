Written by KimRass
# Data Structures
## Stack
- LIFO(Last-In-First-Out)
```python
List.append()
List.pop()
```
- VPS(Valid Parenthesis String)
- NGE(Next Greater Element)
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
### Depth First Search
```python
while stack:
	cur_node = stack.pop()
	if visited[cur_node] == False:
		visited[cur_node] = True
		stack.extend(graph[cur_node])
```
### Breadth First Search
```python
from collections import deque

while dq:
	cur_node = dq.popleft()
	if visited[cur_node] == False:
		visited[cur_node] = True
		dq.extend(graph[cur_node])
```
- 간선에 가중치가 없다면 Shortest path problem에 적용 가능합니다.
```python
from collections import deque

dq = deque([...])
visited = {...}
while dq:
	cur_node, cur_dist = dq.popleft()
	if visited[cur_node] == False:
		visited[cur_node] = True
		for end, dist in graph[cur_node]:
			dq.append((end, cur_dist + dist))
```
### Dijkstra Algorithm
- Time complexity: O((V + E)logV)
- 각 간선에 가중치가 있다면 사용 가능합니다. 간선들 중 단 하나라도 가중치가 음수이면 이 알고리즘은 사용할 수 없습니다.
```python
import math
import heapq as hq

# 지금까지 알려진, 출발지에서 해당 노드로 가는 최단 거리를 기록하는 Dictionary를 만듭니다.
min_dists = {i:0 if i == start else math.inf for i in range(1, V + 1)}
hp = [(0, start)]
while hp:
    cur_dist, cur_node = hq.heappop(hp)
    for dist, next_node in graph[cur_node]:
		# 지금까지 알려진 `next_node`까지의 최단 거리보다 `cur_node`를 거쳐서 `next_node`로 가는 거리가 더 짧다면 그 값으로 `min_dists[next_node]`를 업데이트합니다.
        if cur_dist + dist < min_dists[next_node]:
            min_dists[next_node] = cur_dist + dist
            hq.heappush(hp, (min_dists[next_node], next_node))
```
### Bellman–Ford Algorithm
- Time complexity: O(VE)
- Source: https://en.wikipedia.org/wiki/Bellman%E2%80%93Ford_algorithm
- The Bellman–Ford algorithm is an algorithm that computes shortest paths from a single source vertex to all of the other vertices in a weighted digraph.
- It is slower than Dijkstra's algorithm for the same problem, but more versatile, as it is capable of handling graphs in which some of the edge weights are negative numbers.
- If a graph contains a "negative cycle" (i.e. a cycle whose edges sum to a negative value) that is reachable from the source, then there is no cheapest path: any path that has a point on the negative cycle can be made cheaper by one more walk around the negative cycle. In such a case, the Bellman–Ford algorithm can detect and report the negative cycle.
- Source: https://velog.io/@younge/Python-%EC%B5%9C%EB%8B%A8-%EA%B2%BD%EB%A1%9C-%EB%B2%A8%EB%A7%8C-%ED%8F%AC%EB%93%9CBellman-Ford-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98
- 음수 사이클 존재의 여부를 알 수 있다.
- 음수 사이클 안에서 무한으로 뺑뺑이 도는 경우를 알 수 있는 방법은, 그래프 정점의 개수를 V라고 할 때 인접 간선을 검사하고 거리 값을 갱신하는 과정을 V-1번으로 제한하면 가능해진다. 그래프의 시작 정점에서 특정 정점까지 도달하기 위해 거쳐 가는 최대 간선 수는 V-1개이기 때문에 V번째 간선이 추가되는 순간 사이클이라고 판단할 수 있게 된다.
- 위 과정을 모두 마치고 난 후 거리가 갱신되는 경우가 생긴다면 그래프에 음수 사이클이 존재한다는 것이다.
- Source: https://deep-learning-study.tistory.com/587
- 다익스트라와의 차이점은 매 반복마다 모든 간선을 확인한다는 것입니다. 다익스트라는 방문하지 않는 노드 중에서 최단 거리가 가장 가까운 노드만을 방문했습니다.
```python
import math

min_dists = {i:0 if i == start else math.inf for i in range(1, V + 1)}
breaker1 = False
breaker2 = False
for i in range(V):
    for cur_node in range(1, V + 1):
        for dist, next_node in graph[cur_node]:
            # 지금까지 알려진 `next_node`로의 최단 거리보다 `cur_node`를 거쳐서 가는 경로가 더 짧다면 이 값으로 `min_dists[next_node]`를 업데이트합니다.
            if dist + min_dists[cur_node] < min_dists[next_node]:
                min_dists[next_node] = dist + min_dists[cur_node]
                # `V - 1`번 `min_dists`를 업데이트한 이후에도 또 업데이트가 된다면 Negative cycle이 존재하는 것입니다.
                if i == V - 1:
                    breaker1 = True
                    break
        if breaker1 == True:
            breaker2 = True
            break
    if breaker2 == True:
        print(-1)
        break
else:
    ...
```
### Floyd–Warshall Algorithm
- Time complexity: O(V^3)
- Source: https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm
- Floyd–Warshall algorithm is an algorithm for finding shortest paths in a directed weighted graph with positive or negative edge weights (but with no negative cycles).
- A single execution of the algorithm will find the lengths (summed weights) of shortest paths between all pairs of vertices. Although it does not return details of the paths themselves, it is possible to reconstruct the paths with simple modifications to the algorithm.
```python
import math
from itertools import product

min_dists = {(i, j):0 if i == j else math.inf for i, j in product(range(1, V + 1), range(1, V + 1))}

for btw in range(1, V + 1):
    for start in range(1, V + 1):
        for end in range(1, V + 1):
            if min_dists[(start, btw)] + min_dists[(btw, end)] < min_dists[(start, end)]:
                min_dists[(start, end)] = min_dists[(start, btw)] + min_dists[(btw, end)]
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
- Diameter of a tree: The longest path between any two nodes (which may or may not pass through the root node).
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
- Subtree
	```python
	def subtree(cur_node) :
		size[cur_node] = 1
		for next_node in tree[cur_node]:
			if visited[next_node] == False:
				visited[next_node] = True
				subtree(next_node)
				size[cur_node] += size[next_node]
	```
#### Binary Tree
- A tree whose elements have at most 2 children.
##### Complete Binary Tree
- A binary tree in which all the levels are completely filled except possibly the lowest one, which is filled from the left.
##### Full Binary Tree
- A binary tree in which every node has 0 or 2 children.
##### Binary Search Tree
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
#### Quadtree
- Source: https://en.wikipedia.org/wiki/Quadtree
- A quadtree is a tree data structure in which each internal node has exactly four children. Quadtrees are the two-dimensional analog of octrees and are most often used to partition a two-dimensional space by recursively subdividing it into four quadrants or regions.
```python
from itertools import product

def quadtree(arr):
    if len(arr) == 1:
        return arr[0][0]
    else:
        n = len(arr)//2
        for i, j in product(range(len(arr)), range(len(arr))):
            if arr[i][j] != arr[0][0]:
                return f"({quadtree([i[:n] for i in arr[:n]]) + quadtree([i[n:] for i in arr[:n]]) + quadtree([i[:n] for i in arr[n:]]) + quadtree([i[n:] for i in arr[n:]])})"
        else:
            return arr[0][0]
```
#### Octree
### Depth First Search
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
### Breadth First search
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
- Source: https://jiwon-coding.tistory.com/34
- 길을 가다가 이 길이 아닌 것 같으면 왔던 길로 되돌아가 다른 경로로 진행
- 보통 재귀로 구현하며 조건이 맞지 않으면 종료한다.
- DFS(깊이 우선 탐색) 기반
- Source: https://namu.wiki/w/%EB%B0%B1%ED%8A%B8%EB%9E%98%ED%82%B9
- 상태공간을 트리로 나타낼 수 있을 때 적합한 방식이다.
- DFS를 절대 쓰면 안되는 경우가 있는데, 트리의 깊이가 무한대가 될 때이다. 미로찾기에서 루프(회로)가 발생하는 경우, DFS는 이 가지를 탈출할 수 없게 된다. 물론 중복검사를 막기 위한 장치를 넣을 수도 있지만, 그럴 바에는 BFS가 편하다.
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
## Bubble Sort
- Time Complexity: O(n^2)
- One of stable sorting algorithms
- One of comparison sorting algorithms
```python
for i in range(len(arr), 0, -1):
    for j in range(i - 1):
        if arr[j] > arr[j + 1]:
            arr[j], arr[j + 1] = arr[j + 1], arr[j]
```
## Selection Sort
- Time Complexity: O(n^2)
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
## Insertion Sort
- Time Complexity: O(n^2)
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
## Merge Sort
- Time Complexity: O(nlogn)
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
## Heap Sort
- Time Complexity: O(nlogn)
## Quick Sort
- Time Complexity: O(nlogn)
## Counting Sort
- Not a comparison sorting algorithm
- Time complexity: O(n + k)(k is the range of elements of the array)
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

# Dynamic Programming
```python
mem = dict()
def func(n):
	if n not in mem:
		mem[n] = ...
	return mem[n]
```
- Divide-and-Conquer & Memoization
- 0-1 Knapsack Problem
	- References: https://gsmesie692.tistory.com/113
	```python
	mem = dict()
	# largest_value(idx_bef, max_w): Index `idx_bef` 전까지의 Items를 가지고 무게 `max_w` 이하로 만들 수 있는 최대 Value.
	def largest_value(idx_bef, max_w):
		if (idx_bef, max_w) not in mem:
			if idx_bef == 0:
				if items[0][0] <= max_w:
					mem[(idx_bef, max_w)] = items[0][1]
				else:
					mem[(idx_bef, max_w)] = 0
			else:
				mem[(idx_bef, max_w)] = max(largest_value(idx_bef - 1, max_w - items[idx_bef][0]) + items[idx_bef][1], largest_value(idx_bef - 1, max_w)) if items[idx_bef][0] <= max_w else largest_value(idx_bef - 1, max_w)
		return mem[(idx_bef, max_w)]
	```
- LIS(Longest Increasing Subsequence)
	- Time complexity: O(n^2)
	```python
	arr = [0] + arr

	# `mem[i]`: i를 가장 마지막 원소로 갖는 LIS의 길이.
	mem = {0:0}
	for i in arr[1:]:
		# i를 마지막 원소로 갖는 LIS의 길이는, i보다 작은 값을 마지막 원소로 갖는 LIS의 길이에 1을 더한 값과 같습니다.
		mem[i] = max([v for k, v in mem.items() if k < i]) + 1
	print(max(mem.values()))
	```
	- Time complexity: O(nlogn)
	- Dynamic programming + Binary search
- LCS(Longest Common Subsequence)
	- Time complexity: O(nk)
	```python
	mem = [[[0, ""] for _ in range(len(arr2) + 1)] for _ in range(len(arr1) + 1)]

	maxim = 0
	LCS = ""
	for i in range(len(arr1)):
		for j in range(len(arr2)):
			if arr1[i] == arr2[j]:
				mem[i + 1][j + 1][0] = mem[i][j][0] + 1
				mem[i + 1][j + 1][1] = mem[i][j][1] + arr1[i]
			else:
				if mem[i][j + 1][0] > mem[i + 1][j][0]:
					mem[i + 1][j + 1][0] = mem[i][j + 1][0]
					mem[i + 1][j + 1][1] = mem[i][j + 1][1]
				else:
					mem[i + 1][j + 1][0] = mem[i + 1][j][0]
					mem[i + 1][j + 1][1] = mem[i + 1][j][1]
			if mem[i + 1][j + 1][0] > maxim:
				maxim = mem[i + 1][j + 1][0]
				LCS = mem[i + 1][j + 1][1]
	```
- Longest Common Substring
	- Time complexity: O(nk)
	```python
	mem = [[0 for _ in range(len(string1))] for _ in range(len(string2))]
	for i in range(len(string1)):
		for j in range(len(string2)):
			if string1[i] == string2[j]:
				mem[i][j] = mem[i - 1][j - 1] + 1
			else:
				mem[i][j] = 0
	```
- Prefix Sum
	```python
	pref_sum = [0]
	temp = 0
	for i in arr:
		temp += i
		pref_sum.append(temp)
	```

# Greedy Algorithms
- Source: Source: https://www.geeksforgeeks.org/greedy-algorithms/
- Greedy is an algorithmic paradigm that builds up a solution piece by piece, always choosing the next piece that offers the most obvious and immediate benefit. So the problems where choosing locally optimal also leads to global solution are best fit for Greedy.
- Fractional Knapsack Problem

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

# Binary Search
- Time complexity: O(logn)
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

# Parametric Search
- Source: https://velog.io/@lake/%EC%9D%B4%EB%B6%84%ED%83%90%EC%83%89-%ED%8C%8C%EB%9D%BC%EB%A9%94%ED%8A%B8%EB%A6%AD-%EC%84%9C%EC%B9%98Parametric-Search
- Parametric search is a technique for transforming an optimization algorithm (find the best solution) into a decision algorithm (does this optimization problem have a solution with quality better than some given threshold?).

# Implementation
- 풀이를 떠올리는 것은 쉽지만 소스코드로 옮기기 어려운 문제.
## Exhaustive Search

# String
## Knuth-Morris-Pratt Algorithm(KMP Algorithm)
- Source: https://www.educative.io/edpresso/what-is-the-knuth-morris-pratt-algorithm
- Time complexity: O(n + k) (where `n` and `k` are the lengths of string `s` and pattern `p` respectively).
- The KMP algorithm is an algorithm that is used to search for a pattern `p` in a given string `s`.
- LPS(Longest proper Prefix which is also proper Suffix)(= Failure Function)
	- The mapping of an index `i` to the length of the longest prefix of `p[:i + 1]` which is also a suffix.
	- This function is based on the fact that when a mismatch occurs, all the previous characters match correctly.
	- Time complexity: O(k^2)
		```python
		i = 0
		j = 1
		lps = {i:0 for i in range(len(p))}
		while j < len(p):
			if p[:i + 1] == s[j - i:j + 1]:
				lps[j] = i + 1
			if i == j - 1:
				j += 1
				i = 0
			else:
				i += 1
		```
	- Time complexity: O(k)
		```python
		i = 0
		j = 1
		lps = {i:0 for i in range(len(p))}
		while j < len(p):
			print(i, j, p[i], p[j], lps)
			if p[i] == p[j]:
				lps[j] = i + 1
				i += 1
				j += 1
			else:
				if i == 0:
					j += 1
				else:
					i = lps[i - 1]
		```
	- main
	```python
	i = 0
	j = 0
	ans = list()
	while i < len(p) and j < len(s):
		if p[i] == s[j]:
			if i == len(p) - 1:
				ans.append(j - len(p) + 2)
	#             print(j - len(p) + 2)
				j += 1
			else:
				i += 1
				j += 1
		else:
			if i == 0:
				j += 1
			else:
				i = lps[i - 1]
				
	print(len(ans))
	for i in ans:
		print(i, end=" ")
	```

## Simulation

# Hash

# Union-Find