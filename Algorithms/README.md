# Data Structure

## Trie Data Structure
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