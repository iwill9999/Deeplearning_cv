"""不使用任何内建的哈希表库设计一个哈希集合（HashSet）。

实现 MyHashSet 类：

void add(key) 向哈希集合中插入值 key 。
bool contains(key) 返回哈希集合中是否存在这个值 key 。
void remove(key) 将给定值 key 从哈希集合中删除。如果哈希集合中没有这个值，什么也不做。"""

class MyHashSet:
    def __init__(self):
        self.base = 1009
        self.hashset = [[] for _ in range(self.base)]

    def hash(self, key):
        return key % self.base

    def add(self, key):
        hashkey = hash(key)
        if key in self.hashset[hashkey]:
            return
        self.hashset[hashkey].append(key)

    def remove(self, key):
        hashkey = hash(key)
        if key not in self.hashset[hashkey]:
            return
        self.hashset[hashkey].remove(key)

    def contains(self, key):
        hashkey = hash(key)
        return key in self.hashset[hashkey]