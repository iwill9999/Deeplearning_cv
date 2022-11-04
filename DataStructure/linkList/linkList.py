from collections import deque

# Create a linkedlist
linkedlist = deque()

# Add element
# Time Complexity :O(1)
linkedlist.append(1)
linkedlist.append(2)
linkedlist.append(3)
# [1, 2, 3]
print(linkedlist)

# Insert element
# Time Complexity:O(N)
linkedlist.insert(2,99)
# [1, 2, 99, 3]
print(linkedlist)

# Access element
# Time Complexity: O(n)
element = linkedlist[2]
# 99
print(element)

# Search element
index = linkedlist.index(99)
# 2
print(index)

# Update element
# Time Complexity :O(N)
linkedlist[2] = 88
# [1, 2, 88, 3]
print(linkedlist)

# remove element O(n)
linkedlist.remove(88)   # 删除指定元素
del linkedlist[2]       # 删除指定索引元素


# Length O(1)
length = len(linkedlist)


