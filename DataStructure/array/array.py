# create an array
a = []

# add element  O(1)
a.append(1)
a.append(2)
a.append(3)
print(a)

# insert element  O(n)
a.insert(2, 99)

print(a)

# Update element O(1)
a[2] = 88

# pop会返回删除的那个元素
# Remove element
a.remove(88)  # O(n) 删指定元素
a.pop(1)      # O(n) 根据索引删元素 因为删除之后每个元素都要移动
a.pop()       # O(1) 删除最后一个元素

# get array size
size = len(a)

# Iterate array 三种遍历方式
# Time complexity
for i in a:
    print(i)

# enumerate会返回两个值 一个索引，一个元素值
for index, element in enumerate(a):
    print("Index at", index, "is: ", element)

for i in range(0, len(a)):
    print("i: ", i, "element", a[i])

# Find an element O(n)
index = a.index(2)

# Sort an array O(nlogn)
a = [3, 1, 2]
a.sort()  # [1, 2, 3]
# 从大到小
a.sort(reverse=True)  # [3, 2, 1]


