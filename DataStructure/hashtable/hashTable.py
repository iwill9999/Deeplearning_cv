# Create Hash by Array
hashTable = [''] * 4
# Create HashTable by Dictionary
mapping = {}

# Add element
# Time Cpmplexity O(1)
hashTable[1] = 'hanmeimei'
hashTable[2] = 'lihua'
mapping[1] = 'hanmeimei'
mapping[2] = 'lihua'

# Update element
# Time Complexity O(1)
hashTable[1] = ''
# 用字典创建的两中删除方法
mapping.pop(1)
# del mapping[1]

# Get Value
# Time Complexity O(1)
hashTable[3]

# check Key
# Time Complexity O(1)
3 in mapping     # return Ture 检查key是否存在

# length
# Time Complexity O(1)
boolean = len(mapping) == 0
