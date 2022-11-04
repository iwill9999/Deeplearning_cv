from collections import deque

# Crate a queue
queue = deque()

# Add element
# Time complexity
queue.append(1)


#get the head of the queue
# Time complexity  O(1)
temp1 = queue[0]


# Remove the head of the queue O(1)
# Time complexity
temp2 = queue.popleft()

# Queue is empty?
# Time complexity O(1)
len(queue) == 0

# Time complexity O(N)
while len(queue) != 0:
    temp = queue.popleft()
    