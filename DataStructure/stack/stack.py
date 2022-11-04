# Create a stack
stack = []

# Add element
stack.append(1)
stack.append(2)
stack.append(3)

# Get the top of stack
# Time complexity O(1)
top = stack[-1]

# Remove the top of stack
# Time complexity O(1)
temp = stack.pop()

# Stack is empty?
# Time complexity O(1)
bool = len(stack) == 0

# Iterate Stack
# Time complexity O(n)
while len(stack) > 0 :
    temp = stack.pop()