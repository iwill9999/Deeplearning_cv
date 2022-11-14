from typing import Optional

'''
递归的三个要素：
1.确定递归函数的参数和返回值：
    确定哪些参数是递归的过程中需要处理的，在递归函数的里加上这个参数
    明确每次递归的返回值是什么进而确定递归函数的返回类型

2.确定终止条件：
    操作系统用一个栈来保存每一层递归的信息，如果递归没有终止，造成栈溢出

3.确定单层递归的逻辑:
    确定每一层递归需要处理的信息，在这里就会重复调用自己来实现递归的过程


实例
1.确定递归函数的参数和返回值：
    因为要打印前序遍历节点的数值把需要的参数放入list里，不需要返回值
    def traversal(root):
    
2.确定终止条件：
    当前遍历的节点为空，本层的递归就结束了
    if not root:
        return

3.确定单层递归的逻辑：
    前序遍历是中左右的循序，在单层递归的逻辑中，要先取中间节点的数值
    res.append(root.val) # 中
    traversal(root.left) # 左
    traversal(root.right)# 右


'''


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# 先序遍历二叉树
# 递归法
def preorderTraverseIterate(root:Optional[TreeNode]) -> list[int]:
    res = list()
    def preorder(root):
        if not root:
            return
        res.append(root.val)
        preorder(root.left)
        preorder(root.right)
    preorder(root)
    return res

# 中序遍历二叉树
# 递归法
def inorderTraverse(root:Optional[TreeNode]) -> list[int]:
    res = list()
    def inorder(root):
        if not root:
            return
        inorder(root.left)
        res.append(root.val)
        inorder(root.right)
    inorder(root)
    return res


# 先序遍历二叉树
# 迭代法
def preorderTraverseRecursive(root:Optional[TreeNode]) -> list[int]:
    res = []
    if not root:
        return res
    stack = []
    stack.append(root)
    while stack:
        root = stack.pop()
        res.append(root.val)
        if root.right:
            stack.append(root.right)
        if root.left:
            stack.append(root.left)
    return res

# 中序遍历二叉树
# 迭代法
def inorderTraverseRecursive(root:Optional[TreeNode]) -> list[int]:
    res = []
    if not root:
        return res
    stack = []
    while stack:
        if root or stack:
            if root:
                stack.append(root)
                root = root.left
            else:
                root = stack.pop()
                res.append(root.val)
                root = root.right
    return res

# 后序遍历二叉树
# 迭代法
'''
1.栈顶弹出
2.如果有左，压入左
3.如果有右，压入右
上述输出时 头 左 右的顺序
逆序就是 左 右 头的后序遍历
'''
def postorderTraberseRecursive(root:Optional[TreeNode]) -> list[int]:
    res = []
    if not root:
        return res
    stack = []
    stack.append(root)
    while(stack):
        root = stack.pop()
        res.append(root.val)
        if root.left:
            stack.append(root.left)
        if root.right:
            stack.append(root.right)
    return list(reversed(res))











































