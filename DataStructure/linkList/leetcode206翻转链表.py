'''
给你单链表的头节点 head ，请你反转链表，并返回反转后的链表。
输入：head = [1,2,3,4,5]
输出：[5,4,3,2,1]
'''
from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def reverList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(0)
        dummy.next = head
        while head is not None and head.next is not None:
            h_next = head.next
            head.next = h_next.next
            h_next .next = dummy.next
            dummy.next = h_next
        return dummy.next