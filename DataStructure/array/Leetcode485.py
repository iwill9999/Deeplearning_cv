"""
给定一个二进制数组 nums ， 计算其中最大连续 1 的个数。
示例 1：
输入：nums = [1,1,0,1,1,1]
输出：3
解释：开头的两位和最后的三位都是连续 1 ，所以最大连续 1 的个数是 3.
示例 2:
输入：nums = [1,0,1,1,0,1]
输出：2

提示：
1 <= nums.length <= 105
nums[i]不是0就是1
"""

class Solution:
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        count = 0
        result = 0
        for i, j in enumerate(nums):
            if nums[i] == 1:
                count = count + 1
                if count > result:
                    result = count
            else:
                count = 0
        return result

