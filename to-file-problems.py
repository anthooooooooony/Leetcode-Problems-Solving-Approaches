import collections
from heapq import *

class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution(object):
    # No.424
    def characterReplacement(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: int
        """
        count = {}
        max_count = start = result = 0
        for end in range(len(s)):
            count[s[end]] = count.get(s[end], 0) + 1
            max_count = max(max_count, count[s[end]])  # The maximum count of the char at index 'end'
            if end - start + 1 - max_count > k:
                count[s[start]] -= 1
                start += 1
            result = max(result, end - start + 1)
        return result

    # No. 567
    def checkInclusion(self, s1, s2):
        """
        :type s1: str
        :type s2: str
        :rtype: bool
        """
        # s1 = "ab" s2 = "eidbaooo"
        counter1 = collections.Counter(s1)
        for i in range(0, len(s2) - len(s1) + 1):
            counter2 = collections.Counter(s2[i: i + len(s1)])
            print counter1
            print counter2
            if counter1 == counter2: return True
        return False

    # def checkInclusion(self, s1, s2):
    #     A = [ord(x) - ord('a') for x in s1]
    #     B = [ord(x) - ord('a') for x in s2]
    #
    #     target = [0] * 26
    #     for x in A:
    #         target[x] += 1
    #
    #     window = [0] * 26
    #     for i, x in enumerate(B):
    #         window[x] += 1
    #         if i >= len(A):
    #             window[B[i - len(A)]] -= 1
    #         if window == target:
    #             return True
    #     return False

    def maxTurbulenceSize(self, A):
        N = len(A)
        ans = 1
        anchor = 0

        for i in xrange(1, N):
            c = cmp(A[i-1], A[i])
            if c == 0: # case 'A[i - 1] == A[i] '
                anchor = i
            elif i == N-1 or c * cmp(A[i], A[i+1]) != -1:
                ans = max(ans, i - anchor + 1)
                anchor = i
        return ans

    def longestOnes(self, A, K):
        i = 0
        for j in xrange(len(A)):
            K -= 1 - A[j]
            if K < 0:
                K += 1 - A[i]
                i += 1
        return j - i + 1

    # 15
    def threeSum(self, nums):
        res = []
        nums.sort()
        length = len(nums)
        for i in xrange(length - 2):  # [8]
            if nums[i] > 0: break  # [7]
            if i > 0 and nums[i] == nums[i - 1]: continue  # [1]

            l, r = i + 1, length - 1  # [2]
            while l < r:
                total = nums[i] + nums[l] + nums[r]

                if total < 0:  # [3]
                    l += 1
                elif total > 0:  # [4]
                    r -= 1
                else:  # [5]
                    res.append([nums[i], nums[l], nums[r]])
                    while l < r and nums[l] == nums[l + 1]:  # [6]
                        l += 1
                    while l < r and nums[r] == nums[r - 1]:  # [6]
                        r -= 1
                    l += 1
                    r -= 1
        return res

    # 16
    def threeSumClosest(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        nums.sort()
        ans = nums[0] + nums[1] + nums[2]
        for i in xrange(len(nums)):
            l, r = i + 1, len(nums) - 1
            while l < r:
                total = nums[i] + nums[l] + nums[r]
                if total == target:
                    return total
                if abs(total - target) < abs(ans - target):
                    ans = total

                if total < target:
                    l += 1
                else:
                    r -= 1
        return ans

    #18
    def fourSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        def findNsum(l, r, target, N, result, results):
            if r-l+1 < N or N < 2 or target < nums[l]*N or target > nums[r]*N:  # early termination
                return
            if N == 2: # two pointers solve sorted 2-sum problem
                while l < r:
                    s = nums[l] + nums[r]
                    if s == target:
                        results.append(result + [nums[l], nums[r]])
                        l += 1
                        while l < r and nums[l] == nums[l-1]:
                            l += 1
                    elif s < target:
                        l += 1
                    else:
                        r -= 1
            else: # recursively reduce N
                for i in range(l, r+1):
                    if i == l or (i > l and nums[i-1] != nums[i]):
                        findNsum(i+1, r, target-nums[i], N-1, result+[nums[i]], results)

        nums.sort()
        results = []
        findNsum(0, len(nums)-1, target, 4, [], results)
        return results

    # 26
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if nums:
            anchor = 0
        else:
            return 0
        for i in xrange(1, len(nums)):
            if nums[i] != nums[anchor]:
                anchor += 1
                nums[anchor] = nums[i]
        return anchor + 1

    # 27
    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        start, end = 0, len(nums) - 1
        while start <= end:
            if nums[start] == val:
                nums[start], nums[end], end = nums[end], nums[start], end - 1
            else:
                start += 1
        return start

    # 28
    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        if needle == "":
            return 0
        for i in range(len(haystack) - len(needle) + 1):
            for j in range(len(needle)):
                if haystack[i + j] != needle[j]:
                    break
                if j == len(needle) - 1:
                    return i
        return -1

    # 61
    def rotateRight(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        if not head:
            return None

        if head.next == None:
            return head

        pointer = head
        length = 1

        while pointer.next:
            pointer = pointer.next
            length += 1

        rotateTimes = k%length

        if k == 0 or rotateTimes == 0:
            return head

        fastPointer = head
        slowPointer = head

        for a in range (rotateTimes):
            fastPointer = fastPointer.next


        while fastPointer.next:
            slowPointer = slowPointer.next
            fastPointer = fastPointer.next
        temp = slowPointer.next # temp = [4, 5]
        slowPointer.next = None
        fastPointer.next = head

        return temp

    # 80
    def removeDuplicates(self, nums):
        """
        :type nums: List[int] [1,2,2,2,2,2,3,3]
        :rtype: int
        """
        if nums:
            anchor = 0
        else:
            return 0
        re = False
        for i in xrange(1, len(nums)):
            if nums[i] != nums[anchor]:
                anchor += 1
                nums[anchor] = nums[i]
                re = False
            elif nums[i] == nums[anchor] and not re:
                anchor += 1
                nums[anchor] = nums[i]
                re = True
        return anchor + 1

    # 88
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: None Do not return anything, modify nums1 in-place instead.
        """
        while m > 0 and n > 0:
            if nums1[m-1] >= nums2[n-1]:
                nums1[m+n-1] = nums1[m-1]
                m -= 1
            else:
                nums1[m+n-1] = nums2[n-1]
                n -= 1
        if n > 0:
            nums1[:n] = nums2[:n]
    # 141
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        marker1 = head
        marker2 = head
        while marker2 and marker2.next:
            marker1 = marker1.next
            marker2 = marker2.next.next
            if marker2==marker1:
                return True
        return False

    # 142
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        try:
            Slow = head.next
            Fast = head.next.next
            while Slow!=Fast:
                Slow = Slow.next
                Fast = Fast.next.next
        except:
            return None
        Slow = head
        while Slow != Fast:
            Slow = Slow.next
            Fast = Fast.next
        return Slow

    # 202
    def isHappy(self, n):
        """
        :type n: int
        :rtype: bool
        """
        seen = set()
        while n not in seen:
            seen.add(n)
            n = sum([int(x) **2 for x in str(n)])
        return n == 1

    def middleNode(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head:
            return None
        length = 0
        fst = snd = head
        while fst.next:
            fst = fst.next
            length += 1
        for _ in range((length + 1)/2):
            snd = snd.next
        return snd

    # 876
    def middleNode(self, head):
        tmp = head
        while tmp and tmp.next:
            head = head.next
            tmp = tmp.next.next
        return head

    # 56
    def merge(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: List[List[int]]
        """
        ans = []
        for i in sorted(intervals, key=lambda x: x[0]):
            if ans and i[0] <= ans[-1][-1]:
                ans[-1][-1] = max(ans[-1][-1], i[-1])
            else:
                ans.append(i)
        return ans
    # 57
    def insert(self, intervals, newInterval):
        """
        :type intervals: List[List[int]]
        :type newInterval: List[int]
        :rtype: List[List[int]]
        """
        left, right = [], []
        s, e = newInterval[0], newInterval[1]
        for i in intervals:
            if i[1] < newInterval[0]:
                left.append(i)
            elif i[0] > newInterval[1]:
                right.append(i)
            else:
                s = min(s, i[0])
                e = max(e, i[1])
        return left + [[s, e]] + right
    # 287
    def findDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        low = 1  # value starts at 1
        high = len(nums) - 1  # duplicate makes the highest value is less than the length

        while low < high:
            mid = low + (high - low) / 2
            count = 0
            for i in nums:
                if i <= mid:
                    count += 1
            if count <= mid:
                low = mid + 1
            else:
                high = mid
        return low
    # 442
    def findDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        res = []
        for x in nums:
            if nums[abs(x)-1] < 0:
                res.append(abs(x))
            else:
                nums[abs(x)-1] *= -1
        return res
    # 206
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        previous = None
        while head:
            current = head
            head = head.next
            current.next = previous
            previous = current
        return previous

    # 92
    def reverseBetween(self, head, m, n):
        """
        :type head: ListNode [1, 2, 3, 4, 5]
        :type m: int 2
        :type n: int 4
        :rtype: ListNode
        https://leetcode.com/problems/reverse-linked-list-ii/discuss/30709/Talk-is-cheap-show-me-the-code-(and-DRAWING)
        """

    def reverseBetween(self, head, m, n):
        """
        :type head: ListNode
        :type m: int
        :type n: int
        :rtype: ListNode
        """
        if not head or m == n: return head
        p = dummy = ListNode(None)
        dummy.next = head
        for i in range(m - 1): p = p.next
        tail = p.next

        for i in range(n - m):
            tmp = p.next  # a)
            p.next = tail.next  # b)
            tail.next = tail.next.next  # c)
            p.next.next = tmp  # d)
        return dummy.next

    # 102
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []
        ans, level = [], [root]
        while level:
            ans.append([node.val for node in level])
            temp = []
            for node in level:
                temp.extend([node.left, node.right])
            level = [leaf for leaf in temp if leaf]
        return ans

    # 107
    # bfs + queue
    def levelOrderBottom(self, root):
        queue, res = collections.deque([(root, 0)]), []
        while queue:
            node, level = queue.popleft()
            if node:
                if len(res) < level + 1:
                    res.insert(0, [])
                res[-(level + 1)].append(node.val)
                queue.append((node.left, level + 1))
                queue.append((node.right, level + 1))
        return res

    # dfs recursively
    def levelOrderBottom1(self, root):
        res = []
        self.dfs(root, 0, res)
        return res

    def dfs(self, root, level, res):
        if root:
            if len(res) < level + 1:
                res.insert(0, [])
            res[-(level + 1)].append(root.val)
            self.dfs(root.left, level + 1, res)
            self.dfs(root.right, level + 1, res)

    # 103
    def zigzagLevelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []
        ans, level, reverse = [], [root], False
        while level:
            if reverse:
                ans.append([node.val for node in level[::-1]])
                reverse = False
            else:
                ans.append([node.val for node in level])
                reverse = True
            temp = []
            for node in level:
                temp.extend([node.left, node.right])
            level = [leaf for leaf in temp if leaf]
        return ans

    def averageOfLevels(self, root):
        """
        :type root: TreeNode
        :rtype: List[float]
        """
        if not root:
            return []
        ans, level = [], [root]
        while level:
            level_ans = [node.val for node in level]
            ans.append(sum(level_ans) / float(len(level_ans)))
            temp = []
            for node in level:
                temp.extend([node.left, node.right])
            level = [leaf for leaf in temp if leaf]
        return ans

    # 637
    def averageOfLevels(self, root):
        """
        :type root: TreeNode
        :rtype: List[float]
        """
        if not root:
            return []
        ans, level = [], [root]
        while level:
            level_ans = [node.val for node in level]
            ans.append(sum(level_ans)/float(len(level_ans)))
            temp = []
            for node in level:
                temp.extend([node.left, node.right])
            level = [leaf for leaf in temp if leaf]
        return ans

    # 111
    def minDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        depth, level = 1, [root]
        while level:
            temp = []
            for node in level:
                if not node.left and not node.right:
                    return depth
                else:
                    temp.extend([node.left, node.right])
            depth += 1
            level = [leaf for leaf in temp if leaf]
        return depth

    # 112
    def hasPathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: bool
        """
        if not root:
            return False
        if not root.left and not root.right and root.val == sum:
            return True
        sum -= root.val
        return self.hasPathSum(root.left, sum) or self.hasPathSum(root.right, sum)

    # 113
    def pathSum(self, root, sum):
        if not root:
            return []
        res = []
        self.dfs(root, sum, [], res)
        return res

    def dfs(self, root, sum, ls, res):
        if not root.left and not root.right and sum == root.val:
            ls.append(root.val)
            res.append(ls)
        if root.left:
            self.dfs(root.left, sum - root.val, ls + [root.val], res)
        if root.right:
            self.dfs(root.right, sum - root.val, ls + [root.val], res)

    def pathSum2(self, root, sum):
        if not root:
            return []
        if not root.left and not root.right and sum == root.val:
            return [[root.val]]
        tmp = self.pathSum(root.left, sum - root.val) + self.pathSum(root.right, sum - root.val)
        return [[root.val] + i for i in tmp]

    # BFS + queue
    def pathSum3(self, root, sum):
        if not root:
            return []
        res = []
        queue = [(root, root.val, [root.val])]
        while queue:
            curr, val, ls = queue.pop(0)
            if not curr.left and not curr.right and val == sum:
                res.append(ls)
            if curr.left:
                queue.append((curr.left, val + curr.left.val, ls + [curr.left.val]))
            if curr.right:
                queue.append((curr.right, val + curr.right.val, ls + [curr.right.val]))
        return res

    # DFS + stack I
    def pathSum4(self, root, sum):
        if not root:
            return []
        res = []
        stack = [(root, sum - root.val, [root.val])]
        while stack:
            curr, val, ls = stack.pop()
            if not curr.left and not curr.right and val == 0:
                res.append(ls)
            if curr.right:
                stack.append((curr.right, val - curr.right.val, ls + [curr.right.val]))
            if curr.left:
                stack.append((curr.left, val - curr.left.val, ls + [curr.left.val]))
        return res

        # DFS + stack II

    def pathSum5(self, root, s):
        if not root:
            return []
        res = []
        stack = [(root, [root.val])]
        while stack:
            curr, ls = stack.pop()
            if not curr.left and not curr.right and sum(ls) == s:
                res.append(ls)
            if curr.right:
                stack.append((curr.right, ls + [curr.right.val]))
            if curr.left:
                stack.append((curr.left, ls + [curr.left.val]))
        return res

    def medianSlidingWindow(self, nums, k):
        small_heap, large_heap = [], []




class MedianFinder(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.heaps = [], []

    def addNum(self, num):
        """
        :type num: int
        :rtype: None
        """
        small, large = self.heaps
        heappush(small, -heappushpop(large, num))
        if len(large) < len(small):
            heappush(large, -heappop(small))

    def findMedian(self):
        """
        :rtype: float
        """
        small, large = self.heaps
        if len(large) > len(small):
            return float(large[0])
        return (large[0] - small[0]) / 2.0




#sol = Solution()
#print sol.averageOfLevels(root)

