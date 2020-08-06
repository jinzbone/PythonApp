from base.TreeNode import TreeNode
from base.ListNode import ListNode
import sys

class top100Liked():
    def __init__(self, ):
        self.cur_sum = 0
        self.diameter = 0
        self.stack = []
        self.minStack = []
    
    def mergeTrees(self, t1, t2):
        ''' 617. 合并二叉树
        '''
        def recursion(t1, t2):
            if not t1 and not t2:
                return None
            if not t1:
                return t2
            if not t2:
                return t1
            if t1 and t2:
                node = TreeNode(t1.val + t2.val)
                node.left = recursion(t1.left, t2.left)
                node.right = recursion(t1.right, t2.right)
                return node
        root = recursion(t1, t2)
        return root

    def hammingDistance(self, x, y):
        """ 461. 汉明距离
        :type x: int
        :type y: int
        :rtype: int
        """
        return bin(x^y).count('1')
        pass

    def invertTree(self, root):
        """ 226. 翻转二叉树
        :type root: TreeNode
        :rtype: TreeNode
        """
        if not root:
            return
        root.left, root.right = root.right, root.left
        self.invertTree(root.left)
        self.invertTree(root.right)
        return root

    def maxDepth(self, root):
        """ 104. 二叉树的最大深度
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        return max(self.maxDepth(root.left) +1, self.maxDepth(root.right) + 1)

    def reverseList(self, head):
        """ 206. 反转链表
        :type head: ListNode
        :rtype: ListNode
        """
        pre = None
        while head:
            next = head.next
            head.next = pre
            if not next:
                return head
            pre = head
            head = next
        return head

    def singleNumber(self, nums):
        """ 136. 只出现一次的数字
        :type nums: List[int]
        :rtype: int
        """
        result = nums[0]
        for num in nums[1:]:
            result = result ^ num
        return result

    def majorityElement(self, nums):
        """ 169. 求众数
        :type nums: List[int]
        :rtype: int
        """
        nums.sort()
        return nums[len(nums)//2]

    def mergeTwoLists(self, l1, l2):
        """ 21. 合并两个有序链表
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        l = ListNode(-1)
        head = l
        while l1 and l2:
            if l1.val<=l2.val:
                l.next = l1
                l1 = l1.next
            else:
                l.next = l2
                l2 = l2.next
            l = l.next
        if not l1:
            l.next = l2
        if not l2:
            l.next = l1
        return head.next

    def moveZeroes(self, nums):
        """ 283. 移动零
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        count = 0
        j = 0
        for i in range(len(nums)):
            if nums[i]==0:
                count += 1
            else:
                nums[j] = nums[i]
                j += 1
        nums[j:] = [0 for i in range(len(nums)-j)]
        print(nums)

    def convertBST(self, root):
        """ 538. 把二叉搜索树转换为累加树
        :type root: TreeNode
        :rtype: TreeNode
        """
        self.sum = 0
        def dfs(root):
            if not root:
                return
            dfs(root.right)
            self.sum += root.val
            root.val = self.sum
            dfs(root.left)
            return root
        return dfs(root)

    def inOrder(self, root):
        if not root:
            return
        self.inOrder(root.left)
        print(root.val)
        self.inOrder(root.right)

    def findDisappearedNumbers(self, nums):
        """ 448. 找到所有数组中消失的数字
        :type nums: List[int]
        :rtype: List[int]
        """
        result = []
        for num in nums:
            idx = len(nums) - abs(num)
            if nums[idx]>0:
                nums[idx] = -nums[idx]
        for i in range(len(nums)):
            if nums[i]>0:
                result.append(len(nums)-i)
        return result

    def pathSum1(self, root, sum):
        """ 112. 路径总和
        根节点->叶子节点，是否存在
        :type root: TreeNode
        :type sum: int
        :rtype: bool
        """
        if not root: # 假的叶子节点
            return False
        if sum==root.val and not root.left and not root.right:# 叶子节点
            return True
        sum -= root.val
        return self.hasPathSum(root.left, sum) or self.hasPathSum(root.right, sum)

    def pathSum2(self, root, sum):
        """ 113. 路径总和 II
        根节点->叶子节点，path
        :type root: TreeNode
        :type sum: int
        :rtype: List[List[int]]
        """
        result = []
        path = []
        def dfs(root, sum, path, result):
            if not root:
                return
            if sum==root.val and not root.left and not root.right:
                result.append(path+[root.val])
            sum -= root.val
            dfs(root.left, sum, path + [root.val], result)
            dfs(root.right, sum, path + [root.val], result)
        dfs(root, sum, path, result)
        return result

    def pathSum3(self, root, sum):
        """
        任意节点->任意节点，个数（从上到下）
        dfs
        :type root: TreeNode
        :type sum: int
        :rtype: int
        """
        if not root:
            return 0
        def dfs(root, sum):
            result = 0
            if not root: # 到了假叶子节点
                return 0
            if sum==root.val: # 符合条件，还要继续往下遍历
                result += 1
            result += dfs(root.left, sum-root.val)
            result += dfs(root.right, sum-root.val)
            return result
        return dfs(root, sum) + self.pathSum3(root.left, sum) + self.pathSum3(root.right, sum)

    def maxProfit(self, prices):
        """ 121. 买卖股票的最佳时机
        :type prices: List[int]
        :rtype: int
        """
        if len(prices)<2:
            return 0
        buy = prices[0]
        maxP = 0
        for price in prices:
            profit = price - buy
            if profit<0:
                buy = price
            else:
                maxP = max(maxP, price-buy)
        return maxP

    def push(self, x):
        """ 155. 最小栈
        :type x: int
        :rtype: None
        """
        if not self.stack:
            self.stack.append(x)
            self.minStack.append(x)
        else:
            top = self.minStack[-1]
            if x<top:
                self.minStack.append(x)
                self.stack.append(x)
            else:
                self.minStack.append(top)
                self.stack.append(x)

    def pop(self):
        """
        :rtype: None
        """
        self.stack.pop()
        self.minStack.pop()

    def top(self):
        """
        :rtype: int
        """
        if not self.stack:
            return
        else:
            return self.stack[-1]

    def getMin(self):
        """
        :rtype: int
        """
        if not self.minStack:
            return
        else:
            return self.minStack[-1]
        
    def getIntersectionNode(self, headA, headB):
        """ 160. 相交链表
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        if not headA or not headB:
            return None
        curA, curB = headA, headB
        while curA!=curB:
            curA = curA.next if curA else headB
            curB = curB.next if curB else headA
        return curA

    def isSymmetric(self, root):
        """ 101. 对称二叉树
        :type root: TreeNode
        :rtype: bool
        """
        def check(r1, r2):
            if not r1 and not r2:
                return True
            elif not r1 or not r2:
                return False
            elif r1.val != r2.val:
                return False
            return check(r1.left, r2.right) and check(r1.right, r2.left)
        return check(root, root)

    def mirror(self, root):
        if not root:
            return None
        root.left, root.right = root.right, root.left
        self.mirror(root.left)
        self.mirror(root.right)
        return root

    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head:
            return None
        pre = None
        cur = head
        while True:
            next = cur.next
            cur.next = pre
            pre = cur
            cur = next
            if not next:
                break
        return pre

    def findPath1(root, target):
        if not root:
            return False
        if not root.left and not root.right and target==root.val:
            return True
        return self.findPath1(root.left, target-root.val) or self.findPath1(root.right, target-root.val)
    
    def findPath2(root, target):
        def dfs(root, target, result, path):
            if not root:
                return
            if not root.left and not root.right and target==root.val:
                result.append(path+[root.val])
                return
            dfs(root.left, target-root.val, result, path+[root.val])
            dfs(root.right, target-root.val, result, path+[root.val])
        result = []
        dfs(root, target, result, [])
        return result
        pass

    def findPath3(root, target):
        def dfs(root, target):
            cnt = 0
            if not root:
                return 0
            if target==root.val:
                cnt += 1
            cnt = cnt + dfs(root.left, target-root.val) + dfs(root.right, target-root.val)
        return dfs(root, target) + self.findPath3(root.left, target) + self.findPath3(root.right, target)

        
        pass

    def diameterOfBinaryTree(self, root):
        """ 543. 二叉树的直径
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        def getDepth(root):
            if not root:
                return 0
            return 1 + max(getDepth(root.left), getDepth(root.right))
        leftH = getDepth(root.left)
        rightH = getDepth(root.right)
        return max(leftH+rightH-1, self.diameterOfBinaryTree(root.left), self.diameterOfBinaryTree(root.right))

lc = top100Liked()
nums = [7,1,5,3,6,4]

t1 = TreeNode(1)
t2 = TreeNode(2)
t3 = TreeNode(3)
t4 = TreeNode(4)
t5 = TreeNode(5)
t1.left = t2
t1.right = t3
t2.left = t4
t2.right = t5

l1 = ListNode(1)
l2 = ListNode(2)
l3 = ListNode(3)
l4 = ListNode(4)
l5 = ListNode(5)
l1.next = l2
l2.next = l3
l3.next = l4
l4.next = l5
