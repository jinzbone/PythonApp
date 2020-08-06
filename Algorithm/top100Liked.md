[[TOC]]


# 617.合并二叉树 ☆
<font color=red>二叉树的题目，一般都要用到递归。</font>  
对于t1和t2中的每个节点，新节点t的值，是根据t1和t2生成的。t的左右子树节点，是t1和t2的左右子树生成的。

``` python
    # 64 ms, 在所有 python 提交中击败了85.60%的用户
    def mergeTrees(self, t1, t2):
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
```

# 461.汉明距离 ☆
<font color=red>python有个内置函数 ```bin()``` 可以获得十进制整数的二进制表示，（返回的是字符串形式）。</font>    
所以直接两个int型整数进行异或得到的int值进行bin()操作，得到的就是异或结果的二进制表示，然后取出里面含有的1的个数来。 

```python
    # 12 ms, 在所有 python 提交中击败了97.62%的用户
    def hammingDistance(self, x, y):
        return bin(x^y).count('1')
```

# 226.翻转二叉树 ☆
<font color=red> 二叉树的题目，一般都要用到递归。</font>  
这里，主要是对每个节点进行操作，先把根节点的左右节点交换，再对左右节点做同样的操作。  
最后返回根节点。

```python
    # 32 ms, 在所有 python 提交中击败了14.24%的用户
    def invertTree(self, root):
        if not root:
            return
        root.left, root.right = root.right, root.left
        self.invertTree(root.left)
        self.invertTree(root.right)
        return root
```

# 104.二叉树的最大深度
<font color=red>二叉树的深度就是左右子树的最大深度</font>

```python
    # 32 ms, 在所有 python 提交中击败了79.99%的用户
    def maxDepth(self, root):
        if not root:
            return 0
        return max(self.maxDepth(root.left) +1, self.maxDepth(root.right) + 1)
```

# 206.反转链表
迭代，注意好 pre, head, next 三个的改变就好了

```python
    # 24 ms, 在所有 python 提交中击败了86.82%的用户
    def reverseList(self, head):
        pre = None
        while head:
            next = head.next
            head.next = pre
            if not next:
                return head
            pre = head
            head = next
        return head
```

# 136.只出现一次的数字
<font color=red>用异或，相异为1，相同为0</font>

```python
    # 116 ms, 在所有 python 提交中击败了30.99%的用户
    def singleNumber(self, nums):
        result = nums[0]
        for num in nums[1:]:
            result = result ^ num
        return result
```

# 169.求众数
排序，取中间的数

```python
    # 180 ms, 在所有 python 提交中击败了84.19%的用户
    def majorityElement(self, nums):
        nums.sort()
        return nums[len(nums)//2]
```

# 21.合并两个有序链表 
迭代

```python
    def mergeTwoLists(self, l1, l2):
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
```

# 283.移动零
记录0的个数就行了

```python
    # 36 ms, 在所有 python 提交中击败了93.27%的用户
    def moveZeroes(self, nums):
        count = 0
        j = 0
        for i in range(len(nums)):
            if nums[i]==0:
                count += 1
            else:
                nums[j] = nums[i]
                j += 1
        nums[j:] = [0 for i in range(len(nums)-j)]
```

# 538.把二叉搜索树转换为累加树 ☆ ☆ 
<font color=red>先遍历右节点，再存储根节点，最后遍历左节点</font>

```python
    def convertBST(self, root):
        self.sum = 0
        def dfs(root):
            if not root:
                return
            dfs(root.right)
            self.sum += root.val
            root.val = self.sum
            dfs(root.left)
            return root
        t = dfs(root)
        return t
```


# 448.找到所有数组中消失的数字
限制不能用额外的空间，且时间复杂度为O(n)  
由于数字是在[1, len(nums)]之间的，每个数字和 len(nums)-num 有个对应关系（num<->idx)  
先过一遍把对应idx的数字标负，再把没对应的为正的取出来

```python
    # 380 ms, 在所有 python 提交中击败了89.37%的用户
    def findDisappearedNumbers(self, nums):
        result = []
        for num in nums:
            idx = len(nums) - abs(num)
            if nums[idx]>0:
                nums[idx] = -nums[idx]
        for i in range(len(nums)):
            if nums[i]>0:
                result.append(len(nums)-i)
        return result
```

# 112.路径总和
根节点->叶子节点  
递归的出口。
到了假叶子节点了，肯定return False  
到了真的叶子节点，且和符合条件，则return True  
对每个节点，这样遍历它的左右子树进行判断，等符合条件了就退出了。

```python
    def pathSum1(self, root, sum):
        if not root: # 假的叶子节点
            return False
        if sum==root.val and not root.left and not root.right:# 叶子节点
            return True
        sum -= root.val
        return self.hasPathSum(root.left, sum) or self.hasPathSum(root.right, sum)
```

# 113.路径总和 II
<font color=red>dfs</font>
和 I 是一样的， 关键都在于找到递归条件的出口：
+ 到了假的叶子节点，是不符合要求的
+ 到了真的叶子节点，且和符合要求，才是符合要求的

```python
    # 36 ms, 在所有 python 提交中击败了87.44%的用户
    def pathSum2(self, root, sum):
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
```

# 437.路径总和 III
<font color=red>两层递归</font>

```python
   def pathSum3(self, root, sum):
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
```

# 666.路径和 IV

# 121.买卖股票的最佳时机
先假定第一个价格为买入的，当卖出-买入<0时，最大利润是0，并假设买入为当前的卖出。  
当卖出-买入>0时，取最大利润。

```python
    #40 ms, 在所有 python 提交中击败了99.92%的用户
    def maxProfit(self, prices):
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
```

# 155.最小栈
<font color=red>维持两个栈</font>，一个是正常栈，另一个辅助栈的栈顶一直表示当前元素的最小值。

```python
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []
        self.minStack = []
        
    def push(self, x):
        """
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
```

# 160.相交链表
<u>1 2 3 6 7 4 5</u> 6 7  
<u>4 5 6 7 1 2 3</u> 6 7  
两个指针一起走，当p1走到终点时，就让他指向第二个链表的头；当p2走到终点时，就让p2指向第一个链表的头。  
**边界控制：如果没有公共节点的话，最后两个指针都会指向None，相同的None节点会退出～**

```python
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
```

# 101.对称二叉树

```python
    def isSymmetric(self, root):
        """
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
```




























































