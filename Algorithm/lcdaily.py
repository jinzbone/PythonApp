from base.ListNode import ListNode
from base.TreeNode import TreeNode

class Solution():
    def __init__(self):
        self.diameter = 0
        self.curSum=0
        pass
    def coinChange(self, coins, amount):
        ''' 322. 零钱兑换

        '''
        if amount<0 or len(coins)<1:
            return -1
        if amount==0:
            return 0
        dp = [amount+1 for i in range(amount+1)]
        for i in range(1, amount+1):
            if i in coins:
                dp[i] = 1
            else:
                for coin in coins:
                    if i-coin>0:
                        dp[i] = min(dp[i-coin]+1, dp[i])
        return -1 if dp[-1]==amount+1 else dp[-1]

    def hasCycle(self, head):
        ''' 141. 环形链表

        '''
        if not head or not head.next:
            return False
        fast = head.next
        slow = head
        while True:
            if fast==slow:
                return True
            elif fast.next and fast.next.next:
                fast = fast.next.next
                slow = slow.next
            else:
                return False
        pass

    def maxProfit(self, prices):
        ''' 121. 买卖股票的最佳时机

        '''
        if not prices:
            return 0
        min_ = prices[0]
        max_ = prices[0]
        profit = 0
        for price in prices[1:]:
            if price<min_:
                min_ = price
                max_ = price
            elif price>max_:
                profit = max(profit, price-min_)
                max_ = price
        return profit

    def rob(self, nums):
        ''' 198. 打家劫舍

        '''
        n = len(nums)
        if n==0:
            return 0
        if n==1:
            return nums[0]
        dp = [0 for i in range(n)]
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])
        for i in range(2, n):
            dp[i] = max(dp[i-1], dp[i-2]+nums[i])
        return max(dp)

    def diameterOfBinaryTree(self, root):
        '''543. 二叉树的直径

        '''
        if not root:
            return 0
        leftH = self.getDepth(root.left)
        rightH = self.getDepth(root.right)
        return max(rightH+leftH, self.diameterOfBinaryTree(root.left), self.diameterOfBinaryTree(root.right))

    def getDepth(self, root):
        if not root:
            return 0
        return 1+max(self.getDepth(root.left), self.getDepth(root.right))
        pass

    def canThreePartsEqualSum(self, A):
        '''1013. 将数组分成和相等的三个部分
        
        '''
        n = len(A)
        if n<3:
            return False
        total = sum(A)
        if total%3!=0:
            return False
        target = total/3
        cur_sum = 0
        count = 0
        for i in range(n):
            cur_sum += A[i]
            if cur_sum==target:
                cur_sum = 0
                count += 1                
        return True if count>=3 else False

    def gcdOfStrings(self,str1, str2):
        '''1071. 字符串的最大公因子

        gcd
        '''
        m = len(str1)
        n = len(str2)
        if m==0 or n==0:
            return ""
        if str1+str2!=str2+str1:
            return ""
        gcd_number = self.gcd(m, n)
        print(gcd_number)
        return str1[0: gcd_number]

    def gcd(self,a,b):
        '''求a,b的最大公约数
        '''
        print(a, b)
        return a if b==0 else self.gcd(b, a%b)

    def majorityElement(self, nums):
        '''169. 多数元素

        vote
        '''
        if not nums:
            return
        number = nums[0]
        count = 1
        for i in range(1, len(nums)):
            if nums[i]==number:
                count += 1
            else:
                count -= 1
                if count == 0:
                    number = nums[i]
                    count = 1
        return number
        pass

    def lengthOfLIS(self, nums):
        n = len(nums)
        if n<2:
            return n
        dp = [1 for i in range(n)]
        for i in range(n):
            for j in range(i, -1, -1):
                if nums[j]<nums[i]:
                    dp[i] = max(dp[i], dp[j]+1)
        return max(dp)

    def lengthOfLIS2(self, nums):
        '''利用二分查找 进阶版，和一个ends数组，优化时间。
        '''
        n = len(nums)
        if n<2:
            return n
        dp = [0 for i in range(n)]
        ends = [0 for i in range(n+1)]
        right = 1
        ends[1] = nums[0]
        for i in range(1, n):
            l, r = 1, right
            while l<=r:
                mid = (l+r)//2
                if ends[mid]<nums[i]:
                    l = mid + 1
                else:
                    r = mid - 1
            ends[l] = nums[i]
            dp[i] = l
            right = max(l, right)
        return max(dp)

    def maxAreaOfIsland(self, grid):
        ''' 695. 岛屿的最大面积
        
        dfs: 对每个grid点进行dfs遍历，每次dfs中，将已经访问过的点置为0
        '''
        result = 0
        if not grid:
            return 0
        def dfs(grid, grid_i, grid_j):
            if grid_i<0 or grid_i>=len(grid) or grid_j<0 or grid_j>=len(grid[0]) or grid[grid_i][grid_j]==0:
                return 0
            grid[grid_i][grid_j] = 0
            ans = 1
            ans += dfs(grid, grid_i, grid_j-1)
            ans += dfs(grid, grid_i, grid_j+1)
            ans += dfs(grid, grid_i-1, grid_j)
            ans += dfs(grid, grid_i+1, grid_j)
            return ans
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j]==1:
                    result = max(result, dfs(grid, i, j))
        return result

    def compressString(self, S):
        '''面试题 01.06. 字符串压缩

        '''
        n = len(S)
        if n==0:
            return ""
        result = ""
        cur = S[0]
        count = 0
        for i in range(n):
            c = S[i]
            if c==cur:
                count += 1
            else:
                result = result + str(cur) + str(count)
                cur = c
                count = 1
        result = result + str(cur) + str(count)
        return result if len(result)<n else S

    def countCharacters(self, words, chars):
        '''1160. 拼写单词
        
        '''
        chars_map = {}
        result = 0
        for ch in chars:
            if ch not in chars_map:
                chars_map[ch] = 1
            else:
                chars_map[ch] += 1
        # print(chars_map)
        for word in words:
            word_map = {}
            flag = True
            for c in word:
                if c not in word_map:
                    word_map[c] = 1
                else:
                    word_map[c] += 1
            # print(word_map)
            for k,v in word_map.items():
                if str(k) not in chars or int(chars_map[k])<v:
                    flag = False
                    break
            if flag:
                result += len(word)
        return result
            
    def isRectangleOverlap(self, rec1, rec2):
        '''836. 矩形重叠

        '''
        if rec1[0]>=rec2[2] or rec1[2]<=rec2[0] or rec1[3]<=rec2[1] or rec1[1]>=rec2[3]:
            return False
        return True

    def longestPalindrome(self, s):
        '''409. 最长回文串
        
        奇 偶
        '''
        count = 0
        s_map = {}
        if len(s)==0:
            return 0
        for c in s:
            if c not in s_map:
                s_map[c] = 1
            else:
                s_map[c] += 1
        flag = False
        for k, v in s_map.items():
            if v%2==0:
                count += int(v)
            else:
                count += (v-1)
                flag = True
        return count+1 if flag else count

    def getLeastNumbers(self, arr, k):
        arr.sort()
        return arr[:k]
    def getLeastNumbers2(self, arr, k):
        n = len(arr)
        if k==0:
            return []
        result = []
        for i in range(n-1):
            for j in range(n-1-i):
                if arr[j]<arr[j+1]:
                    arr[j], arr[j+1] = arr[j+1], arr[j]
            result.append(arr[n-1-i])
            print(result)
            if len(result) == k:
                return result
        return result
        pass

    def middleNode(self, head):
        '''876. 链表的中间结点

        '''
        count = 0
        cur = head
        while cur:
            count += 1
            cur = cur.next
        index = count//2
        cur = head
        for i in range(index):
            cur = cur.next
        return cur            

    def hasGroupSizeX(self, deck):
        n = len(deck)
        if n<2:
            return False
        dic = {}
        count = 0
        for e in deck:
            if e not in dic:
                dic[e] = 1
            else:
                dic[e] += 1
        count = dic[deck[0]]
        for k, v in dic.items():
            if v<2:
                return False
            count = self.gcd(count, v)
            if count == 1:
                return False
        return True
        
    def gcd(self,a,b):
        return a if b==0 else self.gcd(b, a%b)

    def lastRemaining(self, n, m):
        '''面试题62. 圆圈中最后剩下的数字

        约瑟夫环
        '''
        if n<1 or m<1:
            return -1
        res = 0
        for i in range(1, n+1):
            res = (res + m) % i
        return res

    def sortArray(self, nums):
        '''912. 排序数组

        快排
        '''
        self.quickSort(nums,0,len(nums)-1)
        return nums

    def quickSort(self, nums, low, high):
        if low>high:
            return
        i, j = low, high
        key = nums[low]
        while i<j:
            while i<j and nums[j]>=key:
                j -= 1
            while i<j and nums[i]<=key:
                i += 1
            if i<j:
                nums[i], nums[j] = nums[j], nums[i]
                continue
            if i==j:
                nums[j], nums[low] = nums[low], nums[j]
                break
        self.quickSort(nums, low, i-1)
        self.quickSort(nums, i+1, high)

    def myAtoi(self, s):
        '''8. 字符串转换整数 (atoi)

        '''
        result = []
        flag = False
        for c in s:
            if flag == False and c == " ":
                continue
            elif flag==False and (c=="+" or c=="-"):
                flag = True
                result.append(c)
            elif c=="0" or c=="1" or c=="2" or c=="3" or c=="4" or c=="5" or c=="6" or c=="7" or c=="8" or c=="9":
                flag = True
                result.append(c)
            elif flag==False and not (c=="1" or c=="2" or c=="3" or c=="4" or c=="5" or c=="6" or c=="7" or c=="8" or c=="9"):
                return 0
                break
            else:
                break
        
        number = 0
        if len(result)<1:
            return 0
        if len(result)==1 and (result[0]=="+" or result[0]=="-"):
            return 0
        if result[0]=="+" or result[0]=="-":
            for c in result[1:]:
                number = int(c)+number*10
            pass
        else:
            for c in result:
                number = int(c)+number*10
        if number>=pow(2, 31):
            number = pow(2, 31)
            return number-1 if result[0]!="-" else -number
        return number if result[0]!="-" else -number

    def trap(self, height):
        '''42. 接雨水

        按列求
        遍历每一列，求它左边的最大数和右边的最大数，二者中取较小值 miner
        如果miner>height[i]，那么此列的水量就是miner-height[i]
        如果miner>=height[i]，那么此列的水量就是0
        '''
        n = len(height)
        if n<2:
            return 0
        result = 0
        left_h = max(height[:1])
        right_h = max(height[2:])
        miner = min(left_h, right_h)
        if miner>height[1]:
            result += (miner-height[1])
        for i in range(2,n-1):
            if height[i-1]>left_h:
                left_h = height[i-1]
            right_h = max(height[i+1:])
            miner = min(left_h, right_h)
            if miner>height[i]:
                result += (miner-height[i])
        return result
        pass

    def rotate(self, matrix):
        '''面试题 01.07. 旋转矩阵
        
        '''
        n = len(matrix)
        for i in range(1, n):
            for j in range(i):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        # print(matrix)
        
        for i in range(n):
            for j in range(n//2):
                matrix[i][j], matrix[i][n-1-j] = matrix[i][n-1-j], matrix[i][j]
        # print(matrix)

    def generateParenthesis(self, n):
        '''22. 括号生成
        
        '''
        result = []
        def dfs(left, right, curStr):
            if left==0 and right==0:
                result.append(curStr)
                return
            if left>0:
                dfs(left-1, right, curStr+"(")
            if right>left:
                dfs(left, right-1, curStr+")")
        dfs(n, n, "")
        return result

    def reverseWords(self, s):
        '''151. 翻转字符串里的单词
        
        '''
        lst = s.split()
        n = len(lst)
        for i in range(n//2):
            lst[i], lst[n-1-i] = lst[n-1-i], lst[i]
        return " ".join(lst)

    def numberOfSubarrays(self, nums, k):
        n = len(nums)
        if k>n:
            return 0
        if n==1:
            if nums[0]==0:
                return 0
            else:
                return 1
        i, j = 0, 1
        count = 0
        if nums[i]%2!=0:
            count+=1
        if nums[j]%2!=0:
            count+=1
        result = 0
        while i<n-1 and j<n-1:
            # print(count)
            if count<k:
                j += 1
                if nums[j]%2!=0:
                    count += 1
            elif count==k:
                result += 1
                print(result)
                # if nums[i]%2!=0: #奇数
                #     while nums[j]%2==0 and j<n and i<n: #是偶数
                #         result += 1
                #         j += 1
                # else: #偶数
                #     # result += 1
                #     i+=1
                
        return result
        pass

    def isHappy(self, n):
        result = 0
        hset = set()
        while True:
            while n!=0:
                bit = n%10
                n = n//10
                result += bit*bit
                print(bit, n)
                print(result)
            if result==1:
                return True
            elif result not in hset:
                    hset.add(result)
                    n = result
                    result = 0
            elif result in hset:
                return False

    def lengthOfLongestSubstring(self, s):
        n = len(s)
        if n<2:
            return n
        i, j = 0, 1
        result = 0
        while i<n and j<n:
            if len(s[i:j+1])==len(set(s[i:j+1])): #没有重复元素
                result = max(result, j+1-i)
                j += 1
            else:
                while len(s[i:j+1])>len(set(s[i:j+1])) and i<j:
                    i += 1
        return result

    def maxSubarray(self, nums):
        '''53. 最大子序和

        '''
        n = len(nums)
        if n==0:
            return 0
        if n==1:
            return nums[0]
        cnt = nums[0]
        result = nums[0]
        for i in range(1, n):
            if cnt<0:
                cnt = 0
            if nums[i]<0:
                cnt += nums[i]
            else:
                cnt += nums[i]
            result = max(result, cnt)
        result = max(result, cnt)
        return result

    def isValidBST(self, root):
        '''98. 验证二叉搜索树

        '''
        def recursive(root, low, high):
            if not root:
                return True 
            # if root.val < low or root.val > high:
            if not low<root.val<high:
                return False
            if root.left and root.left.val > root.val:
                return False
            if root.right and root.right.val < root.val:
                return False
            return recursive(root.left, low, root.val) and recursive(root.right,root.val, high)
        return recursive(root, -(2**32), 2**32)
        pass

    def myPow(self, x, n): 
        '''50. Pow(x, n)

        '''
        def recursion(x, n):
            if n==0:
                return 1
            if n==1:
                return x
            if n%2==0:
                return recursion(x*x, n//2)
            else:
                return recursion(x, n-1)*x
            # return recursion(x, n-1)*x # 超时。上面是优化后的。
        result = recursion(x, abs(n))
        if n<0:
            return 1/result
        return result
        pass
    def myPow2(self, x, n): #超时
        if x==0:
            return 0
        if n==0:
            return 1
        absn = abs(n)
        result = 1
        for i in range(absn):
            result *= x
        if result == 0:
            return 0
        if n<0:
            result = 1/result
        return result

    def sumNums(self, n):
        '''面试题64. 求1+2+…+n

        '''
        n>1 and self.sumNums(n-1)
        self.curSum += n
        return self.curSum
        pass

    


st = Solution()
x = 2.00000
n = 10
result = st.sumNums(10)
print(result)



'''
def quickSort(nums, low, high):
    # 快排
    if low>high:
        return 
    i, j = low, high
    key = nums[low]
    while i<j:
        while i<j and nums[j]>=key:
            j -= 1
        while i<j and nums[i]<=key:
            i += 1
        if i<j:
            nums[i], nums[j] = nums[j], nums[i]
    if i==j:
        nums[low], nums[j] = nums[j], nums[low] 
            # break
    quickSort(nums, low, i-1)
    quickSort(nums, i+1, high)
lst = [1,4,9,2,3,0]
# lst = [1,2,1]
# lst = [5,1,5,2,5,7]
print(lst)
quickSort(lst,0,len(lst)-1)
print(lst)
'''



# ListNode
# l0 = ListNode(0)
# l1 = ListNode(1)
# l2 = ListNode(2)
# l3 = ListNode(3)
# l0.next = l1
# l1.next = l2
# l2.next = l3


# Tree
# r1 = TreeNode(10)
# r2 = TreeNode(5)
# r3 = TreeNode(15)
# r4 = TreeNode(6)
# r5 = TreeNode(20)
# r1.left = r2
# r1.right = r3
# r3.left = r4
# r3.right = r5