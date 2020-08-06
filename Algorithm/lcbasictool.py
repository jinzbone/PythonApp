class solution():
    def gcd(self,a,b):
        '''求a,b的最大公约数
        '''
        return a if b==0 else self.gcd(b, a%b)
        '''
        附 辗转相除法
        两个正整数a和b（a>b），它们的最大公约数等于a除以b的余数c和b之间的最大公约数。
        '''

    def binarySearch(self, A, k):
        '''二分查找 基础版
        在有序数组里查找k，如果存在，返回下标
                        如果不存在，返回-1
        '''
        n = len(A)
        if n==0:
            return -1
        if n==1:
            return 0 if A[0]==k else -1
        i, j = 0, n-1
        while i<=j:
            mid = (i+j)//2
            if A[mid]<k:
                i = mid + 1
            elif A[mid]>k:
                j = mid - 1
            else:
                return mid
        return -1

    def binarySearch2(self, A, k):
        '''二分查找 进阶版
        在有序数组里查找k，如果存在，返回下标
                        如果不存在，返回它应该插入的位置的下标
        '''
        n = len(A)
        if n==0:
            return -1
        if n==1:
            return 0 if A[0]==k else -1
        i, j = 0, n-1
        while i<=j:
            mid = (i+j)//2
            if A[mid]<k:
                i = mid + 1
            else:
                j = mid - 1
        return i

s = solution()
A = [0,2,5]
result = s.binarySearch(A, 2)
print(result)