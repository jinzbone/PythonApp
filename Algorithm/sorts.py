class SortAlgorithm():
    def __init__(self, ):
        pass

    def bubbleSort(self, nums):
        n = len(nums)
        for i in range(n-1):
            for j in range(n-1-i):
                if nums[j]>nums[j+1]:
                    nums[j], nums[j+1] = nums[j+1], nums[j]
        print(nums)
    
    def selectSort(self, nums):
        n = len(nums)
        for i in range(n-1):
            idx = i
            for j in range(i+1, n):
                if nums[j]<nums[idx]:
                    idx = j
            nums[i], nums[idx] = nums[idx], nums[i]
        print(nums)
        pass

    def quickSort(self, nums, low, high):
        if low>high:
            return
        i, j = low, high
        key = nums[i]
        while i<j:
            while i<j and nums[j]>=key:
                j -= 1
            while i<j and nums[i]<=key:
                i += 1
            if i<j:
                nums[i], nums[j] = nums[j], nums[i]
            if i==j:
                nums[i], nums[low] = nums[low], nums[i]
                break
        self.quickSort(nums, low, i-1)
        self.quickSort(nums, i+1, high)
        



sa = SortAlgorithm()
nums = [6,1,2,7,9,3,4,5,10,8]
sa.quickSort(nums, 0, len(nums)-1)