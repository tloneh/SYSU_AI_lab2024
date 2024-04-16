def BinarySearch(nums, target):
    left = 0
    right = len(nums) - 1
    # 定义左右指针，分别指向数组开头和结尾
    while left <= right: # 当left小于right时数组查找完毕，跳出
        mid = (right - left) // 2 + left # 取中间位置节点
        num = nums[mid]
        if num == target: # 查找成功
            return mid
        elif num > target: # 中间值大于目标值
            right = mid - 1
        else: #小于目标值
            left = mid + 1
    return -1 # 未找到

# 测试
nums = [1, 3, 5, 9, 17, 58, 98]
target = 5
res = BinarySearch(nums, target)
print(res) # res = 2, answer is right