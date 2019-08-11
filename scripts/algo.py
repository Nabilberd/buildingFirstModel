def binary_search(arr, num):

    first = 0
    last = len(arr) - 1  # len(shiftArr) - 1

    found = False

    while (first <= last and not found):
        middle = (first + last) // 2
        if arr[middle] == num:
            found = True
            return middle
        else:
            if num < arr[middle] and middle >= 0:
                last = (middle - 1)%2
            else:
                first = (middle + 1)%2
    return -1

arrays = [9, 12, 17, 2, 4, 5]
arrays.sort()
print(binary_search(arrays, 5))