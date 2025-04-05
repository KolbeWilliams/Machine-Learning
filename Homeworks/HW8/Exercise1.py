#Exercise 1:
def minNotRepeated(li):
    minimum = min(li)
    count = li.count(minimum)
    if count > 1:
        li = [num for num in li if num != minimum]
        return minNotRepeated(li)
    else:
        return minimum

li = [2, 2, 3, 4, 5]
print('The smallest number not repeated is :', minNotRepeated(li))