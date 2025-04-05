# Exercise 2:
li = [3, 5, 1, 1, 8]

#Find all unique subsets with at least 2 elements and that leave enough elements to have
#another subset with at least 2 elements
#This utilizes an algorithm I found for generating all possible subsets:
    # - The algorithm uses a bitwise comparison between i (which is all possible combinations of
    # - 1s and 0s) and j (wich moves a 1 to be compared with each bit of i)
subsets = []
n = len(li)
for i in range(1, 2**n):
    subset = []
    for j in range(n):
        if i & (1 << j):
            subset.append(li[j])
    if len(subset) >= 2 and len(subset) <= len(li) - 2 and subset not in subsets:
        subsets.append(subset)

#Finds sums of all subsets
subset_sums = [sum(subset) for subset in subsets]

#Find which subset sums are equal
index1 = None
for i in range(len(subset_sums)):
    for j in range(len(subset_sums)):
        if subset_sums[i] == subset_sums[j] and i != j:
            index1, index2 = i, j
            break
    if index1 != None:
        break
print(f'The two equally balanced subsets are: {subsets[index1]} and {subsets[index2]}')