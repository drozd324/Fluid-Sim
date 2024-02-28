def calculateTotalSum(strin, *arguments):
    totalSum = 0
    for number in arguments:
        totalSum += number
    print(strin, totalSum)
 
# function call
calculateTotalSum("lol", 5, 4, 3, 2, 1)