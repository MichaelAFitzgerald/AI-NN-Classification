import pandas as p
import matplotlib.pyplot as plt

import Perceptron as per
import numpy as np


# Overwrites a value in a list with the normalized value
def normalization(iMax, iMin, iVal):
    return (iVal - iMin) / (iMax - iMin)


# Takes in an unnormalized list and returns it after normalization
def getNormalizeArr(data):
    minHeight = data['Height'].min()
    maxHeight = data['Height'].max()
    minWeight = data['Weight'].min()
    maxWeight = data['Weight'].max()
    for i in data.iterrows():
        i[1]['Height'] = normalization(maxHeight, minHeight, i[1]['Height'])
        i[1]['Weight'] = normalization(maxWeight, minWeight, i[1]['Weight'])


# Takes in a list and a number, and will return two lists
# One that's 25% of the list, the other is the rest
def separateArr(data):
    newFrame = p.DataFrame(columns=('Height', 'Weight', 'Gender'))
    remFrame = p.DataFrame(columns=('Height', 'Weight', 'Gender'))
    count = 0
    for i in data.iterrows():
        if count % 4 == 0:
            newFrame = newFrame.append(other=i[1], ignore_index=True)
        else:
            remFrame = remFrame.append(other=i[1], ignore_index=True)
        count += 1
    return newFrame, remFrame


def plotData(dataframe):
    for val in dataframe.iterrows():
        if val[1]['Gender'] == 1:
            plt.scatter(val[1]['Height'], val[1]['Weight'], color='pink')
        else:
            plt.scatter(val[1]['Height'], val[1]['Weight'], color='blue')


def drawSeparator(hWeight, wWeight, bias, num):
    x = np.array(np.arange(0, 1, 0.1))
    y = (-1 * bias - hWeight * x) / wWeight
    plt.plot(x, y)
    plt.savefig(user_dataset + activation + num + '.png')
    plt.show()


# Defining variables
errorThresholdA = 1 * 10 ** -5
errorThresholdB = 1
errorThresholdC = 1

#
activation = None
errorVal = None
user_dataset = None

# Adding user functionality
while errorVal is None:
    user_dataset = str(input('Choose which dataset you want to test: A, B, or C\n'))
    if user_dataset == 'A' or user_dataset == 'a':
        errorVal = errorThresholdA
        user_dataset = 'A'
    elif user_dataset == 'B' or user_dataset == 'b':
        errorVal = errorThresholdB
        user_dataset = 'B'
    elif user_dataset == 'C' or user_dataset == 'c':
        errorVal = errorThresholdC
        user_dataset = 'C'
    else:
        print('You have made a bad choice, try again\n')

while activation is None:
    user_activation = input('Choose whether you want to test Hard or Soft activation functions\n')
    if user_activation is 'Hard' or user_activation == 'hard' or user_activation == 'h':
        activation = 'Hard'
    elif user_activation == 'Soft' or user_activation == 'soft' or user_activation == 's':
        activation = 'Soft'
    else:
        print('You have made a bad choice, try again\n')

# Read the excel spreadsheet
plot = p.read_excel('Group' + user_dataset + '.xlsx')

# Turns the plot into a list and normalizes it
df = p.DataFrame(plot, columns=['Height', 'Weight', 'Gender'])
for x in ['Height', 'Weight', 'Gender']:
    df[x] = df[x].astype(float)
getNormalizeArr(df)

# Plot the graph

# Turns the original list into two lists for the purposes of the first test
test1, train1 = separateArr(df)
plotData(test1)
plotData(train1)

per1 = per.Perceptron(activation)

weights1 = per1.train(errorVal, train1)
drawSeparator(weights1[0], weights1[1], weights1[2], '1')
per1.test(weights1, test1)

# Turns the original list into two lists for the purposes of the second test
train2, test2 = separateArr(df)
plotData(test2)
plotData(train2)

per2 = per.Perceptron(activation)

weights2 = per2.train(errorVal, train2)
drawSeparator(weights2[0], weights2[1], weights2[2], '2')
per2.test(weights2, test2)
