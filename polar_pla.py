import PiecewiseLinearSegmentation
import matplotlib.pyplot as plt
import statistics
import numpy as np
from numpy.lib.function_base import average

def bottom_up_pla(time_series, max_error):

    x = PiecewiseLinearSegmentation.Bottomup(max_error=max_error)
    output = []

    for trend in x.transform(time_series):
        # first we need the angle of inclination. We will normalize this to a value from -1 to 1 (-90 degrees to 90 degrees)
        # each trend has the structure [startX, startY, endX, endY], however this doesn't lend itself well to ML since 
        # incorrect x prediction values can go backwards.
        startX, startY, endX, endY = trend[0], trend[1], trend[2], trend[3]
        angle = np.rad2deg(np.arctan2(endY - startY, endX - startX))
        # now we need to normalize the angle to a value from -1 to 1
        #angle /= 90
        
        # now we need to calculate the length of the trend
        length = np.sqrt((endX - startX)**2 + (endY - startY)**2)
        output.append(angle)
        output.append(length)#/200)
    max_length = max(output)
    w = open('trends.csv','w')
    for i in range(0,len(output),2):
        w.write(str(output[i])+","+str(output[i+1])+"\n")

    return output

def display_trends(trends, startY):
    # this function will reconstruct a graph using the trends generated by the bottom_up_pla function
    # it will also display the graph
    X = [0]
    Y = [startY]
    for i in range(1,len(trends)):
        #print(trends[i])
        X.append(X[i-1]+(trends[i][1]*200)*np.cos(np.deg2rad(trends[i][0]*90)))
        #print(X[i-1]+(trends[i][1]*10)/np.cos(np.deg2rad(trends[i][0])))
        Y.append(Y[i-1]+(trends[i][1]*200)*np.sin(np.deg2rad(trends[i][0]*90)))
        #print(Y[i-1]+(trends[i][1])*np.sin(np.deg2rad(trends[i][0])))
    plt.plot(X,Y)
    #plt.show()


    trends = []
    f = open('trends.csv', 'r')
    for line in f:
        trends.append([float(line.split(',')[0]), float(line.split(',')[1])])
    display_trends(trends, 733.04)


'''import PiecewiseLinearSegmentation

x = PiecewiseLinearSegmentation.Bottomup(max_error=500)
n = []
f = open('DataSets/eth.csv', 'r')
for line in f:
    n.append(float(line))
data = []
X,Y = [],[]
for trend in x.transform(n):
    X.append(trend[0])
    Y.append(trend[1])
    X.append(trend[2])
    Y.append(trend[3])
plt.plot(X,Y)
plt.show()

#plt.plot(n)
#plt.plot(X,Y)
#plt.show()'''