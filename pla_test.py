import PiecewiseLinearSegmentation
import matplotlib.pyplot as plt
x = PiecewiseLinearSegmentation.Bottomup(max_error=500)
n = []
f = open('DataSets/eth.csv', 'r')
for line in f:
    n.append(float(line))
data = []
X,Y = [], []
for trend in x.transform(n):
    X.append(trend[0])
    Y.append(trend[1])
    X.append(trend[2])
    Y.append(trend[3])

plt.plot(n)
plt.plot(X,Y)
plt.show()