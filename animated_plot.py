# starting parameters:
time_series_filename = 'DataSets/AirPassengers.csv'
max_error = 0

# first we need to read in the trends supplied in our txt files
angles_file = open('angles.txt', 'r')
length_file = open('length.txt', 'r')

angles = []
lengths = []

for line in angles_file.readlines():
    if line.rstrip():
        angles.append(float(line))
        
print(len(angles))
for line in length_file.readlines():
    if line.rstrip():
        lengths.append(float(line)*1)

dataset = []
f = open(time_series_filename, 'r')
for line in f:
    dataset.append(float(line))

import PiecewiseLinearSegmentation
x = PiecewiseLinearSegmentation.Bottomup(max_error=max_error)
trends = []
for trend in x.transform(dataset):
    trends.append(list(trend))

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

fig = plt.figure()
#creating a subplot 
ax1 = fig.add_subplot(1,1,1)
#ax1.set_xlim([0, trends[-1][2]])
def animate(i):
    seq_length = 2
    i %= (len(trends)-seq_length) # each iteration of the animation will add one more trend
    i+=seq_length
    xs = []
    ys = []
    ax1.clear()
    
    '''for ind in range(i,len(trends)):
        xs.append(trends[ind][0])
        xs.append(trends[ind][2])
        ys.append(trends[ind][1])
        ys.append(trends[ind][3])
        ax1.plot(xs, ys, color='white')'''
    for ind in range(0,i):
        xs.append(trends[ind][0])
        xs.append(trends[ind][2])
        ys.append(trends[ind][1])
        ys.append(trends[ind][3])
        ax1.plot(xs, ys, color='orange')
    ax1.plot([trends[i][0],trends[i][2]], [trends[i][1],trends[i][3]], color='magenta', linewidth=5, alpha=0.5)

    if i>=4:
        newx = trends[i][0] + lengths[i-seq_length]
        print( lengths[i-seq_length] - trends[i][0])
        newy = trends[i][1] + lengths[i-seq_length]*np.tan(np.deg2rad(angles[i-seq_length]))
        ax1.plot([trends[i][0],newx], [trends[i][1],newy], color='black', linewidth=2, alpha=0.8)
    #ax1.set_xlim([0, trends[-1][2]])
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Live graph with matplotlib')	
	
    
ani = animation.FuncAnimation(fig, animate, interval=1000) 
plt.show()