'''f = open('DataSets/Metro_Interstate_Traffic_Volume.csv', 'r')
out = []
min = 9999999
max = -9999999
for line in f.readlines()[1:]:
    #if line.split(';')[4] == '?':
    #    next = out[len(out) - 1]
    #else:
    next = float(line.split(',')[8])
    if next < min:
        min = next
    if next > max:
        max = next
    out.append(next)
f.close()
w = open('DataSets/traffic.csv', 'w')
for i in out:
    w.write(str(((i-min)/(max-min))*100) + '\n')
w.close()'''

f = open('DataSets/hpc.csv', 'r')
lines = f.readlines()
f.close()
f2 = open('DataSets/hpc.csv', 'w')
for line in lines:
    f2.write(str(float(line)*5) + '\n')
f2.close()