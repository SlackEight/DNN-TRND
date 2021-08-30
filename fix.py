f = open('DataSets/TQQQ.csv', 'r')
out = []
min = 9999999
max = -1
for line in f.readlines()[1:]:
    next = float(line.split(',')[4])
    if next < min:
        min = next
    if next > max:
        max = next
    out.append(next)
f.close()
w = open('DataSets/TQQQ.csv', 'w')
for i in out:
    w.write(str((i-min)/(max-min)) + '\n')
w.close()