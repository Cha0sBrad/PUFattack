import csv
import random
import numpy as np


def shape(M):
    return len(M), len(M[0])


out1 = open('traindatasetin.csv', 'w')
out2 = open('traindatasetout.csv', 'w')
csv_writer1 = csv.writer(out1)
csv_writer2 = csv.writer(out2)
csv_file = csv.reader(open('仿真Arbiter_PUF.csv', 'r'))

PUFdelay_np = []
PUFdelay = []
delay1 = 0
delay2 = 0

for i in csv_file:
    PUFdelay.append(i)

PUFdelay_np = np.array(PUFdelay, dtype=float)

C_np = [([0] * 64) for i in range(4)]
C_np_csv = []
seed = "01"
counter1 = 0
counter = 0

while counter != 2000:
    for i in range(64):
        binary = random.choice(seed)
        if binary == "1":
            delaymiddle = delay2
            delay2 = delay1
            delay1 = delaymiddle
            delay1 += float(PUFdelay[1][i])
            delay2 += float(PUFdelay[2][i])
            C_np[0][i] = 0
            C_np[3][i] = 0
            counter1 += 1
            if counter1 % 2 != 0:
                C_np[1][i] = 1
                C_np[2][i] = -1
            if counter1 % 2 == 0:
                C_np[1][i] = -1
                C_np[2][i] = 1
        if binary == "0":
            delay1 += float(PUFdelay[0][i])
            delay2 += float(PUFdelay[3][i])
            C_np[1][i] = 0
            C_np[2][i] = 0
            # counter0 += 1
            if counter1 % 2 != 0:
                C_np[0][i] = -1
                C_np[3][i] = 1
            if counter1 % 2 == 0:
                C_np[0][i] = 1
                C_np[3][i] = -1
    for i in range(4):
        for j in range(64):
            C_np_csv.append(C_np[i][j])
    csv_writer1.writerow(C_np_csv)
    if counter1 % 2 == 0:
        if (delay1 > delay2):
            csv_writer2.writerow("1")
        else:
            csv_writer2.writerow("0")
    else:
        if (delay1 <= delay2):
            csv_writer2.writerow("1")
        else:
            csv_writer2.writerow("0")
    counter1 = 0
    delay1 = 0
    delay2 = 0
    C_np_csv = []
    counter += 1