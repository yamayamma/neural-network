import numpy as np
import mnist

# データ読み込み
mn = mnist.MNIST(pathMNIST = './data/mnist')
datL = mn.getImage('L')
labL = mn.getLabel('L')
datT = mn.getImage('T')
labT = mn.getLabel('T')

# データの大きさ
NL = 60000
NT = 10000
D = 784

DatL = [0] * NL
rec = 0

for l in range(NL):
    DatL[l] = np.sum(np.square(datL[l])) # 計算の準備

for i in range(NT):
    dist = DatL -  (datL @ datT[i]) * 2 # 距離の効率的な計算
    if labL[np.argmin(dist)] == labT[i]:
        rec += 1 # 最も距離が近いものとクラス番号が一致していればカウント
    if i%1000 == 0:
        print(i) # 進み具合

print(rec/NT*100, '%') # 識別率 96.91%