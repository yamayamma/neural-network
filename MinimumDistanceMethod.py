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
K = 10

# プロトタイプ作成
mimg = np.empty((K, D))
for l in range(K):
    mimg[l] = np.mean(datL[labL==l],axis=0)

dist = [0] * K
rec = 0

for i in range(NT): # NT個のデータを比較
    dist = np.sum(np.square(datT[i] - mimg), axis=1) # それぞれのプロトタイプとの距離をとる
    if np.argmin(dist) == labT[i]:
        rec += 1 # 最も距離が近いものとクラス番号が一致していればカウント

print(rec/NT*100, '%') # 識別率 82.03%