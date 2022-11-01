import time
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.datasets import make_blobs

# 用matplotlib画图时遇到中文或者负号无法显示的情况，加上下面这两句
# matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
# matplotlib.rcParams['axes.unicode_minus'] = False

"""
对于二维数组的切片操作
X[:,0]  取所有行第0列的数据
X[0,:]  取第0行中所有列的数据
X[:,1:] 取所有行中，第一列到最后一列的所有数据（第0列不要）
"""


def distance(vecA, vecB):
    """
    计算两个向量的欧式距离
    """
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))


def randCenter(dataSet, k, samplesNum):
    """
    随机生成k个点作为质心，其中质心均在真个整个数据的边界之内
    """
    n = dataSet.shape[1]  # 获得数据维度
    # 创建一个k行n列的全零矩阵
    centroids = np.mat(np.zeros((k, n)))
    """centroids = np.mat(np.zeros((k, n)))
    for j in range(n):
        minJ = np.min(dataSet[:, j])
        print(minJ.shape)
        rangeJ = np.float(np.max(dataSet[:, j]) - minJ)
        # np.random.rand返回k个服从0~1均匀分布的的随机样本值
        centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1)"""
    for j in range(k):
        index = int(np.random.uniform(0, samplesNum))
        centroids[j, :] = dataSet[index, :]
    return centroids


def KMeans(dataSet, k, dist=distance, createCenter=randCenter):
    """
    k-means聚类算法 返回最终的k各质心和点的分配结果
    """

    # 获取样本数量
    m = dataSet.shape[0]
    # 构建一个簇分配结果矩阵，共两列，第一列为样本所属的簇类值，第二列为样本到簇心的误差
    clusterAssignment = np.mat(np.zeros((m, 2)))
    # 初始化k个质心
    centroids = randCenter(dataSet, k, m)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        # 遍历所有样本
        for i in range(m):
            minDistance = np.inf  # 先设置为无穷大
            minIndex = -1
            # 找出与样本i最近的质心
            for j in range(k):
                distanceJI = distance(centroids[j, :], dataSet[i, :])
                if distanceJI < minDistance:
                    minDistance = distanceJI
                    minIndex = j
            # 更新样本每一行所属的簇
            if clusterAssignment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssignment[i, :] = minIndex, minDistance ** 2

        # 更新簇质心
        for center in range(k):
            # 获取所属簇center的所有点

            '''
            矩阵.A的意思是将矩阵转化为array数组类型
            让一个数组 == 一个数
            会返回一个True或False的数组
            >>> a = np.array([1,2,0,3])
            >>> a == 3
            array([False, False, False,  True])
            
            np.nonzero(a) 会返回一个数组 数组里是 a里 非零元素的索引
            从dataSet中拿这些索引得到的数组成一个矩阵pluster
            '''
            # np.nonzero返回
            pCluster = dataSet[np.nonzero(clusterAssignment[:, 0].A == center)[0]]
            # 沿矩阵pCluster列的方向求均值 等号左右的维度都是(1, n)
            centroids[center, :] = np.mean(pCluster, axis=0)
    return centroids, clusterAssignment


k = 5
# random_state 是随机生成器的种子
# make_blobs返回两个数组，X的维度是[n_samples, n_features] y的维度是[n_samples]
X, y = make_blobs(n_samples=1000, n_features=2, centers=k, random_state=1)

s = time.time()
myCentroids, clusterAssnment = KMeans(X, k)
print("用K-means算法原理聚类耗时:", time.time()-s,"s")
# 将matrix转化为ndarray
centroids = myCentroids.A
#
y_kmeans = clusterAssnment[:, 0].A[:, 0]

# 未聚类前的数据分布
# subplot(121)表示将图像分割成1行2列 当前表格在位置1
plt.subplot(121)
# plt.scatter绘制散点图
# x,y是散点图的坐标 s散点的面积 c散点的颜色默认蓝色 marker散点样式默认实心圆
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.xlabel('petal length')
plt.ylabel('petal width')
#plt.title("为聚类前的数据分布")
# plt.subplots_adjust()wspace,hspace 子图之间的横向间距、纵向间距分别与子图平均宽度、平均高度的比值
plt.subplots_adjust(wspace=0.5)

plt.subplot(122)
plt.scatter(X[:, 0], X[:, 1], s=50, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, alpha=0.5)
plt.xlabel('petal length')
plt.ylabel('petal width')
#plt.title("用K-Means算法原理聚类的效果")


plt.show()




