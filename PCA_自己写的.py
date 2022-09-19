#针对变量间有相互关系的情况
import numpy as np
#PCA类的主要作用是将数据x去相关化，即返回一个t，而t中的数据都不相关，随后再将t取代x送到MLR类中
'''
属性：X和Y矩阵
方法：分解，确定独立组分数，可借助PCA
方法：建模，根据独立组分数，得到T，将T和Y用MLR建模
方法：预报，传递Xnew，预报Ynew
'''
class PCA:
    def __init__(self,x):
        self.x=x  #x是数据矩阵
    def SVDdecompose(self): #对x做svd分解
        u,s,v=np.linalg.svd(self.x,full_matrices=False)   #false即没有虚数
        self.lamda=s**2  #lamda为特征值，这里是点乘(对应元素相乘），s为一维数组，元素个数为特征值数目
        self.p=v.T   #用p的好处是后面全部可以用列向量来处理
        self.t=u*s  #这里是矩阵乘法
        compare=self.lamda[:-1]/self.lamda[1:]  #即lamda数组内前一个元素比上后一个元素，找突变，书P36
        return compare     #返回特征值的比值

    #在返回cmpare值中，突变位置与前面的元素总个数即特征值数目k，再送到PCA内
    def PCAdecompose(self,k):
        p=self.p[:,:k]  #意思是元素每行都要，但是列取[0,k)，即p[all,0~k-1]
        t=self.t[:,:k]
        return t,p   #即最后，只取前k个特征值

from MLR import MLR

data=np.loadtxt(r'D:\乱七八糟\physics物理\python 精品课\数据\多元线性回归数据02.txt')
#一共7列数据，1-6是x，即样本数据，其中4-6皆与前3个相关，7是检测值y
x=data[:,:-1]
y=data[:,-1]  #注意，y是个列向量
pca=PCA(x)
judge=pca.SVDdecompose()
print(judge)
k=int(input('k='))
T,P=pca.PCAdecompose(k)
mlr=MLR(T,y)
mlr.fit()
print(mlr.Ftest(0.05))




