import numpy as np
import math
#KNN算法：针对点集合
class point:
    def __init__(self,coor,cid):
        self.coor=coor   #coordinate
        self.cid=cid     #character

    def distance(self,otherpoint):
        #n维的点，维度由列表长度来定
        m=np.array(self.coor)
        n=np.array(otherpoint.coor)
        self.distance_vector=(m-n)         #vector存储了两点间的位置向量
        sum=0
        for i in self.distance_vector:
            sum=sum+i**2
        self.distance_abs=np.sqrt(sum)     #abs存储了距离

class knn():
    """
    knn算法总结：
    1.初始化：输入已知点，用self.known_list接收所有已知点
    judge过程：
    2.先将未知点都用unknown_list接收
    3.用循环对每一个未知点进行处理
    4.对每个未知点，用循环取每个已知点与该未知点的距离，但记住，必须存进已知点的distance_abs属性中
    5.利用已知点属性distance_abs将self.known_list重新按距离从小到大排序：此处注意sort函数的用法以及np.array带来的问题
    6.对前k个进行加权运算，决定该未知点的cid
    7.第3步的循坏结束后，返回所有未知点的cid列表
    """

    def __init__(self,x,y):      #  x为坐标列表，y为cid列表
        known_list=[]
        for m,n in zip(x,y):
            p=point(m,n)     #合并为一个点集
            known_list.append(p)
        self.known_list=known_list  #knn的known_list属性是以所有已知点形成的点列表

    def judge_point(self,x,k): # x为未知点坐标列表，k为算法要求的k值
        unknown_list=[]
        for coor in x:            #先让未知点坐标输入未知cid=0，形成点类集
            unknown_point=point(coor,0)
            unknown_list.append(unknown_point)

        result=[]  #作为全体未知点的cid返回列表，与函数judge代码最后呼应
        for unknown_point in unknown_list:   #将所有未知点一个个处理
            #对一个未知点的处理方法：
            for known_point in self.known_list:
                known_point.distance(unknown_point) #套用距离函数，求每一个已知点与未知点的距离，并把距离属性放到已知点中

            #在距离都计算好后将距离从小到大排序 1.已知点集known_list 2.未知点集 unknown_list
            self.known_list.sort(key=lambda x:x.distance_abs) #known_list里元素是点，按已知点的distance_abs排序

            #选择前k个进行加权运算
            sum=0
            for i in range(k):
                known_qualify=self.known_list[i]
                value=known_qualify.cid/known_qualify.distance_abs
                sum=sum+value

            if sum>=0:
                unknown_point.cid=1
            else:
                unknown_point.cid=-1
            #以矩阵形式返回
            result.append(unknown_point.cid)
        return result



#数据
x1=[[5.1, 3.5 ],  [4.9, 3.0],  [4.7, 3.2], [4.6, 3.1], [5.0 , 3.6], [5.4, 3.9],
    [4.6, 3.4], [5.0, 3.4 ], [4.4, 2.9], [4.9, 3.1]]
x2=[[5.5, 2.6 ],  [6.1, 3.0],  [5.8, 2.6], [5.0, 2.3],
    [5.6, 2.7],[5.7, 3.0 ],  [5.7, 2.9],  [6.2, 2.9], [5.1, 2.5],  [5.7, 2.8]]

y1=[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
y2=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

x1x=x1.pop()
x2x=x2.pop()
x_known=np.array(x1+x2)
x_unknown=np.array([x1x,x2x])
y1y=y1.pop()
y2y=y2.pop()
y_known=np.array(y1+y2)

knn1=knn(x_known,y_known)
result=knn1.judge_point(x_unknown,9)
print(result)















