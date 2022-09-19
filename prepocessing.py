import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


'''
1. preprocessing for Xtrain
'''
Xtrain=pd.read_csv(r'E:\Amrican Express data\amex-default-prediction\train_data_S.csv',header=0)
#ytrain=pd.read_csv(r'E:\Amrican Express data\amex-default-prediction\train_labels_S.csv',header=0)
# to find out the features' size
print('feature size:',Xtrain.columns.shape[0])  # it has 191 features
print('*'*20)

# first: D_63 has 6 type(use pd.get_dummies to verify), convert them to int
# for a dict, i represents key, dict[i] represents value: use lambda to transform str to int
Xtrain['D_63']=Xtrain['D_63'].apply(lambda t:{'CL':0,"CO":1,"CR":2,"XL":3,"XM":4,"XZ":5}[t]).astype(dtype='int')
Xtrain['D_64'] = Xtrain['D_64'].apply(lambda t: {np.nan:-1, 'O':0, '-1':1, 'R':2, 'U':3}[t]).astype(dtype='int')
Xtrain.to_csv(r'E:\Amrican Express data\amex-default-prediction\train_data_S_c.csv',index=0)
# there are many noises(they are too small), so 1st winner get it mutiply by 100, and make these minor features to 0
for col in Xtrain.columns:
    if col not in ['customer_ID', 'S_2', 'D_63', 'D_64']:
        Xtrain[col] = np.floor(Xtrain[col] * 100) #np.floor 向下取整

# use one-hot coding to transform certain column: attach to the right of the dataframe
# if flag=True, it will drop original column
def one_hot_transform(x,cols,flag=True):
    for i in cols:
        y=pd.get_dummies(pd.Series(x[i]),prefix="onehot_%s"%i)
        x=pd.concat([x,y],axis=1)
    if(flag):
        x.drop(cols,axis=1,inplace=True)
    return x

x=Xtrain.iloc[:10,3:10]
x=one_hot_transform(x,x.columns[:2],flag=False)
x.to_csv(r'E:\Amrican Express data\amex-default-prediction\test.csv')
#Xtrain.to_csv(r'E:\Amrican Express data\amex-default-prediction\train_data_S_c.csv',index=0)
