#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 11:39:35 2018

@author: genggejianyi
"""
##导入需要的包
import os
import face_recognition
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import glob
from skimage import transform
from sklearn import  linear_model,metrics
from sklearn import neighbors
from sklearn.model_selection import train_test_split,KFold
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from PIL import Image,ImageDraw
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten,Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D,AveragePooling2D
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array




###查看人脸特征识别线
##如眼睛，鼻子，嘴巴等等，在人脸中标出，以第一个人的正面照为例
image=face_recognition.load_image_file('/Users/genggejianyi/Desktop/大数据班/python作业/FaceDB/database/1/MVC-003F.JPG')
face_landmarks_list=face_recognition.face_landmarks(image)
##打印出在图中识别出的人脸数目
print("I found {} face(s) in this photograph.".format(len(face_landmarks_list)))

#创建一个PIL对象，方便标出人的五官
pil_image=Image.fromarray(image)
d=ImageDraw.Draw(pil_image)

for face_landmarks in face_landmarks_list:
    #打印出人的五官像素点位置
    for facial_feature in face_landmarks.keys():
        print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks[facial_feature]))
    #用白线描出人脸中的五官位置
    for facial_feature in face_landmarks.keys():
        d.line(face_landmarks[facial_feature], width=5)
#展示图片
pil_image.show()


###数据增强，随机对原始图片进行旋转，调节色彩对比度，缩放等等操作生成人脸图片，增大样本量
data_gen=ImageDataGenerator(rotation_range=30,horizontal_flip=True,channel_shift_range=40,width_shift_range=0.2,
                            height_shift_range=0.2,
                            rescale=1/255,fill_mode='nearest',data_format='channels_last')

##函数generate_img生成新的人脸图片，扩充样本量，以每个人的第三张与第六张图片为样板生成
# 并将图片存放至各个文件夹
def generate_img():
    for i in range(1,115):
        img=load_img(path+'/'+str(i)+'/'+'MVC-003F.JPG')
        x=img_to_array(img,data_format='channels_last')
        x=x.reshape((1,)+x.shape)
        j=0
        for batch in data_gen.flow(x,batch_size=1,save_to_dir=path+'/'+str(i),save_prefix='MVC-003F',save_format='JPG'):
            j+=1
            if j>19:
                break
##生成图片
if __name__=='__main__':
    generate_img()
            
    
    
###读取图片
##设定图片路径
path='/Users/genggejianyi/Desktop/大数据班/python作业/FaceDB/database'
##函数read_img用来读取图片，检测出人脸，并将人脸图片转为200*200*3格式的RGB图片，返回array格式图片数据和标签
def read_img(path):
    cate=[path+'/'+str(i) for i in range(1,115)]
    imgs=[]
    labels=[]
    # 遍历路径以及索引
    for idx,folder in enumerate(cate):
        # 遍历后缀为.JPG的图片
        for im in glob.glob(folder+'/*.JPG'):
            print('reading the images:%s'%(im))
            # 利用face_recognition读取图片
            img=face_recognition.load_image_file(im)
            # 利用face_recognition自动检测出人脸，返回像素坐标点
            img_locs=face_recognition.face_locations(img)
            # 裁剪出人脸图片，并且生成标签类别，114个人分别对应0-113，将图片转为200*200*3格式
            for img_loc in img_locs:
                top,right,bottom,left=img_loc
                # 裁剪出人脸图片
                img_face=img[top:bottom,left:right]
                # 将图片转为200*200*3格式的彩色图片
                img_face=transform.resize(img_face,(200,200,3))
                imgs.append(img_face)
                # 生成标签
                labels.append(idx)
    return np.asarray(imgs,np.float32),np.asarray(labels,np.int32)
##生成图片数据和标签
data,label=read_img(path)
##划分训练集测试集，训练集占比80%
data_train,data_test,label_train,label_test=train_test_split(data,label,random_state=12,train_size=0.75)
        
    

###搭建CNN网络
##设置CNN网络全局变量
batch_size = 256  # 批处理样本数量
nb_classes = 114  # 分类数目
epochs = 40  # 迭代次数
img_rows, img_cols = 200,200 #输入图片样本的宽高
nb_filters = 32  # 卷积核的个数
pool_size = (2, 2)  # 池化层的大小
kernel_size = (3, 3)  # 卷积核的大小
input_shape = (img_rows,img_cols,3)  # 输入图片的维度
##将标签转为one-hot格式，便于CNN输入
label_train = tf.keras.utils.to_categorical(label_train, nb_classes)
label_test = tf.keras.utils.to_categorical(label_test, nb_classes)

##搭建CNN网络
model = Sequential()
model.add(Conv2D(nb_filters,kernel_size,input_shape=input_shape,strides=1,padding='same',activation='relu'))  # 卷积层1
model.add(MaxPooling2D(pool_size=pool_size,strides=2,padding='same'))  # 池化层1
model.add(Conv2D(64,kernel_size,strides=1,padding='same',activation='relu'))  # 卷积层2
model.add(MaxPooling2D(pool_size=pool_size,strides=2,padding='same'))  # 池化层2
model.add(Conv2D(128,kernel_size,strides=1,padding='same',activation='relu'))  #  卷积层3
model.add(Dropout(0.4)) # dropout层，比例为40%
model.add(MaxPooling2D(pool_size=pool_size,strides=2,padding='same')) # 池化层3
model.add(Conv2D(256,kernel_size,strides=1,padding='same',activation='relu')) # 卷积层4
model.add(Dropout(0.4)) # dropout层，比例为40%
model.add(MaxPooling2D(pool_size=pool_size,strides=2,padding='same')) #  池化层4
model.add(Dropout(0.4)) # dropout层，比例为40%
model.add(Flatten())  # 拉成一维数据
model.add(Dense(512)) # 全连接层1
model.add(Dropout(0.4)) # dropout防止过拟合
model.add(Activation('relu')) # relu激活函数
model.add(Dense(nb_classes,activation='softmax'))  # 全连接层2，输出层采用softmax激活函数
EarlyStopping(monitor='val_acc',patience=5) # 设置提前终止，观察val_acc，上限轮次为5
model.summary() # 打印出模型结构
# 编译模型
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# 训练模型
model.fit(data_train, label_train, batch_size=batch_size, epochs=epochs,verbose=1, validation_data=(data_test,label_test))
# 评估模型
score = model.evaluate(data_test, label_test, verbose=0)
##打印出测试集准确率，准确率为79.66%
print('Test accuracy:', score[1])
##保存模型到本地
model.save('/Users/genggejianyi/Desktop/大数据班/python作业/FaceDB/人脸识别.model')
##画出模型训练过程中val_acc随epochs的变化图
epochs=list(range(1,41))
val_acc=[0.0023,0.0068,0.0307,0.0625,0.0920,0.1159,0.2239,0.2023,0.3386,0.4716,0.5080,0.5489,0.5818,0.6386,0.6682,0.6795,
         0.6557,0.7205,0.7273,0.7125,0.7091,0.6625,0.7432,0.7284,0.7500,0.7705,0.7625,0.7648,0.7602,0.7716,0.7966,0.7909,
         0.7773,0.7966,0.7932,0.7852,0.7761,0.7920,0.7841,0.7739]
plt.plot(epochs,val_acc)
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.title('val_acc')
plt.show()



###SVM人脸识别
##先对数据生成主成分，主成分个数为50，并将数据转为array格式
pca=PCA(n_components=50,svd_solver='auto',whiten=True).fit(data.reshape(3519,-1))
data_pca=pca.transform(data.reshape(3519,-1))
X=np.array(data_pca)
Y=np.array(label)
##定义函数SVM，采用10折交叉验证划分训练集测试集，输入为核函数以及gamma值，输出为测试集平均准确率
def SVM(kernel_name,param):
    kf=KFold(n_splits=10,shuffle=True)
    precision_avg=0.0
    # 采用gridsearch方法遍历寻找最优的C值，C值范围为0.1，1，10，100，1000
    param_grid = {'C': [0.1,1,10,100,1e3]}
    clf=GridSearchCV(svm.SVC(kernel=kernel_name,class_weight='balanced',gamma=param),param_grid,cv=5,iid=True)
    # 10折交叉验证训练数据并且计算测试集上的平均准确率
    for train,test in kf.split(X):
        clf=clf.fit(X[train],Y[train])
        test_pred=clf.predict(X[test])
        precision=0
        for i in range(0,len(Y[test])):
            if (Y[test][i]==test_pred[i]):
                precision=precision+1
        precision_avg=precision_avg+float(precision)/len(Y[test])
    precision_avg=precision_avg/10
    return precision_avg
##尝试使用rbf kernel以及gamma=0.001，预测准确率为52.46%
SVM('rbf',0.001)

##定义不同kernel：rbf,poly,sigmoid
##gamma取值0.0001-0.01之间100个数，画出不同kernel下预测准确率随gamma值变化图
kernel_to_test=['rbf','poly','sigmoid']
plt.figure(1)
for kernel_name in kernel_to_test:
    # 在0.0001-0.01间取100个gamma值
    x_label=np.linspace(0.0001,0.01,5)
    y_label=[]
    for i in x_label:
        y_label.append(SVM(kernel_name,i))
    plt.plot(x_label,y_label,label=kernel_name)
plt.xlabel('Gamma')
plt.ylabel('precision')
plt.title('diff kernel results')
plt.legend()
plt.show()

        


###KNN人脸识别
##划分测试集训练集，训练集占比80%
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=123,train_size=0.8)
##计算不同n_neighbors时模型测试集准确率
n_nb=[]
ACC=[]
for i in range(1,47):
    # 建立KNN模型
    knn_clf=neighbors.KNeighborsClassifier(i,algorithm='ball_tree',weights='distance')
    knn_clf.fit(X_train,Y_train)
    # 计算测试集准确率
    acc=sum(map(int,knn_clf.predict(X_test)==Y_test))/len(Y_test)
    n_nb.append(i)
    ACC.append(acc)
    # 画出不同n_neighbors（取1-46）情况下测试集准确率变化图
    plt.plot(n_nb,ACC)
plt.xlabel('n_neighbors')
plt.ylabel('ACC')
plt.title('diff n_neighbors acc')
plt.show()






###logistic regression人脸识别
##10折交叉验证寻找最优参数C
lr=linear_model.LogisticRegressionCV(cv=10,solver='liblinear',class_weight='balanced')
lr.fit(X_train,Y_train)
##打印出训练集准确率与测试集准确率
print('logistic regression train acc:',\
      metrics.accuracy_score(Y_train,lr.predict(X_train)))
print('logistic regression test acc:',\
      metrics.accuracy_score(Y_test,lr.predict(X_test)))

