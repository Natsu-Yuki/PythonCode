from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten


import keras.optimizers as optimizers
import numpy as np
np.random.seed(0)
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
x_train, x_test, y_train, y_test = train_test_split(
    people.images, people.target, random_state=0)

#plt.imshow(x_train[0,:,:])
#plt.title(people.target_names[0])
#plt.show()

x_train_re=x_train.reshape(-1, 87, 65,1)
x_test_re=x_test.reshape(-1, 87, 65,1)

x_train_re=x_train_re[:,0:84,0:64,:]
x_test_re=x_test_re[:,0:84,0:64,:]

x_train_nor=x_train_re.astype(np.float32)/255
x_test_nor=x_test_re.astype(np.float32)/255

y_train_onehot=np_utils.to_categorical(y_train)
y_test_onehot=np_utils.to_categorical(y_test)

print('x_train_nor.shape:{},y_train_onehot.shape:{}'.format(x_train_nor.shape,y_train_onehot.shape))


print(x_train_nor[0])
print(y_train_onehot[0])
model=Sequential()

model.add(
    Conv2D(filters=16,
           kernel_size=(5,5),
           padding='same',
           input_shape=(84,64,1),
           activation='relu'
    )
)
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=32,
                 kernel_size=(5,5),
                 padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.1))
model.add(Flatten())

model.add(Dense(units=248,activation='relu'))
model.add(Dropout(0.1))


model.add(Dense(units=62,activation='softmax'))
model.add(Dropout(0.1))
print(model.summary())



model.compile(loss='categorical_crossentropy',
              optimizer='adam',metrics=['accuracy'])
train_history=model.fit(
    x=x_train_nor,y=y_train_onehot,validation_split=0.2,epochs=10,batch_size=200,verbose=2
)





