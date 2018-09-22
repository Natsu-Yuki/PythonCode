import time
t=time.clock()
import numpy as np
from keras.callbacks import TensorBoard
from keras.utils import np_utils
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


np.random.seed(0)

(x_img_train,y_label_train),(x_img_test, y_label_test)=mnist.load_data()


def plot_images_labels_prediction(images, labels, prediction,
                                  idx, num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num > 25: num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1 + i)
        ax.imshow(images[idx], cmap='binary')

        title = 'label='+str(labels[idx])
        if len(prediction) > 0:
            title += ',prediction='+str(prediction[idx])

        ax.set_title(title, fontsize=10)
        ax.set_xticks([]);
        ax.set_yticks([])
        idx += 1
    plt.show()

#   plot_images_labels_prediction(x_img_test,y_label_test,[],0,25)

def plot_image(image):
    plt.imshow(image,cmap='binary')
    plt.show()


x_train=x_img_train.reshape(60000,784).astype('float32')
x_test=x_img_test.reshape(10000,784).astype('float32')

x_train_normal=x_train/255
x_test_normal=x_test/255

y_train_onehot=np_utils.to_categorical(y_label_train)
y_test_onehot=np_utils.to_categorical(y_label_test)


model=Sequential()
model.add(Dense(units=256,input_dim=784,kernel_initializer='normal',activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=10,kernel_initializer='normal',activation='softmax'))
model.add(Dropout(0.5))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
train_history=model.fit(
    x=x_train_normal,y=y_train_onehot,validation_split=0.2,epochs=10,batch_size=200,verbose=2,
    callbacks=[TensorBoard(log_dir=r'C:\Users\Natsu\Desktop\f')]

)
print(model.summary())

def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

#   show_train_history(train_history,'acc','val_acc')
#   show_train_history(train_history,'loss','val_loss')


scores = model.evaluate(x_test_normal, y_test_onehot)
print()
print('accuracy=',scores[1])


prediction=model.predict_classes(x_test)
print(prediction)
#plot_images_labels_prediction(x_img_test,y_label_test, prediction,idx=340)





print('********************')
print ('exe time：{}s'.format(time.clock()-t))
#from sklearn.linear_model import Ridge
#ridge=Ridge()
#ridge.fit(x_img_train,y_label_train)
#print(ridge.score(x_img_train,y_label_train))
#print(ridge.score(x_img_test,y_label_test))