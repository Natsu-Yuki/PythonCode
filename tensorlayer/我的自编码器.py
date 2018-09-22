import numpy as np

np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
import matplotlib.pyplot as plt
import tensorlayer as tl
import tensorflow as tf
from tensorlayer.layers import *
# X shape (60,000 28x28), y shape (10,000, )
(x_train, _), (x_test, y_test) = mnist.load_data()

# 数据预处理
x_train = x_train.astype('float32') / 255. - 0.5  # minmax_normalized
x_test = x_test.astype('float32') / 255. - 0.5  # minmax_normalized
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))

x_train=x_train[0:10000,:]
# 压缩特征维度至2维
encoding_dim = 2

print(x_train.shape)
print(x_test.shape)

sess=tf.InteractiveSession()

x=tf.placeholder(dtype=tf.float32,shape=[None,784],name='x')
y_=tf.placeholder(dtype=tf.float32,shape=[None,784],name='x')

input=InputLayer(x,name='x')

encoder=DenseLayer(input,n_units=128,name='encoder1',act=tf.nn.relu)
encoder=DenseLayer(encoder,n_units=64,name='encoder2',act=tf.nn.relu)
encoder=DenseLayer(encoder,n_units=10,name='encoder3',act=tf.nn.relu)
encoder=DenseLayer(encoder,n_units=2,name='encoder4')

decoder=DenseLayer(encoder,n_units=10,name='decoder1',act=tf.nn.relu)
decoder=DenseLayer(decoder,n_units=64,name='decoder2',act=tf.nn.relu)
decoder=DenseLayer(decoder,n_units=128,name='decoder3',act=tf.nn.relu)
decoder=DenseLayer(decoder,n_units=784,name='decoder4',act=tf.nn.tanh)

y=decoder.outputs

mse=tf.reduce_sum(tf.squared_difference(x,y),axis=1)
mse=tf.reduce_mean(mse)

train_params=decoder.all_params

train_op=tf.train.AdamOptimizer(0.001).minimize(mse,var_list=train_params)


tl.layers.initialize_global_variables(sess=sess)

decoder.print_params()
decoder.print_layers()

tl.utils.fit(
    sess=sess,train_op=train_op,network=decoder,cost=mse,
    X_train=x_train,y_train=x_train,x=x,y_=y_,
    n_epoch=10,batch_size=1000,print_freq=5
)

encoded_imgs=sess.run(encoder.outputs,feed_dict={x:x_test,y_:x_test})
plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test, s=3)
plt.colorbar()
plt.show()