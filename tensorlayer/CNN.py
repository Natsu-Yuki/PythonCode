import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import numpy as np
np.random.seed(0)

import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets(r"C:\Users\Natsu\Desktop\MacroMap\TENSORV\data", one_hot=False)

x_train=mnist.train.images.reshape(-1,28,28,1)
x_test=mnist.test.images.reshape(-1,28,28,1)
x_val=mnist.validation.images.reshape(-1,28,28,1)

y_train=mnist.train.labels
y_test=mnist.test.labels
y_val=mnist.validation.labels

sess=tf.InteractiveSession()
batch_size=128
x=tf.placeholder(dtype=tf.float32,shape=[batch_size,28,28,1])
y_=tf.placeholder(dtype=tf.int64,shape=[batch_size,])


network=InputLayer(x,name='input')
network=Conv2d(
    network,n_filter=32,filter_size=(5,5),strides=(1,1),act=tf.nn.relu,padding='SAME',name='cnn1'
)
network=MaxPool2d(
    network,filter_size=(2,2),strides=(2,2),padding='SAME',name='pool1'
)
network=Conv2d(
    network,n_filter=64,filter_size=(5,5),strides=(1,1),act=tf.nn.relu,padding='SAME',name='cnn2'
)
network=MaxPool2d(
    network,filter_size=(2,2),strides=(2,2),padding='SAME',name='pool2'
)
network=FlattenLayer(
    network,name='flatten'
)
network=DropoutLayer(
    network,keep=0.5,name='drop1'
)
network=DenseLayer(
    network,n_units=256,act=tf.nn.relu,name='relu1'
)
network=DropoutLayer(network,keep=0.5)
network=DenseLayer(
    network,n_units=10,act=tf.identity,name='output'
)


y=network.outputs
cost=tl.cost.cross_entropy(y,y_,name='cost')
cor=tf.equal(tf.argmax(y,1),y_)
acc=tf.reduce_mean(tf.cast(cor,tf.float32))

train_params=network.all_params
train_op=tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost,var_list=train_params)
tl.layers.initialize_global_variables(sess=sess)

network.print_params()
network.print_layers()

tl.utils.fit(
    sess=sess,network=network,train_op=train_op,cost=cost,acc=acc,
    X_train=x_train,y_train=y_train,x=x,y_=y_,X_val=x_val,y_val=y_val,
    n_epoch=200,batch_size=batch_size,eval_train=False,print_freq=5
)

tl.utils.test(
    sess=sess,network=network,X_test=x_test,y_test=y_test,x=x,y_=y,cost=cost,acc=acc,batch_size=None
)