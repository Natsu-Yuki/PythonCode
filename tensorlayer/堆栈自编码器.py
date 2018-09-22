import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import numpy as np
np.random.seed(0)

x_train,y_train,x_test,y_test,x_val,y_val=tl.files.load_mnist_dataset(
    shape=(-1,784)
)

x_train=x_train[0:5000,:]
y_train=y_train[0:5000]

model='sigmoid'

sess=tf.InteractiveSession()

if model=='relu':
    act=tf.nn.relu
    act_recon=tf.nn.softplus
elif model=='sigmoid':
    act=tf.nn.sigmoid
    act_recon=tf.nn.sigmoid

x=tf.placeholder(dtype=tf.float32,shape=[None,784],name='x')
y_=tf.placeholder(dtype=tf.int64,shape=[None,],name='y_')

network=InputLayer(x,name='input')

#   降噪层
network=DropoutLayer(network,keep=0.5,name='denoising1')

network=DropoutLayer(network,keep=0.8,name='drop1')

#   第一个降噪自编码器
network=DenseLayer(network,n_units=800,act=act,name='dense1')
x_recon1=network.outputs
recon_layer1=ReconLayer(network,x_recon=x,n_units=784,act=act_recon,name='recon_layer1')

#   第二个降噪自编码器
network=DropoutLayer(network,keep=0.5,name='drop2')
network=DenseLayer(network,n_units=800,act=act,name='dense2')
recon_layer2=ReconLayer(network,x_recon=x_recon1,n_units=800,act=act_recon,name='recon_layer2')

network=DenseLayer(network,n_units=10,act=tf.identity,name='output')

y=network.outputs
y_op=tf.argmax(tf.nn.softmax(y),axis=1)

cost=tl.cost.cross_entropy(y,y_,name='cost')

train_params=network.all_params
train_op=tf.train.AdamOptimizer(0.0001).minimize(cost,var_list=train_params)

tl.layers.initialize_global_variables(sess)

recon_layer1.pretrain(
    sess=sess,x=x,X_train=x_train,X_val=x_val,denoise_name='denoising1',
    n_epoch=100,batch_size=128,print_freq=10,save=True,save_name='wlpre'
)
recon_layer2.pretrain(
    sess=sess, x=x, X_train=x_train, X_val=x_val, denoise_name='denoising1',
    n_epoch=100, batch_size=128, print_freq=10, save=False
)
network.print_params()
network.print_layers()

cor=tf.equal(tf.argmax(y,1),y_)
acc=tf.reduce_mean(tf.cast(cor,tf.float32))

train_acc_list=[]
val_acc_list=[]

epoch_list=[]

for epoch in range(20):
    for x_train_a,y_train_a in tl.iterate.minibatches(x_train,y_train,batch_size=250,shuffle=True):
        feed_dict={x:x_train_a,y_:y_train_a}
        feed_dict.update(network.all_drop)
        #   feed_dict[set_keep['denoising1']]=1
        sess.run(train_op,feed_dict=feed_dict)

        train_loss,train_acc,n_batch=0,0,0
        for x_train_a, y_train_a in tl.iterate.minibatches(x_train, y_train, batch_size=250, shuffle=True):
            dp_dict=tl.utils.dict_to_one(network.all_drop)
            feed_dict = {x: x_train_a, y_: y_train_a}
            feed_dict.update(dp_dict)
            err,ac=sess.run([cost,acc],feed_dict=feed_dict)
            train_loss+=err
            train_acc+=ac
            n_batch+=1
        print('train loss:{}'.format(train_loss/n_batch))
        print('train acc:{}'.format(train_acc/n_batch))
        train_acc_list.append(train_acc/n_batch)
        val_loss, val_acc, n_batch = 0, 0, 0
        for x_val_a, y_val_a in tl.iterate.minibatches(x_val, y_val, batch_size=250, shuffle=True):
            dp_dict = tl.utils.dict_to_one(network.all_drop)
            feed_dict = {x: x_val_a, y_: y_val_a}
            feed_dict.update(dp_dict)
            err, ac = sess.run([cost, acc], feed_dict=feed_dict)
            val_loss += err
            val_acc += ac
            n_batch += 1
        print('val loss:{}'.format(val_loss / n_batch))
        print('val acc:{}'.format(val_acc / n_batch))
        val_acc_list.append(val_acc/n_batch)
        epoch_list.append(20*epoch)
        tl.visualize.W(
            network.all_params[0].eval(),second=10,saveable=True,shape=[28,28],name='w1_'+str(epoch+1),fig_idx=2012
        )

import matplotlib.pyplot as plt
p=plt.figure()
plt.plot(epoch_list,train_acc_list,label='tran_acc')
plt.plot(epoch_list,val_acc_list,label='val_acc')
plt.legend()
plt.show()
