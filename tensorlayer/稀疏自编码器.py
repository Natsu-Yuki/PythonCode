import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import numpy as np
np.random.seed(0)

x_train,y_train,x_test,y_test,x_val,y_val=tl.files.load_mnist_dataset(
    shape=(-1,784)
)

learning_rate=0.0001
lambda_l2_w=0.01
n_epochs=20
batch_size=128
print_interval=784

hidden_size=196
input_size=784
image_width=28

model='sigmoid'

x=tf.placeholder(dtype=tf.float32,shape=[None,784],name='x')

print('Build Network')

if model=='relu':
    network=InputLayer(inputs=x,name='input')
    network=DenseLayer(network,n_units=hidden_size,act=tf.nn.relu,name='relu1')
    encoded_img=network.outputs
    recon_layer1=DenseLayer(network,n_units=input_size,act=tf.nn.softplus,name='recon_layer1')
elif model=='sigmoid':
    network=InputLayer(x,name='input')
    network=DenseLayer(network,n_units=hidden_size,act=tf.nn.sigmoid,name='sigmoid1')
    encoded_img=network.outputs
    recon_layer1=DenseLayer(network,n_units=input_size,act=tf.nn.sigmoid,name='recon_layer1')


y=recon_layer1.outputs
train_params=recon_layer1.all_params[-4:]

mse=tf.reduce_sum(tf.squared_difference(y,x),1)
mse=tf.reduce_mean(mse)

l2_w=tf.contrib.layers.l2_regularizer(lambda_l2_w)(train_params[0])+\
     tf.contrib.layers.l2_regularizer(lambda_l2_w)(train_params[2])

activation_out=recon_layer1.all_layers[-2]
l1_a=0.001*tf.reduce_mean(activation_out)

beta=5
rho=0.15
p_hat=tf.reduce_mean(activation_out,0)

kld=beta*tf.reduce_sum(
    rho*tf.log(tf.divide(rho,p_hat))+(1-rho)*tf.log((1-rho)/(tf.subtract(float(1),p_hat)))
)

if model=='sigmoid':
    cost=mse+l2_w+kld
elif model=='relu':
    cost=mse+l2_w+l1_a


train_op=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
saver=tf.train.Saver()

total_batch=2500

with tf.Session() as sess:
    initialize_global_variables(sess)
    for epoch in range(n_epochs):
         avg_cost=0.0
         for i in range(total_batch):
             batch_x,batch_y=x_train[i*batch_size:(i+1)*batch_size],y_train[i*batch_size:(i+1)*batch_size]
             batch_x=np.array(batch_x).astype(np.float32)
             batch_cost,_=sess.run([cost,train_op],feed_dict={x:batch_x})
             avg_cost+=batch_cost
             if not i%print_interval:
                 print('minibatch:{},cost:{}'.format(i+1,batch_cost))

         print('epoch:{},avg_cost:{}\n'.format(epoch + 1, avg_cost / (i + 1)))
    saver.save(sess,save_path=r'C:\Users\Natsu\Desktop\f\autoencoder.ckpt')


import matplotlib.pyplot as plt

n_image=15
fig,axes=plt.subplots(nrows=2,ncols=n_image,sharex=True,sharey=True,figsize=(20,2.5))
test_image=x_test[:n_image]

with tf.Session() as sess:
    saver.restore(sess,save_path=r'C:\Users\Natsu\Desktop\f\autoencoder.ckpt')
    decode=sess.run(recon_layer1.outputs,feed_dict={x:test_image})
    if model=='relu':
        weights=sess.run(tl.layers.get_variables_with_name('relu1/w:0',False,True))
    elif model=='sigmoid':
        weights=sess.run(tl.layers.get_variables_with_name('sigmoid1/w:0',False,True))
    recon_weights=sess.run(tl.layers.get_variables_with_name('recon_layer1/w:0',False,True))
    recon_bias=sess.run(tl.layers.get_variables_with_name('recon_layer1/b:0',False,True))
    for i in range(n_image):
        for ax,img in zip(axes,[test_image,decode]):
            ax[i].imshow(img[i].reshape((image_width,image_width)),cmap='binary')
    plt.show()
    

