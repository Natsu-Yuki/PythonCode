import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import numpy as np
np.random.seed(0)


x_train,y_train,x_test,y_test,x_val,y_val=tl.files.load_mnist_dataset(
    shape=(-1,784)
)
model='sigmoid'
sess=tf.InteractiveSession()

x=tf.placeholder(dtype=tf.float32,shape=[None,784],name='x')
y_=tf.placeholder(dtype=tf.int64,shape=[None,],name='y_')

network = InputLayer(x, name='input')
network=DropoutLayer(network,name='denoising1',keep=0.5)
network = DenseLayer(network, n_units=196, act=tf.nn.sigmoid, name='sigmoid1')
encoded_img=network.outputs
recon_layer1=ReconLayer(network,x_recon=x,n_units=784,act=tf.nn.sigmoid,name='recon_later1')

tl.layers.initialize_global_variables(sess)

recon_layer1.pretrain(
    sess=sess,x=x,X_train=x_train,X_val=x_val,
    denoise_name='denoising1',n_epoch=200,
    batch_size=128,print_freq=10,save=True,
    save_name='wlpre'
)

saver=tf.train.Saver()
save_path=saver.save(sess,save_path=r'C:\Users\Natsu\Desktop\f\model.ckpt')
sess.close()


n_image=15
test_image=x_test[:n_image]

with tf.Session() as sess:
    saver.restore(sess,save_path=r'C:\Users\Natsu\Desktop\f\model.ckpt')
    decode=sess.run(recon_layer1.outputs,feed_dict={x:test_image})
    if model=='relu':
        weights=sess.run(tl.layers.get_variables_with_name('relu1/w:0',False,True))
    elif model=='sigmoid':
        weights=sess.run(tl.layers.get_variables_with_name('sigmoid1/w:0',False,True))
    recon_weights=sess.run(tl.layers.get_variables_with_name('recon_layer1/w:0',False,True))
    recon_bias=sess.run(tl.layers.get_variables_with_name('recon_layer1/b:0',False,True))


sample1=test_image[1].reshape([1,784])
sample2=test_image[12].reshape([1,784])

with tf.Session() as sess:
    saver.restore(sess,save_path=r'C:\Users\Natsu\Desktop\f\model.ckpt')
    dp_dict=tl.utils.dict_to_one(recon_layer1.all_drop)
    feed_dict={x:sample1}
    feed_dict.update(dp_dict)
    encode1=sess.run(encoded_img,feed_dict=feed_dict)

    feed_dict={x:sample2}
    feed_dict.update(dp_dict)
    encode2=sess.run(encoded_img,feed_dict=feed_dict)

encode=tf.placeholder(dtype=tf.float32,shape=[None,196],name='encode')
recon_weights,recon_bias=recon_weights[0],recon_bias[0]

test_network=tl.layers.InputLayer(encode,name='test')
test_recon_layer1=tl.layers.DenseLayer(test_network,784,act=tf.nn.sigmoid,name='test_recon_layer1')

diff=encode1-encode2
num_inter=10
delta=diff/num_inter
encoded_all=encode1

for i in range(1,num_inter+1):
    encoded_all=np.vstack(encoded_all,encode1+delta*i)
with tf.Session() as sess:
    decoded_all=sess.run(test_recon_layer1.outputs,feed_dict={encode:encoded_all})
import matplotlib.pyplot as plt
fig,axes=plt.subplots(nrows=1,ncols=num_inter+1,sharex=True,sharey=True,figsize=(15,1.5))

for i in range(num_inter+1):
    axes[i].imshow(decoded_all[i].reshape((28,28)),cmap='binary')

plt.show()