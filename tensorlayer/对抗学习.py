import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import numpy as np
np.random.seed(0)

def generator(inputs,is_train=True,reuse=False):
    s=64
    s2,s4,s8,s16=32,16,8,4

    batch_size=64

    w_init=tf.random_normal_initializer(stddev=0.02)
    gamma_init=tf.random_normal_initializer(mean=1.,stddev=0.02)
    with tf.variable_scope(name_or_scope='generator',reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        #   输入层，输入为随机潜在变量
        net_in=InputLayer(inputs=inputs,name='g/in')

        net_h0=DenseLayer(net_in,n_units=64*8*4*4,W_init=w_init,act=tf.identity,name='g/h0/lin')
        net_h0=ReshapeLayer(net_h0,shape=[-1,4,4,64*8],name='g/h0/res')

        net_h0=BatchNormLayer(net_h0,act=tf.nn.relu,is_train=is_train,gamma_init=gamma_init,name='g/h0/bn')

        #   二维反卷积层 步长为(2,2) 过滤矩阵为(5,5) 输出尺寸为1/8
        net_h1=DeConv2d(net_h0,64*4,(5,5),(8,8),(2,2),padding='SAME',batch_size=batch_size,W_init=w_init,name='g/h1/de2d')
        net_h1=BatchNormLayer(net_h1,act=tf.nn.relu,is_train=is_train,gamma_init=gamma_init,name='g/h1/bn')

        #   二维反卷积层 1/4
        net_h2=DeConv2d(net_h1,64*2,(5,5),(16,16),(2,2),padding='SAME',batch_size=batch_size,name='g/h2/de2d')
        net_h2=BatchNormLayer(net_h2,act=tf.nn.relu,is_train=is_train,gamma_init=gamma_init,name='g/h2/bn')

        #   二维反卷积层 1/2
        net_h3 = DeConv2d(net_h2, 64 , (5, 5), (32, 32), (2, 2), padding='SAME', batch_size=batch_size,
                          name='g/h3/de2d')
        net_h3 = BatchNormLayer(net_h3, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='g/h3/bn')

        #   反卷积层
        net_h4=DeConv2d(net_h3,3,(5,5),(64,64),(2,2),padding='SAME', batch_size=batch_size,
                          name='g/h4/de2d')
        logits=net_h4.outputs

        net_h4.outputs=tf.nn.tanh(net_h4.outputs)

        return net_h4,logits


def discriminator(inputs,is_train=True,reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(mean=1., stddev=0.02)
    lrelu=lambda x:tl.act.lrelu(x,0.2)
    with tf.variable_scope(name_or_scope='discriminator',reuse=reuse):
        net_in=InputLayer(inputs=inputs,name='d/in')
        net_ho=Conv2d(
            net_in,n_filter=64,filter_size=(5,5),strides=(2,2),padding='SAME',act=lrelu,W_init=w_init,name='d/h0/con2d'
        )



        net_h1=Conv2d(
            net_ho,n_filter=64*2,filter_size=(5,5),strides=(2,2),padding='SAME',act=None,W_init=w_init,name='d/h1/con2d'

        )

        net_h1=BatchNormLayer(net_h1,act=lrelu,is_train=is_train,gamma_init=gamma_init,name='d/h1/bn')



        net_h2 = Conv2d(
            net_h1, n_filter=64 * 4, filter_size=(5, 5), strides=(2, 2), padding='SAME', act=None, W_init=w_init,
            name='d/h2/con2d'

        )

        net_h2 = BatchNormLayer(net_h2, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='d/h2/bn')



        net_h3 = Conv2d(
            net_h2, n_filter=64 * 8, filter_size=(5, 5), strides=(2, 2), padding='SAME', act=None, W_init=w_init,
            name='d/h3/con2d'

        )

        net_h3 = BatchNormLayer(net_h3, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='d/h3/bn')


        net_h4=FlattenLayer(net_h3,name='d/h4/flatten')

        net_h4=DenseLayer(net_h4,n_units=1,act=tf.identity,W_init=w_init,name='d/h4/lin_sigmoid')

        logits=net_h4.outputs

        net_h4.outputs=tf.nn.sigmoid(net_h4.outputs)

        return net_h4,logits

z=tf.placeholder(dtype=tf.float32,shape=[64,100],name='z_noise')

real_images=tf.placeholder(dtype=tf.float32,shape=[64,64,64,3],name='real_images')
net_g,g_logits=generator(z,is_train=True,reuse=False)
net_g2,g2_logits=generator(z,is_train=False,reuse=True)

net_d,d_logits=discriminator(net_g.outputs,is_train=True,reuse=False)
net_d2,d2_logits=discriminator(real_images,is_train=True,reuse=True)

d_loss_real=tl.cost.sigmoid_cross_entropy(d2_logits,tf.ones_like(d2_logits),name='dreal')

d_loss_fake=tl.cost.sigmoid_cross_entropy(d_logits,tf.zeros_like(d_logits),name='dfake')

d_loss=d_loss_fake+d_loss_real

g_loss=tl.cost.sigmoid_cross_entropy(d_logits,tf.ones_like(d_logits),name='gfake')

d_optim=tf.train.AdamOptimizer(learning_rate=0.0002,beta1=0.5).minimize(d_loss)
g_optim=tf.train.AdamOptimizer(learning_rate=0.0002,beta1=0.5).minimize(g_loss)











