import tensorflow as tf
import random
random.seed(10)

layer1_node=500

batch_size=100
learning_rate_base=0.8
learning_rate_decay=0.99
regularization_rate=0.0001
training_steps=30000
moving_average_decay=0.99



def get_weight_variable(shape,regularizer):
    weights=tf.get_variable(name='weight',shape=shape,initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer!=None:
        tf.add_to_collection('losses',regularizer(weights))
    return weights

input_node=784
output_node=10

image_size=28
num_channels=1
num_labels=10

conv1_deep=32
conv1_size=5

conv2_deep=64
conv2_size=5

fc_size=512

def inference(input_tensor,train,regularizer):
    with tf.variable_scope('layers_conv1'):
        conv1_weights=tf.get_variable(
            'weight',[conv1_size,conv1_size,num_channels,conv1_deep],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        conv1_biases=tf.get_variable(
            'bias',[conv1_deep],initializer=tf.constant_initializer(0.0)
        )
        conv1=tf.nn.conv2d(
            input_tensor,conv1_weights,strides=[1,1,1,1],padding='SAME'
        )
        relu1=tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))

    with tf.name_scope('layer2_pool1'):
        pool1=tf.nn.max_pool(
            relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME'
        )
    with tf.variable_scope('layer3_conv2'):
        conv2_weights=tf.get_variable(
            'weight',[conv2_size,conv2_size,conv1_deep,conv2_deep],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        conv2_biases=tf.get_variable(
            'bias',[conv2_deep],initializer=tf.constant_initializer(0.0)
        )
        conv2=tf.nn.conv2d(
            pool1,conv2_weights,strides=[1,1,1,1],padding='SAME'
        )
        relu2=tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))
    with tf.name_scope('layer4_pools'):
        pool2=tf.nn.max_pool(
            relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME'
        )
    pool_shape=pool2.get_shape().as_list()
    nodes=pool_shape[1]*pool_shape[2]*pool_shape[3]
    reshaped=tf.reshape(pool2,(-1,nodes))
    with tf.variable_scope('layer5_fc1'):
        fc1_weights=tf.get_variable(
            'weight',[nodes,fc_size],initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        if regularizer!=None:
            tf.add_to_collection('losses',regularizer(fc1_weights))
        fc1_biases=tf.get_variable(
            'bias',[fc_size],initializer=tf.constant_initializer(0.0)
        )
        fc1=tf.nn.relu(tf.matmul(reshaped,fc1_weights)+fc1_biases)
        if train:fc1=tf.nn.dropout(fc1,0.5)
    with tf.variable_scope('layer6_fc2'):
        fc2_weights=tf.get_variable(
            'weight',[fc_size,num_labels],initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        if regularizer!=None:
            tf.add_to_collection('losses',regularizer(fc2_weights))
        fc2_biases=tf.get_variable(
            'bias',[num_labels],initializer=tf.constant_initializer(0.0)
        )
        logit=tf.matmul(fc1,fc2_weights)+fc2_biases
    return logit

import os
import tensorflow.examples.tutorials.mnist.input_data as input_data

def train(mnist):
    x=tf.placeholder(tf.float32,[None,784],name='input')
    y_=tf.placeholder(tf.float32,[None,output_node],name='output')

    x_reshape=tf.reshape(x,(-1,28,28,1))

    regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)
    y = inference(x_reshape,train=True ,regularizer=regularizer)

    global_step = tf.Variable(0, trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(
        moving_average_decay, global_step
    )

    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_, 1)
    )

    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    with tf.name_scope('Loss'):
        loss=cross_entropy_mean+tf.add_n(tf.get_collection('losses'))
        tf.summary.scalar('loss', loss)

    learning_rate = tf.train.exponential_decay(
        learning_rate_base, global_step, 55000 / batch_size, learning_rate_decay
    )

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')


    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

    with tf.name_scope('Accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar(' accuracy', accuracy)


    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        for i in range(3001):
            xs, ys = mnist.train.next_batch(batch_size)

            _,loss_value,step=sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})
            if i % 100 == 0:
                accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                print('After {} steps,loss value is {},accuracy is {}'.format(step, loss_value, accuracy_score))



mnist = input_data.read_data_sets(r"C:\Users\Natsu\Desktop\MacroMap\TENSORV\data", one_hot=True)
train(mnist)







