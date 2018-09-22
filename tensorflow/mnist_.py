import tensorflow as tf
import random
random.seed(10)
input_node=784
output_node=10
layer1_node=500

batch_size=100
learning_rate_base=0.8
learning_rate_decay=0.99
regularization_rate=0.0001
training_steps=30000
moving_average_decay=0.99

model_save_path=r'C:\Users\Natsu\Desktop\MacroMap\TENSORV'
model_name='model.ckpt'

def get_weight_variable(shape,regularizer):
    weights=tf.get_variable(name='weight',shape=shape,initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer!=None:
        tf.add_to_collection('losses',regularizer(weights))
    return weights

def inference(input_tensor,regularizer):
    with tf.variable_scope(name_or_scope='layer1'):
        with tf.name_scope('Weights1'):
            weights=get_weight_variable([input_node,layer1_node],regularizer)
            tf.summary.histogram('weights1',weights)
        with tf.name_scope('Biases1'):
            biases=tf.get_variable(name='biases',shape=[layer1_node],initializer=tf.constant_initializer(0.0))
            tf.summary.histogram('biases1',biases)
        layer1=tf.nn.relu(tf.matmul(input_tensor,weights)+biases)
    with tf.variable_scope(name_or_scope='layer2'):
        with tf.name_scope('Weights2'):
            weights=get_weight_variable([layer1_node,output_node],regularizer)
            tf.summary.histogram('weights2', weights)
        with tf.name_scope('Biases2'):
            biases=tf.get_variable(name='biases',shape=[output_node],initializer=tf.constant_initializer(0.0))
            tf.summary.histogram('biases2', biases)
        layer2=tf.matmul(layer1,weights)+biases
    return layer2


import os
import tensorflow.examples.tutorials.mnist.input_data as input_data

def train(mnist):
    x=tf.placeholder(tf.float32,[None,input_node],name='input')
    y_=tf.placeholder(tf.float32,[None,output_node],name='output')

    regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)
    y = inference(x, regularizer)

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

    validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

    saver=tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("C:\\Users\\Natsu\\Desktop\\a", sess.graph)
        for i in range(3001):
            xs, ys = mnist.train.next_batch(batch_size)
            _,loss_value,step=sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})
            if i%100==0:
                accuracy_score=sess.run(accuracy,feed_dict=validate_feed)
                print('After {} steps,loss value is {},accuracy is {}'.format(step,loss_value,accuracy_score))
                saver.save(sess,os.path.join(model_save_path,model_name),global_step=global_step)
                result=sess.run(merged,feed_dict=validate_feed)
                writer.add_summary(result,i)

mnist = input_data.read_data_sets(r"C:\Users\Natsu\Desktop\MacroMap\TENSORV\data", one_hot=True)
train(mnist)


