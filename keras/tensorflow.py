import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist = input_data.read_data_sets(r"C:\Users\Natsu\Desktop\MacroMap\TENSORV\data", one_hot=True)





def layer(output_dim,input_dim,inputs, activation=None):
    W = tf.Variable(tf.random_normal([input_dim, output_dim]))
    b = tf.Variable(tf.random_normal([1, output_dim]))
    XWb = tf.matmul(inputs, W) + b
    if activation is None:
        outputs = XWb
    else:
        outputs = activation(XWb)
    return outputs

x = tf.placeholder("float", [None, 784])

h1=layer(output_dim=256,input_dim=784,
         inputs=x ,activation=tf.nn.relu)

y_predict=layer(output_dim=10,input_dim=256,
                    inputs=h1,activation=None)

y_label = tf.placeholder("float", [None, 10])

loss_function = tf.reduce_mean(
                  tf.nn.softmax_cross_entropy_with_logits
                         (logits=y_predict ,
                          labels=y_label))


global_=tf.Variable(0)
learning_rate=tf.train.exponential_decay(0.001,global_,100,0.96,staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) \
                    .minimize(loss_function)

correct_prediction = tf.equal(tf.argmax(y_label  , 1),
                              tf.argmax(y_predict, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

trainEpochs = 15
batchSize = 100
totalBatchs = int(mnist.train.num_examples/batchSize)
epoch_list=[];loss_list=[];accuracy_list=[]
from time import time
startTime=time()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(trainEpochs):
    for i in range(totalBatchs):
        batch_x, batch_y = mnist.train.next_batch(batchSize)
        sess.run(optimizer, feed_dict={x: batch_x, y_label: batch_y})

    loss, acc = sess.run([loss_function, accuracy],
                         feed_dict={x: mnist.validation.images,
                                    y_label: mnist.validation.labels})

    epoch_list.append(epoch);
    loss_list.append(loss)
    accuracy_list.append(acc)
    print("Train Epoch:", '%02d' % (epoch + 1), "Loss=", \
          "{:.9f}".format(loss), " Accuracy=", acc)

duration = time() - startTime
print("Train Finished takes:", duration)

