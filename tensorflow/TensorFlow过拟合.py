import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from  sklearn.preprocessing import  LabelBinarizer

digits=load_digits()
X=digits.data
y=digits.target
y=LabelBinarizer().fit_transform(y)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

def add_layer(inputs, in_size, out_size, layer_name,activation_function=None):
    weight = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)
    wx_plus_b = tf.matmul(inputs, weight)+biases
    if activation_function is None:
        outputs = wx_plus_b
    else:
        outputs = activation_function(wx_plus_b)
    tf.summary.histogram(layer_name+'/outputs',outputs)
    return  outputs

xs=tf.placeholder(tf.float32,[None,64])
ys=tf.placeholder(tf.float32,[None,10])

l1=add_layer(xs,64,100,'l1',activation_function=tf.nn.tanh)
predition=add_layer(l1,100,10,'l2',activation_function=tf.nn.softmax)

cross_entropy=tf.reduce_mean(
    -tf.reduce_sum(
        ys*tf.log(predition),reduction_indices=[1]
    )
)



tf.summary.scalar('loss',cross_entropy)
train_step=tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy)
with tf.Session() as sess:
    merged=tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(r"C:\Users\Natsu\Desktop\TENSORV\c", sess.graph)

    test_writer = tf.summary.FileWriter(r"C:\Users\Natsu\Desktop\TENSORV\d", sess.graph)
    for i in range(500):
        sess.run(train_step, feed_dict={xs: X_train, ys: y_train})

        if i%50==0:
            train_result=sess.run(merged,feed_dict={xs: X_train, ys: y_train})
            test_result = sess.run(merged, feed_dict={xs: X_test, ys: y_test})
            train_writer.add_summary(train_result,i)
            test_writer.add_summary(test_result,i)
