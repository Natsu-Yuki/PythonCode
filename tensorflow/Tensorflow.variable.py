import tensorflow as tf
state=tf.Variable(0,name='counter')
one=tf.constant(1)
new=tf.add(state,one)
update=tf.assign(state,new)
init=tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    for i in range(3):
        print(sess.run(state))


