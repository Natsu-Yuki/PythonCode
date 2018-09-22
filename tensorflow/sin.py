import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

def Cyclic_neural_network():
    x=[1,2]
    state=[0.0,0.0]
    w_cell_state=np.asarray([[0.1,0.2],
                             [0.3,0.4]
                             ])
    w_cell_input=np.asarray([0.5,0.6])
    b_cell=np.asarray([0.1,-0.1])
    w_output=np.asarray([[1.0],
                         [2.0]
                         ])
    b_output=0.1

    for i in range(len(x)):
        before_activation=np.dot(state,w_cell_state)+x[i]*w_cell_input+b_cell
        state=np.tanh(before_activation)
        final_output=np.dot(state,w_output)+b_output
        print('before_activation:{}'.format(before_activation))
        print('state:{}'.format(state))
        print('final_output:{}'.format(final_output))

hidden_size=30
num_layers=2

timesteps=10
training_step=10000
batch_size=32

training_examples=10000
testing_examples=1000
sample_gap=0.01

def generate_data(sep):
    x=[]
    y=[]
    for i in range(len(sep)-timesteps):
        x.append([sep[i:i+timesteps]])
        y.append([sep[i+timesteps]])
    return np.array(x,dtype=np.float32),np.array(y,dtype=np.float32)


def lstm_model(x,y,is_training):
    cell=tf.nn.rnn_cell.MultiRNNCell(
        [tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        for _ in range(num_layers)]
    )
    outputs,_=tf.nn.dynamic_rnn(cell,x,dtype=tf.float32)
    output=outputs[:,-1,:]
    predictions=tf.contrib.layers.fully_connected(
        output,1,activation_fn=None
    )
    if not is_training:
        return predictions,None,None
    loss=tf.losses.mean_squared_error(labels=y,predictions=predictions)
    train_op=tf.contrib.layers.optimize_loss(
        loss,tf.train.get_global_step(),optimizer='Adagrad',learning_rate=0.1
    )
    return predictions,loss,train_op

def train(sess,train_x,train_y):
    ds=tf.data.Dataset.from_tensor_slices((train_x,train_y))
    ds=ds.repeat().shuffle(1000).batch(batch_size)
    x,y=ds.make_one_shot_iterator().get_next()
    predictions,loss,train_op=lstm_model(x,y,True)
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        _,l=sess.run([train_op,loss])
        if i%100==0:
            print('train steps:{},loss:{}'.format(i,l))

def run_eval(sess,test_x,test_y):
    ds = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    ds = ds.batch(1)

    x, y = ds.make_one_shot_iterator().get_next()
    print(0)
    prediction, loss, train_op = lstm_model(x, [0.0], False)
    print(1)
    predictions=[]
    labels=[]
    for i in range(testing_examples):
        p,l=sess.run([prediction,y])
        predictions.append(p)
        labels.append(l)
    predictions=np.array(predictions).squeeze()
    labels=np.array(labels).squeeze()
    rmse=np.sqrt(((predictions-labels)**2).mean(axis=0))
    print('MSE:{}'.format(rmse))

    plt.figure()
    plt.plot(predictions,label='predictions')
    plt.plot(labels,label='real_sin')
    plt.legend()
    plt.show()


test_start=(training_examples+timesteps)*sample_gap
test_end=test_start+(training_examples+timesteps)*sample_gap
train_x,train_y=generate_data(
    np.sin(
        np.linspace(
            0,test_start,training_examples+timesteps,dtype=np.float32
        )
    )
)

test_x,test_y=generate_data(
    np.sin(
        np.linspace(
            test_start,test_end,training_examples+timesteps,dtype=np.float32
        )
    )
)

with tf.Session() as sess:
    train(sess,train_x,train_y)
    run_eval(sess,test_x,test_y)