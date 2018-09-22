import time
t=time.clock()
import urllib.request
import os
import tarfile
url="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
filepath="data/aclImdb_v1.tar.gz"
if not os.path.isfile(filepath):
    result=urllib.request.urlretrieve(url,filepath)
    print('downloaded:',result)
if not os.path.exists("data/aclImdb"):
    tfile = tarfile.open("data/aclImdb_v1.tar.gz", 'r:gz')
    result=tfile.extractall('data/')

from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,Flatten
from keras.layers.embeddings import Embedding

import re
def rm_tags(text):
    re_tag = re.compile(r'<[^>]+>')
    return re_tag.sub('', text)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def read_files(filetype):
    path = "data/aclImdb/"
    file_list = []

    positive_path = path + filetype + "/pos/"
    for f in os.listdir(positive_path):
        file_list += [positive_path + f]

    negative_path = path + filetype + "/neg/"
    for f in os.listdir(negative_path):
        file_list += [negative_path + f]

    print('read', filetype, 'files:', len(file_list))

    all_labels = ([1] * 12500 + [0] * 12500)

    all_texts = []
    for fi in file_list:
        with open(fi, encoding='utf8') as file_input:
            all_texts += [rm_tags(" ".join(file_input.readlines()))]

    return all_labels, all_texts

y_train,train_text=read_files("train")
y_test,test_text=read_files("test")



token = Tokenizer(num_words=2000)
token.fit_on_texts(train_text)
print(token.document_count)
print(token.word_index)

x_train_seq = token.texts_to_sequences(train_text)
x_test_seq  = token.texts_to_sequences(test_text)

x_train = sequence.pad_sequences(x_train_seq, maxlen=100)
x_test  = sequence.pad_sequences(x_test_seq,  maxlen=100)

model = Sequential()
model.add(Embedding(output_dim=32,
                    input_dim=2000,
                    input_length=100))
model.add(Dropout(0.2))

from keras.layers.recurrent import SimpleRNN
model.add(SimpleRNN(units=16))

from keras.layers.recurrent import LSTM
model.add(LSTM(units=32))



model.add(Flatten())
model.add(Dense(units=256,
                activation='relu' ))
model.add(Dropout(0.2))
model.add(Dense(units=1,
                activation='sigmoid' ))
model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

train_history =model.fit(x_train, y_train,batch_size=100,
                         epochs=10,verbose=2,
                         validation_split=0.2)
scores = model.evaluate(x_test, y_test, verbose=1)
print(scores[1])
probility=model.predict(x_test)

import matplotlib.pyplot as plt
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
show_train_history(train_history,'acc','val_acc')

print('time:{}s'.format(time.clock()-t))