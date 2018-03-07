import sys
import unicodedata
import json
from nltk.stem.lancaster import LancasterStemmer
import nltk
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.optimizers import Adam
steamer=LancasterStemmer()


punctuation=dict.fromkeys([i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P')])

def remove_pun(sentence):
    return sentence.translate(punctuation)

steamer=LancasterStemmer()
data=None
with open('data.json','r') as f:
    data=json.load(f)

categories=list(data.keys())

train_x=[]
final_row=[]

for i in data:
    for k in data[i]:
        remove_pu=remove_pun(k)

        token=nltk.word_tokenize(remove_pu)
        steaming=[steamer.stem(i.lower()) for i in token]
        final_row.extend(steaming)
        train_x.append((steaming,i))

final_row=sorted(list(set(final_row)))



row_matrix=[]
loss_calculation_matrix=[]

for i in train_x:
    dat=i[0]
    row_matrix.append([1 if i in dat else 0 for i in final_row])
    dat2=i[1]

    loss_calculation_matrix.append([1 if i in dat2 else 0 for i in categories])

tf.reset_default_graph()
categories=['Future','Present']

net=tflearn.input_data(shape=[None,34])
net=tflearn.fully_connected(net,8)
net=tflearn.fully_connected(net,8)
net=tflearn.fully_connected(net,2,activation='softmax')
adam=Adam(learning_rate=0.001)
net=tflearn.regression(net,optimizer=adam)

model=tflearn.DNN(net,tensorboard_dir='tflearn_log_dir')

model.load('/Users/exepaul/Desktop/model_save/modelprescience.tflearn')

sent_1 = "What is tomorrow Bitcoin price?"
sent_2 = "Future price of Bitcoin?"
sent_3 = "Current rate of Bitcoin?"
sent_4 = "Today price of Bitcoin?"
sent_5="What will be the Bitcoin price?"
sent_6="tell me price bitcoin price"
sent_7="future price of bitcoin"
sent_77="next price of bitcoin?"
sent_8="forward price of bitcoin?"




def testing_model(sentence):
    global final_row
    token=nltk.word_tokenize(sentence)
    steaming12=[steamer.stem(i.lower()) for i in token]
    boe=[0]*len(final_row)
    for i in steaming12:
        for m,n in enumerate(final_row):
            if i==n:
                boe[m]=1
    return np.array(boe)

print(categories[np.argmax(model.predict([testing_model(sent_1)]))])
print(categories[np.argmax(model.predict([testing_model(sent_2)]))])
print(categories[np.argmax(model.predict([testing_model(sent_3)]))])
print(categories[np.argmax(model.predict([testing_model(sent_4)]))])
print(categories[np.argmax(model.predict([testing_model(sent_5)]))])
print(categories[np.argmax(model.predict([testing_model(sent_6)]))])
print(categories[np.argmax(model.predict([testing_model(sent_77)]))])
print(categories[np.argmax(model.predict([testing_model(sent_7)]))])
print(categories[np.argmax(model.predict([testing_model(sent_8)]))])