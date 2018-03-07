import sys
import unicodedata
import json
from nltk.stem.lancaster import LancasterStemmer
import nltk
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.optimizers import Adam



punctuation=dict.fromkeys([i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P')])

def remove_pun(sentence):
    return sentence.translate(punctuation)

steamer=LancasterStemmer()
data=None
with open('data.json','r') as f:
    data=json.load(f)

categories=list(data.keys())
print(categories)

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

print(len(final_row))
print("en1",len(row_matrix[0]))
print("len2",len(loss_calculation_matrix[0]))
print([len(i) for i in row_matrix])
tf.reset_default_graph()

net=tflearn.input_data(shape=[None,len(row_matrix[0])])
net=tflearn.fully_connected(net,8)
net=tflearn.fully_connected(net,8)
net=tflearn.fully_connected(net,len(loss_calculation_matrix[0]),activation='softmax')
adam=Adam(learning_rate=0.001)
net=tflearn.regression(net,optimizer=adam)

model=tflearn.DNN(net,tensorboard_dir='tflearn_log_dir')
model.fit(row_matrix,loss_calculation_matrix,n_epoch=10000,batch_size=8,show_metric=True)
model.save('/Users/exepaul/Desktop/model_save/modelprescience.tflearn')

sent_1 = "What is tomorrow Bitcoin price?"
sent_2 = "Future price of Bitcoin?"
sent_3 = "Current rate of Bitcoin?"
sent_4 = "Today price of Bitcoin?"


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



# ['about', 'ahead', 'ar', 'be', 'bitco', 'bitcoin', 'cur', 'doing', 'highest', 'how', 'i', 'in', 'invest', 'is', 'me', 'mon', 'next', 'of', 'on', 'peopl', 'point', 'predict', 'pric', 'should', 'talk', 'tel', 'the', 'today', 'tomorrow', 'was', 'week', 'what', 'wil', 'yesterday']
# [(['How', 'Bitcoin', 'will', 'be', 'doing', 'tomorrow'], 'Future'), (['How', 'bitcoin', 'will', 'be', 'doing', 'in', 'next', 'one', 'week'], 'Future'), (['How', 'Bitcoin', 'will', 'be', 'doing', 'in', 'one', 'month', 'ahead'], 'Future'), (['What', 'will', 'be', 'the', 'Bitcoing', 'price', 'tomorrow'], 'Future'), (['Should', 'i', 'invest', 'in', 'Bitcoin'], 'Future'), (['What', 'is', 'prediction', 'of', 'bitcoin', 'Tomorrow'], 'Future'), (['How', 'is', 'Bitcoin', 'doing', 'today'], 'Present'), (['What', 'People', 'are', 'talking', 'about', 'Bitcoin'], 'Present'), (['What', 'is', 'current', 'Bitcoin', 'price'], 'Present'), (['Tell', 'me', 'highest', 'point', 'of', 'Bitcoin', 'today'], 'Present'), (['What', 'was', 'Bitcoing', 'price', 'Yesterday'], 'Present')]
# [0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 1 0 0]

