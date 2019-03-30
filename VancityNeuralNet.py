# Train deep neural network using Vancity RRSP Data. To understand tensorflow framework and building the layers of the NN

import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib as plt

df = pd.read_csv('vcRSP2017.csv')

totalRows,totalFeatures = df.shape

train_df,xval_df,test_df = np.split(df,[int(0.6*len(df)),int(0.8*len(df))])
# train_df = df.iloc[: int(round(totalRows * 0.6,0)),:]
# xval_df = df.iloc[int(round(totalRows * 0.6,0)):int(round(totalRows * 0.2 + totalRows * 0.6,0)),:]
# test_df = df.iloc[int(round(totalRows * 0.2 + totalRows * 0.6,0)):,:]

# print(df.shape)

combine = [train_df,xval_df,test_df]

for dataset in combine:
    dataset.loc[(dataset['APURCH'] == 'Y'), 'APURCH'] = 1
    dataset.loc[(dataset['APURCH'] == 'N'),'APURCH'] = 0



# grid = sns.FacetGrid(train_df, col='APURCH')
# grid.map (plt.hist, 'age',bins=20)
# plt.show()

freq_age = train_df.age.dropna().median()
# print(freq_age)
for dataset in combine:
    dataset.loc[dataset['age']==0,'age'] = freq_age

# train_df['ageBand'] = pd.cut(train_df['age'],5)
# print(pd.cut(train_df['age'],5))

for dataset in combine:
    dataset.loc[(dataset['age'] <= 28.2),'age'] = 0
    dataset.loc[(dataset['age'] > 28.2) & (dataset['age'] <=38.4), 'age'] = .25
    dataset.loc[(dataset['age'] > 38.4) & (dataset['age'] <=48.6), 'age'] = .5
    dataset.loc[(dataset['age'] > 48.6) & (dataset['age'] <=58.8), 'age'] = .75
    dataset.loc[(dataset['age'] > 58.8), 'age'] = 1

    dataset['gender'] = 0
    dataset.loc[(dataset['gendm']==1),'gender'] = 1
    dataset.loc[(dataset['gendm']==0)&(dataset['gendf']==0),'gender'] = 0



    dataset['BALCHQ'] = dataset['BALCHQ'].fillna(0)
    dataset['BALSAV'] = dataset['BALSAV'].fillna(0)
    dataset['savings'] = dataset['BALSAV'] + dataset['BALCHQ']

for dataset in combine:
    dataset['transactions'] = dataset['TXBRAN'] + dataset['TXATM'] + dataset['TXATM'] +dataset['TXTEL'] + dataset['TXPOS'] + dataset['TXWEB']+dataset['TXCHQ']
# print(pd.qcut(train_df['transactions'], 5))

for dataset in combine:
    dataset.loc[(dataset['transactions'] <=1.25),'transactions'] = 0
    dataset.loc[(dataset['transactions'] > 1.25) & (dataset['transactions'] <=6.833), 'transactions'] = 0.25
    dataset.loc[(dataset['transactions'] > 6.833) & (dataset['transactions'] <= 18.5), 'transactions'] = 0.5
    dataset.loc[(dataset['transactions'] > 18.5) & (dataset['transactions'] <= 36.0), 'transactions'] = 0.75
    dataset.loc[(dataset['transactions'] > 36.0), 'transactions'] = 1

# for dataset in combine:
#     dataset['avgincome'] = dataset['avginc_1'] + dataset['avginv_1']
# print(pd.qcut(train_df['avgincome'], 5))
#
# for dataset in combine:
#     dataset.loc[(dataset['avgincome'] <= 27450.635), 'transactions'] = 0
# print(pd.qcut(train_df['transactions'], 5))

train_df = train_df.drop(['gendm','gendf','BALSAV','BALCHQ'],axis = 1)
xval_df = xval_df.drop(['gendm','gendf','BALSAV','BALCHQ'],axis = 1)
test_df = test_df.drop(['gendm','gendf','BALSAV','BALCHQ'],axis = 1)

# print(pd.qcut(train_df['savings'],5))

combine = [train_df,xval_df,test_df]

for dataset in combine:
    dataset.loc[(dataset['savings'] <= 99.396),'savings'] = 0
    dataset.loc[(dataset['savings'] > 99.396) & (dataset['savings'] <= 645.431),'savings'] = 0.25
    dataset.loc[(dataset['savings'] > 645.431) & (dataset['savings'] <= 1830.809), 'savings'] = .5
    dataset.loc[(dataset['savings'] > 1830.809) & (dataset['savings'] <= 4866.124), 'savings'] = .75
    dataset.loc[(dataset['savings'] > 4866.124), 'savings'] = 1
# print(train_df[['savings','APURCH']].groupby(['savings'],as_index=False).mean().sort_values(by='APURCH',ascending=False))
# print(pd.qcut(train_df['TOTDEP'],5))
freq_segment = train_df.valsegm.dropna().mode()[0]
# print(freq_segment)

for dataset in combine:
    dataset.loc[(dataset['TOTDEP'] <= 325.126),'TOTDEP'] = 0
    dataset.loc[(dataset['TOTDEP'] > 325.126) & (dataset['TOTDEP'] <= 1118.013), 'TOTDEP'] = 1
    dataset.loc[(dataset['TOTDEP'] > 1118.013) & (dataset['TOTDEP'] <= 3225.706), 'TOTDEP'] = 2
    dataset.loc[(dataset['TOTDEP'] > 3225.706) & (dataset['TOTDEP'] <= 10410.799), 'TOTDEP'] = 3
    dataset.loc[(dataset['TOTDEP'] > 10410.799),'TOTDEP'] = 4

    dataset['valsegm'] = dataset['valsegm'].fillna(freq_segment)

segmentMapping = {"A":0,"B":0.25,"C":0.5,"D":.75,"E":1}

for dataset in combine:
    dataset['valsegm'] = dataset['valsegm'].map(segmentMapping)

train_df = train_df.drop(['unique', 'pcode', 'BALLOAN', 'BALLOC','BALMRGG','TXBRAN','TXATM','TXPOS','TXCHQ',
                          'TXWEB','TXTEL','TOTSERV','CH_NM_SERV','CH_NM_PRD','N_IND_INC_','numrr_1','numcon_1',
                          'avginc_1','avginv_1'], axis=1)
test_df = test_df.drop(['unique', 'pcode', 'BALLOAN', 'BALLOC','BALMRGG','TXBRAN','TXATM','TXPOS','TXCHQ',
                          'TXWEB','TXTEL','TOTSERV','CH_NM_SERV','CH_NM_PRD','N_IND_INC_','numrr_1','numcon_1',
                          'avginc_1','avginv_1'], axis=1)
xval_df = xval_df.drop(['unique', 'pcode', 'BALLOAN', 'BALLOC','BALMRGG','TXBRAN','TXATM','TXPOS','TXCHQ',
                          'TXWEB','TXTEL','TOTSERV','CH_NM_SERV','CH_NM_PRD','N_IND_INC_','numrr_1','numcon_1',
                          'avginc_1','avginv_1'], axis=1)

combine = [train_df,xval_df,test_df]
for dataset in combine:
    dataset['NOTPURCH'] = 0
    dataset.loc[(dataset['APURCH'] == 0),'NOTPURCH'] = 1

# train_df = train_df.drop(['age','valsegm','gender','savings'],axis=1)
# xval_df = xval_df.drop(['age','valsegm','gender','savings'],axis=1)
# test_df = test_df.drop(['age','valsegm','gender','savings'],axis=1)

train_x = train_df.drop(['APURCH','NOTPURCH'],axis = 1).copy()
train_y = np.array([train_df['APURCH'],train_df['NOTPURCH']]).T
X_val = xval_df.drop(['APURCH','NOTPURCH'], axis = 1).copy()
Y_val = np.array([xval_df['APURCH'],xval_df['NOTPURCH']]).T
test_x = test_df.drop(['APURCH','NOTPURCH'],axis = 1).copy()
test_y = np.array([test_df['APURCH'],test_df['NOTPURCH']]).T

# print(X_train,Y_train)
# print(len(train_x.columns))

#####MODELLING####################

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 2
batch_size = 100

#height x width
x = tf.placeholder('float',[None, len(train_x.columns)])
y = tf.placeholder('float')

def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([len(train_x.columns),n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                      'biases': tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])

    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    hm_epochs = 200

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())


        for epoch in range(hm_epochs):
            epoch_loss = 0

            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size
                batch_x = np.array(train_x[start:end])
                # print(batch_x.shape)
                # batch_y = np.array(train_y[start:end])
                batch_y = np.array(train_y[start:end])
                # print(batch_y.shape)

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_size

            print('Epoch', epoch+1, 'completed out of ', hm_epochs, 'loss:', epoch_loss)



            correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
            accuracy = tf.reduce_mean(tf.cast(correct,'float'))
            print('Accuracy',accuracy.eval({x:test_x, y:np.array(test_y)}))

train_neural_network(x)