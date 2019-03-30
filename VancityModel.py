# Given a financial dataset of various customer, predict the likelihood of a customer to sign up for a RRSP account

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing

from mlxtend.plotting import plot_learning_curves

# read csv data
df = pd.read_csv('vcRSP2017.csv')

totalRows,totalFeatures = df.shape

# 0.8 Training, 0.2 X-val, 0.2 Testing data
train_df,xval_df,test_df = np.split(df,[int(0.6*len(df)),int(0.8*len(df))])

combine = [train_df,xval_df,test_df]

for dataset in combine:
    dataset.loc[(dataset['APURCH'] == 'Y'), 'APURCH'] = 1
    dataset.loc[(dataset['APURCH'] == 'N'),'APURCH'] = 0



# grid = sns.FacetGrid(train_df, col='APURCH')
# grid.map (plt.hist, 'age',bins=20)
# plt.show()

# Set NA Values to median values
freq_age = train_df.age.dropna().median()
# print(freq_age)
for dataset in combine:
    dataset.loc[dataset['age']==0,'age'] = freq_age

# feature engineer data
# for dataset in combine:
#     dataset.loc[(dataset['age'] <= 28.2),'age'] = 0
#     dataset.loc[(dataset['age'] > 28.2) & (dataset['age'] <=38.4), 'age'] = .25
#     dataset.loc[(dataset['age'] > 38.4) & (dataset['age'] <=48.6), 'age'] = .5
#     dataset.loc[(dataset['age'] > 48.6) & (dataset['age'] <=58.8), 'age'] = .75
#     dataset.loc[(dataset['age'] > 58.8), 'age'] = 1
#
#     dataset['gender'] = 0
#     dataset.loc[(dataset['gendm']==1),'gender'] = 1
#     dataset.loc[(dataset['gendm']==0)&(dataset['gendf']==0),'gender'] = 0
#
#

    dataset['BALCHQ'] = dataset['BALCHQ'].fillna(0)
    dataset['BALSAV'] = dataset['BALSAV'].fillna(0)
    dataset['BALLOAN'] = dataset['BALLOAN'].fillna(0)
    dataset['BALLOC'] = dataset['BALLOC'].fillna(0)
    dataset['BALMRGG'] = dataset['BALMRGG'].fillna(0)
    dataset['N_IND_INC_']= train_df.N_IND_INC_.dropna().mode()[0]
    dataset['numrr_1']= dataset['numrr_1'].fillna(train_df.numrr_1.dropna().mode()[0])
    dataset['numcon_1']= dataset['numcon_1'].fillna(train_df.numcon_1.dropna().mode()[0])
    dataset['avginv_1'] = dataset['avginv_1'].fillna(0)
    dataset['avginc_1'] = dataset['avginc_1'].fillna(0)
    dataset['CH_NM_PRD'] = dataset['CH_NM_PRD'].fillna(0)
    dataset['CH_NM_SERV'] = dataset['CH_NM_SERV'].fillna(0)

    # dataset['savings'] = dataset['BALSAV'] + dataset['BALCHQ']

# for dataset in combine:
#     dataset['transactions'] = dataset['TXBRAN'] + dataset['TXATM'] + dataset['TXATM'] +dataset['TXTEL'] + dataset['TXPOS'] + dataset['TXWEB']+dataset['TXCHQ']
# print(pd.qcut(train_df['transactions'], 5))

# for dataset in combine:
#     dataset.loc[(dataset['transactions'] <=1.25),'transactions'] = 0
#     dataset.loc[(dataset['transactions'] > 1.25) & (dataset['transactions'] <=6.833), 'transactions'] = 0.25
#     dataset.loc[(dataset['transactions'] > 6.833) & (dataset['transactions'] <= 18.5), 'transactions'] = 0.5
#     dataset.loc[(dataset['transactions'] > 18.5) & (dataset['transactions'] <= 36.0), 'transactions'] = 0.75
#     dataset.loc[(dataset['transactions'] > 36.0), 'transactions'] = 1

# for dataset in combine:
#     dataset['avgincome'] = dataset['avginc_1'] + dataset['avginv_1']
# print(pd.qcut(train_df['avgincome'], 5))

# for dataset in combine:
#     dataset.loc[(dataset['avgincome'] <= 27450.635), 'transactions'] = 0
# # print(pd.qcut(train_df['transactions'], 5))
#
# train_df = train_df.drop(['gendm','gendf','BALSAV','BALCHQ'],axis = 1)
# xval_df = xval_df.drop(['gendm','gendf','BALSAV','BALCHQ'],axis = 1)
# test_df = test_df.drop(['gendm','gendf','BALSAV','BALCHQ'],axis = 1)

# print(pd.qcut(train_df['savings'],5))

combine = [train_df,xval_df,test_df]

# for dataset in combine:
#     dataset.loc[(dataset['savings'] <= 99.396),'savings'] = 0
#     dataset.loc[(dataset['savings'] > 99.396) & (dataset['savings'] <= 645.431),'savings'] = 0.25
#     dataset.loc[(dataset['savings'] > 645.431) & (dataset['savings'] <= 1830.809), 'savings'] = .5
#     dataset.loc[(dataset['savings'] > 1830.809) & (dataset['savings'] <= 4866.124), 'savings'] = .75
#     dataset.loc[(dataset['savings'] > 4866.124), 'savings'] = 1
# print(train_df[['savings','APURCH']].groupby(['savings'],as_index=False).mean().sort_values(by='APURCH',ascending=False))
# print(pd.qcut(train_df['TOTDEP'],5))
freq_segment = train_df.valsegm.dropna().mode()[0]
# print(freq_segment)

for dataset in combine:
    # dataset.loc[(dataset['TOTDEP'] <= 325.126),'TOTDEP'] = 0
    # dataset.loc[(dataset['TOTDEP'] > 325.126) & (dataset['TOTDEP'] <= 1118.013), 'TOTDEP'] = 1
    # dataset.loc[(dataset['TOTDEP'] > 1118.013) & (dataset['TOTDEP'] <= 3225.706), 'TOTDEP'] = 2
    # dataset.loc[(dataset['TOTDEP'] > 3225.706) & (dataset['TOTDEP'] <= 10410.799), 'TOTDEP'] = 3
    # dataset.loc[(dataset['TOTDEP'] > 10410.799),'TOTDEP'] = 4

    dataset['valsegm'] = dataset['valsegm'].fillna(freq_segment)

segmentMapping = {"A":0,"B":0.25,"C":0.5,"D":.75,"E":1}

for dataset in combine:
    dataset['valsegm'] = dataset['valsegm'].map(segmentMapping)

# Remove
train_df = train_df.drop(['unique', 'pcode','N_IND_INC_','numrr_1','numcon_1','TXBRAN','TXATM','TXPOS','TXCHQ'], axis=1)
test_df = test_df.drop(['unique', 'pcode','N_IND_INC_','numrr_1','numcon_1','TXBRAN','TXATM','TXPOS','TXCHQ'], axis=1)
xval_df = xval_df.drop(['unique', 'pcode','N_IND_INC_','numrr_1','numcon_1','TXBRAN','TXATM','TXPOS','TXCHQ'], axis=1)



# train_df = train_df.drop(['unique', 'pcode', 'BALLOAN', 'BALLOC','BALMRGG','TXBRAN','TXATM','TXPOS','TXCHQ',
#                           'TXWEB','TXTEL','TOTSERV','CH_NM_SERV','CH_NM_PRD','N_IND_INC_','numrr_1','numcon_1',
#                           'avginc_1','avginv_1'], axis=1)
# test_df = test_df.drop(['unique', 'pcode', 'BALLOAN', 'BALLOC','BALMRGG','TXBRAN','TXATM','TXPOS','TXCHQ',
#                           'TXWEB','TXTEL','TOTSERV','CH_NM_SERV','CH_NM_PRD','N_IND_INC_','numrr_1','numcon_1',
#                           'avginc_1','avginv_1'], axis=1)
# xval_df = xval_df.drop(['unique', 'pcode', 'BALLOAN', 'BALLOC','BALMRGG','TXBRAN','TXATM','TXPOS','TXCHQ',
#                           'TXWEB','TXTEL','TOTSERV','CH_NM_SERV','CH_NM_PRD','N_IND_INC_','numrr_1','numcon_1',
#                           'avginc_1','avginv_1'], axis=1)

# train_df = train_df.drop(['age','valsegm','gender','savings'],axis=1)
# xval_df = xval_df.drop(['age','valsegm','gender','savings'],axis=1)
# test_df = test_df.drop(['age','valsegm','gender','savings'],axis=1)

X_train = train_df.drop('APURCH',axis = 1)
X_train = preprocessing.scale(X_train)
Y_train = train_df['APURCH']
X_val = xval_df.drop('APURCH', axis = 1).copy()
X_val = preprocessing.scale(X_val)
Y_val = xval_df['APURCH']

# print(train_df.head())
# print(xval_df[['NEWMRGG','APURCH']].groupby(['NEWMRGG'],as_index=False).mean().sort_values(by='APURCH',ascending=False))

# print(X_val.head())
# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train,Y_train)
Y_test = logreg.predict(X_val)
acc_log = round(logreg.score(X_train, Y_train)*100 ,2)

# Determine correlation of each feature to the result
coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df. columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])
print(coeff_df.sort_values(by='Correlation',ascending=False))
# print(train_df.head(20))
clf = MLPClassifier(max_iter=2000,alpha=0.001)
# print(X_train.shape,Y_train.shape,X_val.shape,Y_val.shape)
# clf = SVC()
plot_learning_curves(X_train,Y_train,X_val,Y_val, clf)
plt.show()
# print(xval_df['APURCH'].size)
# print(Y_val.shape)

# plt.plot(train_sizes,train_scores,'r')
# plt.plot(train_scores,valid_scores,'b')
# plt.show()

#Train various model using data and return it's accuraries
svc = SVC()
svc.fit(X_train,Y_train)
Y_val = svc.predict(X_val)
acc_svc = round(svc.score(X_train,Y_train)*100,2)
print("acc svc = ", acc_svc)

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_val = random_forest.predict(X_val)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print("acc randon forest = ",acc_random_forest)

# count = 0
# correct = 0
# for i in range(xval_df['APURCH'].size):
#     if Y_val[i] == xval_df.iloc[i]['APURCH']:
#         correct = correct + 1
#     count = count + 1

# print("score: ", correct,". count:", count , "acc:", round(correct/count,2))
#
# print("acc log = ", acc_log)
# print(train_df.head())
    # dataset['BALLOAN'] = dataset['BALLOAN'].fillna(0).astype(int)
    # dataset['BALLOC'] = dataset['BALLOC'].fillna(0).astype(int)
    # dataset['BALMRGG'] = dataset['BALMRGG'].fillna(0).astype(int)
    # dataset['loans'] = dataset['BALLOAN'] +dataset['BALLOC'] + dataset['BALMRGG']
    # dataset['hasLoan'] = 1
    # dataset.loc[dataset['loans'] < 1, 'hasLoan'] = 0
#
# grid = sns.FacetGrid(train_df, col='APURCH')
# grid.map (plt.hist, 'age',bins=5)
# plt.show()
# print(pd.qcut(train_df['loans'],4))





# print(train_df[['TOTDEP', 'APURCH']].groupby(['TOTDEP'], as_index=False).mean().sort_values(by='APURCH',
#                                                                                                   ascending=False))
    # dataset = dataset.drop(['gen'])
# print(train_df[['gender','APURCH']].groupby(['gender'],as_index=False).mean().sort_values(by='APURCH',ascending=False))
    # dataset['gender'] = dataset.loc[]
# print(train_df[['age','APURCH']].groupby(['age'],as_index=False).mean().sort_values(by='APURCH',ascending=False))

# print(train_df[['gendm','APURCH']].groupby(['gendm'],as_index=False).mean().sort_values(by='APURCH',ascending=False))

# print(xval_df.head(20))
# print(train_df.iloc[[766]])

