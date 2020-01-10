# train_features=np.array(train_features)
# labels=np.array(labels)

train_data=[]
train_labels=[]
for i in range(len(train_features)):
    if list(set(train_features[i]))!=[0]:
        sum=np.zeros(100)
        k=0
        for j in range(len(train_features[i])):
            if train_features[i][j]!=0:
                sum+=embeddings_matrix[train_features[i][j]]
                k+=1
        a=sum/k
        train_data.append(list(a))
        train_labels.append(labels[i])

train_features=np.array(train_data)
labels=np.array(train_labels)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
x_train, x_test, y_train, y_test = train_test_split(train_features, labels, test_size=0.1)

y_train=y_train
LR=LogisticRegression()
LR.fit(x_train,y_train)
print("模型得分:",accuracy_score(y_train,LR.predict(x_train)))
print('ROC面积:',roc_auc_score(y_train, LR.predict(x_train)))
print(classification_report(y_train,LR.predict(x_train)))