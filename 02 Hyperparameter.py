from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

test = []

wine = load_wine()

Xtrain,Xtest,Ytrain,Ytest = train_test_split(wine.data,wine.target,train_size=0.7)

for i in range(10):
    clf = tree.DecisionTreeClassifier(criterion="entropy"
                                      ,max_depth=i+1
                                      ,random_state=30
                                      )
    clf = clf.fit(Xtrain,Ytrain)
    score = clf.score(Xtest,Ytest)
    test.append(score)
plt.plot(range(1,11),test,color = 'red',label='max_depth')
plt.legend()
plt.show()
