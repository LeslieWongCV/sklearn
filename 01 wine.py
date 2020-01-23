#3.3 树训练完毕 但是图打印出问题
#3.3 两种方式解决可视化 tree.export_graphviz(clf,feature_names=....,out_file='tree.dot') 输出dot文件后终端 dot -Tpdf tree.dot -o tree.pdf 打印图片
#    或者不输出 dot_data 传入pydotplus.graph_from_dot_data() 直接转换为pdf
#输出图片 节点颜色越浅，代表不纯度越高
from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import pydotplus as pyd
import pandas as pd
import graphviz

wine = load_wine()
#print(pd.concat([pd.DataFrame(wine.data) ,pd.DataFrame(wine.target)],axis=1))

#print(wine)

Xtrain,Xtest,Ytrain,Ytest = train_test_split(wine.data,wine.target,train_size=0.7)
print(Xtrain.shape)
print(Ytrain.shape)

clf = tree.DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(Xtrain,Ytrain)
#score = clf.score(Xtest,Ytest)
#print(wine.keys()) 查看字典的所有键的名称
#print(score)
###########################################方法一#################################################
# dot_data = tree.export_graphviz(clf
#                                 ,feature_names=wine.feature_names
#                                 ,rounded = True
#                                 ,filled = True
#                                 ,class_names=wine.target_names
#                                 ,out_file='tree.dot') #out_file 选择输出文件'name.dot'格式，再终端dot -Tpdf name.dot -o tree.pdf 查看决策树
###########################################方法二#################################################
dot_data = tree.export_graphviz(clf
                                ,feature_names=wine.feature_names
                                ,class_names=wine.target_names
                                ,filled=True
                                ,rounded=True
                                ,out_file=None)#ouf_file选择None，import pydotplus
graph = pyd.graph_from_dot_data(dot_data)
graph.write_pdf('tree.pdf')
##################################################################################################
clf.feature_importances_
feature_importance = list(zip(wine.feature_names, clf.feature_importances_))
for a in (feature_importance):
    print(a)
