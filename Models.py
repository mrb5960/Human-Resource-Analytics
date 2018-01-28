import pandas as pd
import numpy as np
import matplotlib.pyplot as pplot
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
import pydotplus
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

color_index = -1

def classify(classifier, df, classifier_name):
    global color_index
    train, test = train_test_split(df, test_size=0.2, random_state=0)

    train_features = train.ix[:, train.columns != 'left']

    train_target = train.left

    test_features = test.ix[:, test.columns != 'left']

    test_target = test.left

    X = train

    y = train.pop('left')
    classifier.fit(X, y)
    test_predictions = classifier.predict(test_features)
    train_predictions = classifier.predict(train_features)
    test_score = metrics.f1_score(test_target, test_predictions)
    train_score = metrics.f1_score(train_target, train_predictions)
    print('Training score: ', train_score)
    print('Testing score: ', test_score)

    cmat = metrics.confusion_matrix(test_target, test_predictions)
    print(cmat)

    precision, recall, fscore, support = metrics.precision_recall_fscore_support(test_target, test_predictions)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('Fscore: ', fscore)
    print('Support: ', support)

    fpr, tpr, threshold = metrics.roc_curve(test_target, test_predictions)
    roc_auc = metrics.auc(fpr, tpr)

    colors = ['blue', 'red', 'green', 'black', 'magenta', 'brown', 'grey', 'yellow', 'orange', 'pink', 'cyan']
    color_index += 1
    pplot.title('Receiver Operating Characteristic')
    pplot.plot(fpr, tpr, 'b', label='%(c_name)s = %(accuracy)0.2f' % {"c_name" : classifier_name, "accuracy" : roc_auc}, color=colors[color_index])
    pplot.legend(loc='lower right')
    pplot.plot([0, 1], [0, 1], 'r--')
    pplot.xlim([0, 1])
    pplot.ylim([0, 1])
    pplot.ylabel('True Positive Rate')
    pplot.xlabel('False Positive Rate')
    #pplot.show()

def main():
    path = "C:\\Study\\RIT\\BDA\\BDA_Term_Project\\HR_comma_sep.csv"

    df = pd.read_csv(path)
    df_new = df.copy()
    departments = df_new.department.unique().tolist()
    #print(departments)
    df_new['department'] = df_new['department'].replace(to_replace=departments, value=range(0,len(departments)))
    #print(df_new['department'].unique())
    salaries = df_new.salary.unique().tolist()
    df_new['salary'] = df_new['salary'].replace(to_replace=salaries, value=range(0, len(salaries)))
    #print(df_new['salary'].unique())
    #print(df_new.head())

    # print('-------------Random Forest--------------')
    rf = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=-1)
    classify(rf, df_new, "Random Forest")
    # rf.fit(X, y)






    # feature_importance = rf.feature_importances_
    # imp = pd.DataFrame(columns = [test_features.columns, feature_importance])
    # imp.columns = ["Percentage"]
    # imp.sort()
    # imp = imp.values.reshape(9,2)
    # imp = pd.DataFrame(imp)
    # print(imp.shape, " ", imp.head())
    # feature_importance.sort()

    # plot feature importance
    # pplot.bar(np.arange(0, len(feature_importance)), feature_importance*100)
    # pplot.xticks(np.arange(0, len(feature_importance)), test_features.columns, rotation='vertical')
    # pplot.title('Random forest feature importance')
    # pplot.xlabel('Features')
    # pplot.ylabel('Percentage')
    # pplot.tight_layout()
    # pplot.show()
    #
    # test_predictions = rf.predict(test_features)
    # train_predictions = rf.predict(train_features)
    # test_score = metrics.f1_score(test_target, test_predictions)
    # train_score = metrics.f1_score(train_target, train_predictions)
    # print('Training score: ', train_score)
    # print('Testing score: ', test_score)
    #
    # cmat = metrics.confusion_matrix(test_target, test_predictions)
    # print(cmat)
    #
    # precision, recall, fscore, support = metrics.precision_recall_fscore_support(test_target, test_predictions)
    # print('Precision: ', precision)
    # print('Recall: ', recall)
    # print('Fscore: ', fscore)
    # print('Support: ', support)

    # dot_data = tree.export_graphviz(rf, out_file=None, feature_names=test.columns)
    # graph = pydotplus.graph_from_dot_data(dot_data)
    # graph.write_pdf("C:/Study/RIT/BDA/BDA_Term_Project/RandomForest.pdf")

    # print('-------------kNN--------------')
    # for k in range(2,11):
    kNN = KNeighborsClassifier(n_neighbors=3)
    classify(kNN, df_new, "kNN")

    # print(('------------Gaussian Naive Bayes------------'))
    gnb = GaussianNB()
    classify(gnb, df_new, "Naive Bayes")
    #
    # print('-------------Decision Tree--------------')
    dtree = DecisionTreeClassifier()
    classify(dtree, df_new, "Decision Tree")
    #
    # print('------------Neural Network-------------')
    nnet = MLPClassifier()
    classify(nnet, df_new, "ANN")
    # x = df_new['satisfaction_level']
    # y = df_new['average_monthly_hours']
    # c = df_new['left']
    # p = pplot.scatter(x=x, y=y, c=c)
    # pplot.xlabel('Satisfaction level')
    # pplot.ylabel('Average monthly hours')
    # #pplot.legend(p)
    pplot.show()

def performPCA(df_new):
    df_pca = df_new.copy()
    df_pca.drop(['left', 'department', 'salary'], axis=1)
    X = np.array(df_pca)
    X = scale(X)
    pca = PCA().fit(X)
    # pca_transform = pca.transform(X)

    temp = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4) * 100)

main()
