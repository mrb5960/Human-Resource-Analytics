import matplotlib.pyplot as pplot
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import scale
from sklearn import metrics
from sklearn.model_selection import train_test_split


color_index = -1

def classify(classifier, df, classifier_name):
    global color_index
    train, test = train_test_split(df, test_size=0.2)

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

def scatter_plot(df_new, x_data, y_data):
    x = df_new[x_data]
    y = df_new[y_data]
    c = df_new['left']
    p = pplot.scatter(x=x, y=y, c=c)
    pplot.xlabel(x_data)
    pplot.ylabel(y_data)
    #pplot.legend(c)
    pplot.show()

def bar_plot(df_new, x_data, y_data):
    x = df_new[x_data].unique()
    y = df_new[y_data]
    #c = df_new['left']
    p = pplot.bar(x=x, y=y)
    pplot.xlabel(x_data)
    pplot.ylabel(y_data)
    # pplot.legend(c)
    pplot.show()


def main():
    path = "C:\\Study\\RIT\\BDA\\BDA_Term_Project\\HR_comma_sep.csv"

    df = pd.read_csv(path)

    df_new = df.copy()

    departments = df_new.department.unique().tolist()

    df_new['department'] = df_new['department'].replace(to_replace=departments, value=range(0, len(departments)))

    salaries = df_new.salary.unique().tolist()

    df_new['salary'] = df_new['salary'].replace(to_replace=salaries, value=range(0, len(salaries)))

    scatter_plot(df_new, 'satisfaction_level', 'last_evaluation')
    scatter_plot(df_new, 'satisfaction_level', 'average_monthly_hours')
    scatter_plot(df_new, 'average_monthly_hours', 'last_evaluation')
    #bar_plot(df_new, 'time_spend_company', 'satisfaction_level')

    df_pca = df_new.copy()
    df_pca.drop(['left', 'department', 'salary'], axis=1)
    X = np.array(df_pca)
    X = scale(X)
    pca = PCA().fit(X)


    temp = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
    pplot.plot(temp)
    pplot.xlim(0,12,1)
    pplot.xlabel('Number of components')
    pplot.ylabel('Cumulative explained variance')
    pplot.title('Principal Component Analysis')
    pplot.show()
    print(pca.explained_variance_ratio_)
    print(pca.components_[0])

    pca = PCA(n_components=8)
    pca_transform = pca.fit_transform(df_new)
    pca_transform = pd.DataFrame(pca_transform)

    print(pca_transform.shape)

main()
