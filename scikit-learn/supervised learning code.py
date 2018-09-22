import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mglearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer


def in_0():
    X, y = mglearn.datasets.make_forge()
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    plt.legend(["Class 0", "Class 1"], loc=4)
    plt.xlabel("Fist feature")
    plt.ylabel("Second feature")
    plt.show()


def in_1():
    X, y = mglearn.datasets.make_wave(n_samples=40)
    plt.plot(X, y, 'o')
    plt.ylim(-3, 3)
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.show()


def in_2():
    fig, axes = plt.subplot(1, 3, figsize=(10, 3))
    for n, ax in zip([1, 3, 9], axes):
        clf = KNeighborsClassifier(n_neighbors=n).fit(X, y)
        mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=.5, ax=ax)
        mglearn.discrete_scatter(X[:, 0], X[:, 0], y, ax=ax)
        ax.set_title("{} neighbor".format(n))
        ax.set_xlabel("feature 0")
        ax.set_ylabel("feature 1")
        plt.show()
    axes[0].legend(loc=3)


def in_3():
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target,
                                                        random_state=66)
    training_accuracy = []
    test_accuracy = []
    neighbors_settting = range(1, 11)

    for n in neighbors_settting:
        clf = KNeighborsClassifier(n_neighbors=n)
        clf.fit(X_train, y_train)
        training_accuracy.append(clf.score(X_train, y_train))
        test_accuracy.append(clf.score(X_test, y_test))

    plt.plot(neighbors_settting, training_accuracy, lable="training accuracy")
    plt.plot(neighbors_settting, test_accuracy, label='test accuracy')
    plt.xlabel('accuracy')
    plt.ylabel('n_neighbor')
    plt.legend()
    plt.show()


def in_26():
    from sklearn.linear_model import LinearRegression
    x,y=mglearn.datasets.make_wave(n_samples=60)
    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42)
    lr=LinearRegression().fit(x_train,y_train)
    print('coef_:{}'.format(lr.coef_))
    print('intercept_:{}'.format(lr.intercept_))
    print('train score:{}'.format(lr.score(x_train,y_train)))
    print('test score:{}'.format(lr.score(x_test,y_test)))

def in_31():
    from sklearn.linear_model import Ridge
    x, y = mglearn.datasets.make_wave(n_samples=60)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
    ridge=Ridge(alpha=0.1).fit(x_train,y_train)
    print('coef_:{}'.format(ridge.coef_))
    print('intercept_:{}'.format(ridge.intercept_))
    print('train score:{}'.format(ridge.score(x_train, y_train)))
    print('test score:{}'.format(ridge.score(x_test, y_test)))

def in_36():
    from sklearn.linear_model import Lasso
    x, y = mglearn.datasets.make_wave(n_samples=60)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
    lr = Lasso(alpha=0.1,max_iter=1000).fit(x_train, y_train)
    print('coef_:{}'.format(lr.coef_))
    print('intercept_:{}'.format(lr.intercept_))
    print('train score:{}'.format(lr.score(x_train, y_train)))
    print('test score:{}'.format(lr.score(x_test, y_test)))
    print('number of features:{}'.format(np.sum(lr.coef_!=0)))

def in_42():
    from sklearn.datasets import  load_breast_cancer
    from sklearn.linear_model import LogisticRegression
    cancer=load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target,stratify=cancer.target, random_state=42)
    lr=LogisticRegression(C=100).fit(x_train,y_train)
    print('coef_:{}'.format(lr.coef_))
    print('intercept_:{}'.format(lr.intercept_))
    print('train score:{}'.format(lr.score(x_train, y_train)))
    print('test score:{}'.format(lr.score(x_test, y_test)))
    print('number of features:{}'.format(np.sum(lr.coef_ != 0)))
    print('features:\n{}'.format(cancer.feature_names[:30]))
def in_47():
    from  sklearn.datasets import make_blobs
    x,y=make_blobs(random_state=42)
    mglearn.discrete_scatter(x[:,0],x[:,1],y)
    plt.xlabel('frature 0')
    plt.ylabel('feature 1')
    plt.legend(['classs 0','class 1','class 2'])
    plt.show()


def in_48():
    from sklearn.datasets import make_blobs

    x, y = make_blobs(random_state=42)

    from sklearn.svm import LinearSVC
    linear_svm=LinearSVC().fit(x,y)
    print('coef_:{}'.format(linear_svm.coef_))
    print('intercept_:{}'.format(linear_svm.intercept_))



def in_58():
    from sklearn.tree import DecisionTreeClassifier
    cancer=load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target,
                                                        random_state=42)
    tree=DecisionTreeClassifier(max_depth=4,random_state=0)
    tree.fit(x_train,y_train)
    print('train score:{}'.format(tree.score(x_train, y_train)))
    print('test score:{}'.format(tree.score(x_test, y_test)))
    from sklearn.tree import export_graphviz
    #export_graphviz(tree,out_file=r'C:\Users\Natsu\Desktop\tree.dot',class_names=['malignant','benign'],feature_names=cancer.feature_names,impurity=False,filled=False)
    print('feature importance:{}'.format(tree.feature_importances_))
    n=cancer.data.shape[1]
    plt.barh(range(n),tree.feature_importances_,align='center')
    plt.yticks(np.arange(n),cancer.feature_names)
    plt.xlabel('feature importance')
    plt.ylabel('feature')
    plt.show()

def in_66():

    from sklearn.tree import  DecisionTreeRegressor
    x, y = mglearn.datasets.make_wave(n_samples=60)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

    ridge =DecisionTreeRegressor(max_depth=3).fit(x_train, y_train)

    print('train score:{}'.format(ridge.score(x_train, y_train)))
    print('test score:{}'.format(ridge.score(x_test, y_test)))

def in_68():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_moons
    x,y=make_moons(n_samples=100,noise=0.25,random_state=3)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
    forest=RandomForestClassifier(random_state=2,max_depth=5,n_estimators=100)
    forest.fit(x_train,y_train)
    print(forest.score(x_train,y_train))
    print(forest.score(x_test,y_test))


def in_69():
    from sklearn.ensemble import GradientBoostingClassifier

    from sklearn.datasets import make_moons
    x, y = make_moons(n_samples=100, noise=0.25, random_state=3)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
    forest = GradientBoostingClassifier(random_state=2, max_depth=3, n_estimators=100,learning_rate=0.1)
    forest.fit(x_train, y_train)
    print(forest.score(x_train, y_train))
    print(forest.score(x_test, y_test))


def in_91():
    line=np.linspace(-30, 300, 100)
    plt.plot(line, np.tanh(line),label= 'tanh')
    plt.plot(line,np.maximum(line,0),label='relu')
    plt.xlabel('x')
    plt.ylabel('relu(x),tanh(x)')
    plt.legend()
    plt.show()

def in_93():
    from sklearn.neural_network import MLPClassifier
    from sklearn.datasets import make_moons
    x, y = make_moons(n_samples=100, noise=0.25, random_state=3)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
    mlp=MLPClassifier(solver='lbfgs',random_state=0,hidden_layer_sizes=[10,10,10],activation='relu')
    mlp.fit(x_train,y_train)
    mglearn.plots.plot_2d_separator(mlp,x_train,fill=True,alpha=0.3)
    mglearn.discrete_scatter(x_train[:,0],x_train[:,1],y_train)
    plt.xlabel('feature 0')
    plt.ylabel('feature 1')
    print(mlp.score(x_train,y_train))
    print(mlp.score(x_test,y_test))



    plt.show()

def in_101():
    from sklearn.neural_network import MLPClassifier
    cancer=load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

    mean=x_train.mean(axis=0)
    std=x_train.std(axis=0)
    x1=(x_train-mean)/std
    x2=(x_test-mean)/std
    mlp=MLPClassifier(random_state=0,max_iter=1000,alpha=0.1)
    mlp.fit(x1,y_train)
    print(mlp.score(x1,y_train))
    print(mlp.score(x2,y_test))
    print(mlp.hidden_layer_sizes)
    print(mlp.predict_proba(x_test[:10]))

def in_101():
    from sklearn.datasets import load_iris
    from sklearn.ensemble import GradientBoostingClassifier
    iris=load_iris()
    x_train, x_test, y_train, y_test = train_test_split( iris.data,  iris.target, random_state=42)
    gbrt=GradientBoostingClassifier(learning_rate=0.1,random_state=0)
    gbrt.fit(x_train,y_train)

    print(gbrt.decision_function(x_test[0:5]))
    print(gbrt.predict_proba(x_test[0:5]))
    print(y_train[0])
    for i in y_train[0:5]:
        print(iris.target_names[i])







