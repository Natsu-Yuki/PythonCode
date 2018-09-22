from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np




iris_dataset = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
X_new =np.array([[5,2.9,1,0.2]])
prediction=knn.predict(X_new)
y_pre=knn.predict(X_test)



def acquaintance_data():

    print("Keys of isir_dataset:\n{}".format(iris_dataset.keys()) + "\n\n.........")

    print("Target names of isir_dataset:\n{}".format(iris_dataset['target_names']) + "\n\n.........")

    print("Feature names of isir_dataset:\n{}".format(iris_dataset['feature_names']) + "\n\n.........")

    print("Data of isir_dataset:\n{}".format(iris_dataset['data'][:5]) + "\n\n.........")

    print("Target of isir_dataset:\n{}".format(iris_dataset['target'][:5]) + "\n\n.........")

def train_test_data():

    print("X_train shape:{}".format(X_train.shape))

    print("X_test shape:{}".format(X_test.shape))

    print("y_train shape:{}".format(y_train.shape))

    print("y_test shape:{}".format(y_test.shape))

def scatter_plot():
    iris_dataframe=pd.DataFrame(X_train,columns=iris_dataset['feature_names'])
    grr=pd.scatter_matrix(iris_dataframe,c=y_train,figsize=(15,15),marker='o',
                          hist_kwds={'bins':20},s=60,alpha=0.8
                          )

def main():
    print('\n')
    #print(knn.fit(X_train,y_train))
    print("Prediction :{}".format(prediction))
    print("Prediction target name:{}".format(iris_dataset['target_names'][prediction]))

    print("Test set preditions:{}\n".format(y_pre))
    print("Test set score:{:.2f}".format(np.mean(y_pre==y_test)))
    print("Test set score:{:.2f}".format(knn.score(X_test,y_test)))


main()



