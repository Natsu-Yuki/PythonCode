import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mglearn
from sklearn.model_selection import train_test_split


def in2():
    mglearn.plots.plot_scaling()
    plt.show()


def in3():
    from sklearn.preprocessing import MinMaxScaler
    scaled = MinMaxScaler()
    scaled.fit(x_train)
    x_train_scaled = scaled.transform(x_train)

    print(x_train_scaled.max(axis=0))
    print(x_train_scaled.min(axis=0))

    scaled.fit(x_test)
    x_test_scaled = scaled.transform(x_test)
    print(x_test_scaled.max(axis=0))
    print(x_test_scaled.min(axis=0))


def in10():
    from sklearn.svm import SVC
    from sklearn.datasets import load_breast_cancer
    cancer = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target,
                                                        random_state=0)
    svm = SVC(C=100)
    svm.fit(x_train, y_train)
    print(svm.score(x_test, y_test))

    from sklearn.preprocessing import MinMaxScaler
    scaled = MinMaxScaler()
    scaled.fit(x_train)
    x_train_scaled = scaled.transform(x_train)
    x_test_scaled = scaled.transform(x_test)
    svm.fit(x_train_scaled, y_train)
    print(svm.score(x_test_scaled, y_test))

    from sklearn.preprocessing import StandardScaler
    ssc = StandardScaler()
    ssc.fit(x_train)
    svm.fit(ssc.transform(x_train), y_train)
    print(svm.score(ssc.transform(x_test), y_test))


def in13():
    mglearn.plots.plot_pca_illustration()
    plt.show()


def in15():
    from sklearn.datasets import load_breast_cancer
    cancer = load_breast_cancer()
    from sklearn.preprocessing import StandardScaler
    scaled = StandardScaler()
    scaled.fit(cancer.data)
    x_sclaed = scaled.transform(cancer.data)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(x_sclaed)
    x_pca = pca.transform(x_sclaed)

    plt.figure(figsize=(8, 8))
    mglearn.discrete_scatter(x_pca[:, 0], x_pca[:, 1], cancer.target)
    plt.legend(cancer.target_names, loc='best')
    plt.xlabel('first principal component')
    plt.ylabel('second principal component')
    # plt.show()
    print(pca.components_)


def in21():
    from sklearn.datasets import fetch_lfw_people
    people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
    mask = np.zeros(people.target.shape, dtype=np.bool)
    for target in np.unique(people.target):
        mask[np.where(people.target == target)[0][:50]] = 1
    x_people = people.data[mask]
    y_people = people.target[mask]
    x_people = x_people / 255
    from sklearn.neighbors import KNeighborsClassifier
    x_train, x_test, y_train, y_test = train_test_split(x_people, y_people, stratify=y_people, random_state=0)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=100, whiten=True, random_state=0)
    pca.fit(x_train)
    x_train_pca = pca.transform(x_train)
    x_test_pca = pca.transform(x_test)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(x_train_pca, y_train)
    print(knn.score(x_test_pca, y_test))


def in36():
    from sklearn.datasets import fetch_lfw_people
    people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
    mask = np.zeros(people.target.shape, dtype=np.bool)
    for target in np.unique(people.target):
        mask[np.where(people.target == target)[0][:50]] = 1
    x_people = people.data[mask]
    y_people = people.target[mask]
    x_people = x_people / 255
    from sklearn.neighbors import KNeighborsClassifier
    x_train, x_test, y_train, y_test = train_test_split(x_people, y_people, stratify=y_people, random_state=0)
    from sklearn.decomposition import NMF
    nmf = NMF(n_components=15, random_state=0)
    nmf.fit(x_train)
    x_train_nmf = nmf.transform(x_train)
    x_test_nmf = nmf.transform(x_test)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(x_train_nmf, y_train)
    print(knn.score(x_test_nmf, y_test))


def in48():
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    x, y = make_blobs(random_state=1)
    kmean = KMeans(n_clusters=3)
    kmean.fit(x)

    mglearn.discrete_scatter(x[:, 0], x[:, 1], kmean.predict(x), markers='o')
    mglearn.discrete_scatter(kmean.cluster_centers_[:, 0], kmean.cluster_centers_[:, 1], range(3),
                             markers='^', markeredgewidth=6

                             )

    plt.legend(['cluster 0', 'cluster 1', 'cluster 2'])
    plt.show()


def in59():
    from sklearn.datasets import make_moons
    x, y = make_moons(n_samples=200, noise=0.05, random_state=0)
    from sklearn.cluster import KMeans
    kmean = KMeans(n_clusters=10, random_state=0)
    kmean.fit(x)
    y_pre = kmean.labels_

    plt.scatter(x[:, 0], x[:, 1], c=y_pre, s=60, cmap='Paired')

    plt.scatter(kmean.cluster_centers_[:, 0], kmean.cluster_centers_[:, 1], s=60,
                marker='^', c=range(kmean.n_clusters), linewidths=12, cmap='Paired'
                )

    plt.xlabel('feature 0')
    plt.ylabel('feature 1')



def in62():
    from sklearn.datasets import make_blobs
    from sklearn.cluster import AgglomerativeClustering
    x,y=make_blobs(random_state=1)
    agg=AgglomerativeClustering(n_clusters=3)
    ass=agg.fit(x)
    mglearn.discrete_scatter(x[:, 0], x[:, 1],agg.fit_predict(x),markers='o')
    plt.legend(['cluster 0', 'cluster 1', 'cluster 2'])
    plt.show()


def in64():
    from scipy.cluster.hierarchy import dendrogram,ward
    from sklearn.datasets import make_blobs
    x,y=make_blobs(random_state=0,n_samples=24)
    link=ward(x)
    dendrogram(link)
    ax=plt.gca()
    bounds=ax.get_xbound()
    ax.plot(bounds,[13,13],'--',c='k')
    ax.plot(bounds,[8,8],'--',c='k')
    ax.text(bounds[1],13,'two',va='center')
    ax.text(bounds[1], 8,'three', va='center')
    plt.xlabel('sample index')
    plt.ylabel('cluster distance')
    plt.show()


def in66():
    from sklearn.cluster import DBSCAN
    from sklearn.datasets import make_blobs
    x,y=make_blobs(random_state=1,n_samples=36)
    db=DBSCAN(min_samples=2,eps=1)
    c=db.fit_predict(x)
    plt.scatter(x[:, 0], x[:, 1], c=c, cmap=mglearn.cm2, s=60)
    plt.show()




def in67():
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import make_moons
    from sklearn.cluster import DBSCAN
    x,y=make_moons(n_samples=200,random_state=0,noise=0.05)
    scaler=StandardScaler()
    scaler.fit(x)
    x_scaled=scaler.transform(x)

    db=DBSCAN(min_samples=5,eps=0.5)
    clu=db.fit_predict(x_scaled)

    plt.scatter( x_scaled[:, 0],  x_scaled[:, 1],c=clu,cmap=mglearn.cm2,s=60)
    print(db.fit(x))
    #plt.show()

    from sklearn.metrics.cluster import adjusted_rand_score
    from sklearn.metrics.cluster import normalized_mutual_info_score
    from sklearn.metrics import accuracy_score
    print(adjusted_rand_score(clu,y))
    print(normalized_mutual_info_score(clu,y))
    print(accuracy_score(clu,y))


def in70():
    from sklearn.datasets import make_moons
    from sklearn.metrics.cluster import silhouette_score
    x,y=make_moons(n_samples=200,noise=0.05,random_state=0)
    from sklearn.preprocessing import StandardScaler
    std=StandardScaler()
    std.fit(x)
    x_scaled=std.transform(x)

    from sklearn.cluster import KMeans

    from sklearn.cluster import AgglomerativeClustering

    from sklearn.cluster import DBSCAN

    fig,axer=plt.subplots(1,3,figsize=(15,3))
    axer[0].scatter(x_scaled[:, 0], x_scaled[:, 1], c=KMeans().fit_predict(x_scaled), cmap=mglearn.cm2, s=60)
    axer[0].set_title('KMeans:{}'.format(silhouette_score(x_scaled,KMeans().fit_predict(x_scaled))))

    axer[1].scatter(x_scaled[:, 0], x_scaled[:, 1], c=AgglomerativeClustering().fit_predict(x_scaled), cmap=mglearn.cm2, s=60)
    axer[1].set_title('AgglomerativeClustering:{}'.format(silhouette_score(x_scaled,AgglomerativeClustering().fit_predict(x_scaled))))

    axer[2].scatter(x_scaled[:, 0], x_scaled[:, 1], c=DBSCAN().fit_predict(x_scaled), cmap=mglearn.cm2, s=60)
    axer[2].set_title('DBSCAN:{}'.format(silhouette_score(x_scaled, DBSCAN().fit_predict(x_scaled))))
    plt.legend(['feature 0','feature 1'])
    plt.show()


def in71():
    from sklearn.decomposition import PCA
    pca=PCA(n_components=100)

    from sklearn.datasets import fetch_lfw_people
    people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
    mask = np.zeros(people.target.shape, dtype=np.bool)
    for target in np.unique(people.target):
        mask[np.where(people.target == target)[0][:50]] = 1
    x_people = people.data[mask]
    y_people = people.target[mask]
    x_people = x_people / 255

    pca.fit_transform(x_people)
    x_pca=pca.fit_transform(x_people)

    from sklearn.cluster import  DBSCAN
    db=DBSCAN(min_samples=3,eps=15)
    labels=db.fit_predict(x_pca)
    print(np.unique(labels))
    print(np.bincount(labels+1))
    from sklearn.cluster import KMeans
    km=KMeans(n_clusters=10,random_state=0)
    km.fit_transform(x_pca)
    mglearn.plots.plot_kmeans_faces(km,pca,x_pca,x_people,y_people,people.target_names)
    plt.show()


in71()


