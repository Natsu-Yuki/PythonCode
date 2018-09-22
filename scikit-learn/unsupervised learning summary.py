###预处理与缩放
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
mglearn.plots.plot_scaling()
###降维
from sklearn.decomposition import PCA
mglearn.plots.plot_pca_illustration()
pca=PCA(n_components=,whiten=)

from sklearn.decomposition import NMF
mglearn.plots.plot_nmf_illustration()
nmf=NMF(n_components=)
###聚类
from sklearn.cluster import KMeans
mglearn.plots.plot_kmeans_algorithm()
k=KMeans(n_clusters=)

from sklearn.cluster import AgglomerativeClustering
mglearn.plots.plot_agglomerative_algorithm()
agg=AgglomerativeClustering(n_clusters=)

from sklearn.cluster import DBSCAN
mglearn.plots.plot_dbscan()
db=DBSCAN(min_samples=,eps=)

def in64():   #树状图
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
###评估
#有真实值
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import accuracy_score
#无真实值
from sklearn.metrics.cluster import  silhouette_score