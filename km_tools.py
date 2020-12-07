#importar librerias necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cufflinks as cf
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


class km():
    def __init__(df, features):
        '''
        '''
        self.x_ = df[features]

    def fit_km(self, n_clusters, max_iter):
        '''
        '''
        km = KMeans(n_clusters=n_clusters, max_iter=max_iter)
        self.km = km.fit(self.x_)

    def get_labels(self):
        '''
        '''
        return self.km.labels_

    def get_centroids(self):
        '''
        '''
        return self.km.cluster_centers_

    def get_optimal_nclust(self, max_clusters, max_iter):
        '''
        '''
        wcs = [self.fit_km(nclust, max_iter)
               for nclust in range(1, max_clusters)]
        plt.plot(range(1, max_clusters), wcs)
        plt.show()

    def get_km_plot(self):
        '''
        '''
        pca = PCA(n_components=2)
        pca_x = pca.fit_transform(self.x_)
        pca_x = pd.DataFrame(data=pca_x, columns=['comp_1', 'comp_2'])
        pca_nombres_x = pd.concat([pca_x, self.km.labels_, axis=1])

        fig = plt.figure(1, figsize=(7, 7))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('comp_1', fontsize=15)
        ax.set_ylabel('comp_2', fontsize=15)
        ax.set_title('PCA', fontsize=15)
        color_theme = np.array(['blue', 'orange', 'green'])
        ax.scatter(x=pca_nombres_x.comp_1, y=pca_nombres_x.comp_2,
                   c=color_theme[pca_nombres_x.km_labels], s=50)
        plt.show()

    def perform_km_multiple_analysis(self):
        '''
        '''
        return

if __name__ == '__main__':

#GRÁFICA CON EL PUNTO DE CODO DEL Nº DE CULSTERS
Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
kmeans
score = [kmeans[i].fit(X).score(X) for i in range(len(kmeans))]
score
plt.plot(Nc,score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()

#APLICAR K-MEANS Y MOSTRAR CENTROIDES
kmeans = KMeans(n_clusters=3).fit(X)
centroids = kmeans.cluster_centers_
print(centroids)

def gen_matrix(niter, X):
    toret = []
    for i in range(niter):
        kmeans = KMeans(3).fit(X)
        toret.append(kmeans.labels_)
       return toret

def get_mean(lista):
    return lista.mean()


def plot_cluster(gdf, cluster):
    gdf[gdf.cluster==cluster]['municipio'].plot()

for i in range(3):
    matrix = gen_matrix(1, X)

    matrix = np.matrix(matrix).T
    res = np.apply_along_axis(get_mean, axis=1, arr=matrix)
    res = res.tolist()

    res = matrix[0].tolist()
    df['cluster'] = res
    df.head()
