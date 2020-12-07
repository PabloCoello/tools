#importar librerias necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sb


class km():
    def __init__(self, df, features):
        '''
        '''
        self.x_ = df[features]


    def get_km__(self, n_clusters, max_iter):
        '''
        '''
        km = KMeans(n_clusters=n_clusters, max_iter=max_iter)
        return km.fit(self.x_)
        
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
        wcs = [self.get_km__(nclust, max_iter).inertia_
               for nclust in range(1, max_clusters)]
        plt.plot(range(1, max_clusters), wcs)
        plt.show()

    def get_km_plot(self):
        '''
        '''
        pca = PCA(n_components=2)
        pca_x = pca.fit_transform(self.x_)
        pca_x = pd.DataFrame(data=pca_x, columns=['comp_1', 'comp_2'])
        pca_nombres_x = pd.concat([pca_x, pd.Series(self.km.labels_, name='km_labels')], axis=1)

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
    
    #Load test dataframe
    df = sb.load_dataset('iris')
    df.columns
    
    #Instanciate class with df and feature columns
    kmeans = km(df, ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    
    #generate optimal number of clusters plot
    kmeans.get_optimal_nclust(40, 1000)
    
    #Fit desired number of clusters
    kmeans.fit_km(3, 1000)
    
    # Get labels from fit and store it in original df
    kmeans.get_labels()
    df['km_labels'] = kmeans.get_labels()
    
    # Get centroids position
    kmeans.get_centroids()
    
    # Get cluster plot
    kmeans.get_km_plot()


'''
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
'''