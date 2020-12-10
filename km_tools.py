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
        self.clusters = {}
        self.df = df

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
        pca_nombres_x = pd.concat(
            [pca_x, pd.Series(self.km.labels_, name='km_labels')], axis=1)

        fig = plt.figure(1, figsize=(7, 7))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('comp_1', fontsize=15)
        ax.set_ylabel('comp_2', fontsize=15)
        ax.set_title('PCA', fontsize=15)
        color_theme = np.array(['blue', 'orange', 'green'])
        ax.scatter(x=pca_nombres_x.comp_1, y=pca_nombres_x.comp_2,
                   c=color_theme[pca_nombres_x.km_labels], s=50)
        plt.show()

    def perform_km_multiple_analysis(self, n_clust, mc_iterations):
        '''
        '''
        matrix = self.get_cluster_matrix(n_clust, mc_iterations)
        for cluster in range(matrix.max()+1):
            self.df['cluster_'+str(cluster)] = np.asarray((matrix ==
                                                           cluster).sum(0)).reshape(-1)
        return self.df

    def get_cluster_matrix(self, n_clust, mc_iterations):
        '''
        '''
        toret = []
        clusters = {}
        for i in range(mc_iterations):
            self.fit_km(n_clust, 1000)
            self.identify_clusters(n_clust=n_clust)
            current_pair = self.get_current_pair(cent=self.get_centroids(),
                                                 labels=self.get_labels())
            toret.append(self.assign_labels(current_pair))
        return np.matrix(toret)

    def get_current_pair(self, cent, labels):
        '''
        '''
        current_pair = [(elem, cent[elem]) for elem in labels]
        return current_pair

    def identify_clusters(self, n_clust):
        '''
        '''
        for cluster in range(n_clust):
            if len(self.clusters.values()) == 0:
                self.clusters[cluster] = self.get_centroids()[cluster]

            logic = [(self.get_centroids()[cluster] != value).all()
                     for value in self.clusters.values()]
            if all(logic):
                self.clusters[np.max(list(self.clusters.keys())) +
                              1] = self.get_centroids()[cluster]

    def assign_labels(self, current_pair):
        '''
        '''
        toret = []
        for pair in current_pair:
            label = [key for key, value in self.clusters.items() if (
                value == pair[1]).all()][0]
            toret.append(label)
        return toret


if __name__ == '__main__':

    #Load test dataframe
    df = sb.load_dataset('iris')
    df.columns

    #Instanciate class with df and feature columns
    kmeans = km(df, ['sepal_length', 'sepal_width',
                     'petal_length', 'petal_width'])

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

    df = kmeans.perform_km_multiple_analysis(4, 10)
    kmeans.clusters
