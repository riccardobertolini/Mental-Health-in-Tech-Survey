import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import chi2_contingency


def perform_clustering(df):
    # Convert categorical variables to integers
    df_encoded = df.apply(lambda x: pd.factorize(x)[0])

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_encoded)

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(df_scaled)
    # The PCA component labels are defined here
    principalDf = pd.DataFrame(data=principalComponents, columns=['Socio-Economic Profile', 'Demographic Characteristics'])

    kmeans = KMeans(n_clusters=3)
    clusters = kmeans.fit_predict(principalDf)
    df['cluster'] = clusters

    plt.figure(figsize=(8, 6))
    plt.scatter(principalComponents[:, 0], principalComponents[:, 1], c=clusters, cmap='viridis')
    plt.xlabel('Socio-Economic Profile')
    plt.ylabel('Demographic Characteristics')
    plt.title('Clusters')

    plt.show()

    for i in range(3):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        df[df['cluster'] == i]['remote_work'].value_counts().plot(kind='bar',
                                                                  title=f'Cluster {i} Remote Work Distribution')
        plt.subplot(1, 2, 2)
        df[df['cluster'] == i]['mental_health_consequence'].value_counts().plot(kind='bar',
                                                                                title=f'Cluster {i} Mental Health Consequence Distribution')
        plt.tight_layout()
        plt.show()

    for i in range(3):
        table = pd.crosstab(df[df['cluster'] == i]['remote_work'], df[df['cluster'] == i]['mental_health_consequence'])
        chi2, p, dof, ex = chi2_contingency(table)
        print(f'Cluster {i} contingency table:')
        print(table)
        print(f'Chi-square Test for Cluster {i}: Chi2={chi2}, p={p}')

        plt.figure(figsize=(8, 6))
        sns.heatmap(table, annot=True, fmt='d', cmap='viridis')
        plt.title(f'Cluster {i} Contingency Table')
        plt.show()

    return clusters
