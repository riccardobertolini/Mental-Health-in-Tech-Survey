import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from clustering import perform_clustering
from visualization import visualize_distributions

def chi_square_test(df_cluster, cluster):
    contingency_table = pd.crosstab(df_cluster['remote_work'], df_cluster['mental_health_consequence'])
    print("Cluster " + str(cluster) + " contingency table:")
    print(contingency_table)
    chi2, p, _, _ = chi2_contingency(contingency_table)
    print("Chi-square Test for Cluster " + str(cluster) + ": Chi2=" + str(chi2) + ", p=" + str(p))

def main():
    df = pd.read_csv('data/survey.csv')

    df_clustered, principal_df = perform_clustering(df)

    # Visualization of Clusters
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=principal_df, x='principal component 1', y='principal component 2', hue=df_clustered['Cluster'])
    plt.title('PCA Clusters')
    plt.show()

    for cluster in set(df_clustered['Cluster']):
        df_cluster = df_clustered[df_clustered['Cluster'] == cluster]
        visualize_distributions(df_cluster, cluster)
        chi_square_test(df_cluster, cluster)

if __name__ == "__main__":
    main()
