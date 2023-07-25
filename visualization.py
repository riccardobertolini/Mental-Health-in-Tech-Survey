import matplotlib.pyplot as plt
import seaborn as sns

def visualize_distributions(df_cluster, cluster):
    fig, ax =plt.subplots(1,2)
    sns.countplot(data=df_cluster, x='remote_work', ax=ax[0]).set_title('Cluster ' + str(cluster) + ' remote_work distribution')
    sns.countplot(data=df_cluster, x='mental_health_consequence', ax=ax[1]).set_title('Cluster ' + str(cluster) + ' mental_health consequence distribution')
    fig.show()

