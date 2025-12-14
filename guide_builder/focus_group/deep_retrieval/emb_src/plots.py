import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_embeddings_2d(embeddings, query_embeddings=None, labels=None, idxs=None):
    """
    Reduce dimensionality of embeddings and plot them on a 2D scatter plot.

    Args:
        embeddings (np.ndarray): Encoded sentences with shape [seq_length, vector_size].
        labels (list, optional): Labels for each point in the plot (True/False). Defaults to None.
        idxs (list, optional): Indices for each point in the plot. Defaults to None.
    """
    # Reduce dimensionality to 2D using PCA
    pca = PCA(n_components=2)
    pca = pca.fit(embeddings)
    reduced_embeddings = pca.transform(embeddings)

    # Determine colors based on labels
    colors = ['green' if label else 'red' for label in tf.reduce_sum(labels, axis=-1).numpy()] if labels is not None else 'blue'

    # Create scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=colors, alpha=0.7)
    if query_embeddings is not None:
        reduced_query = pca.transform(query_embeddings)
        plt.scatter(reduced_query[:, 0], reduced_query[:, 1], c='black', marker='x', s=100, label='Query', alpha=0.7)

    # Add indices as text if provided
    if idxs:
        for i, idx in enumerate(idxs):
            plt.text(reduced_embeddings[i, 0], reduced_embeddings[i, 1], str(idx), fontsize=9)

    plt.title("2D Scatter Plot of Embeddings")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.show()