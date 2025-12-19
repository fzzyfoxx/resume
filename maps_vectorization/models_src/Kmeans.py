import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans

class ImageClusters:
    """Utility for clustering image pixels and producing cluster masks and centroids.

    This class wraps sklearn KMeans to cluster pixels of a square image and
    provides helpers to compute cluster masks, centroids in image coordinates,
    and to generate smoothed/combined cluster masks using pooling and IoU-based
    similarity.
    """
    def __init__(self, img_size ,n_clusters, n_init):
        """Initialize ImageClusters.

        Args:
            img_size (int): Width and height of the square image in pixels.
            n_clusters (int): Number of clusters to compute with KMeans.
            n_init (int): Number of KMeans initializations (n_init passed to sklearn.KMeans).
        """

        self.n_clusters = n_clusters
        self.img_size = img_size
        self.kmeans = KMeans(n_clusters=n_clusters, n_init=n_init)
        self.coords = self.gen_coords()

    def gen_coords(self,):
        """Generate xy coordinates for each pixel in the image.

        Returns:
            numpy.ndarray: Array of shape (img_size*img_size, 2) where each row is (x, y).
        """
        xs = np.repeat(np.arange(self.img_size)[np.newaxis, :, np.newaxis], self.img_size, axis=0)
        ys = np.transpose(xs, axes=[1,0,2])
        coords = np.reshape(np.concatenate([xs,ys], axis=-1), (-1,2))
        return coords

    def get_clusters(self, x):
        """Run KMeans clustering on an image array.

        Args:
            x (array-like): Image array of shape (img_size, img_size, channels) or a compatible shape.

        Returns:
            tuple: (kmeans_results, clusters)
                - kmeans_results: fitted sklearn KMeans object.
                - clusters: 1-D array of cluster labels for each pixel (length img_size*img_size).
        """
        kmeans_results = self.kmeans.fit(np.reshape((x), (-1,3)))
        clusters = kmeans_results.labels_
        return kmeans_results, clusters
    
    def get_coords_centroids(self, clusters):
        """Compute centroids (mean coordinates) for each cluster.

        Args:
            clusters (array-like): 1-D array of cluster labels for each pixel.

        Returns:
            numpy.ndarray: Array of shape (C, 2) with (x, y) centroid for each cluster.
        """
        C = np.max(clusters)+1

        centroids = []
        for c in range(C):
            mask = np.where(clusters==c, 1, 0)[:,np.newaxis]
            denominator = np.sum(mask)
            mean_sum = np.sum(self.coords*mask, axis=0)
            mean = mean_sum/denominator
            centroids.append(mean)

        return np.stack(centroids, axis=0)
    
    def get_cluster_masks(self, clusters):
        """Convert 1-D cluster labels into binary masks per cluster.

        Args:
            clusters (array-like): 1-D array of cluster labels for each pixel.

        Returns:
            numpy.ndarray: Array of shape (C, img_size, img_size, 1) with binary masks for each cluster.
        """
        C = np.max(clusters)+1

        masks = []
        for c in range(C):
            masks.append(np.reshape(np.where(clusters==c, 1, 0), (self.img_size,self.img_size,1)))

        return np.stack(masks, axis=0)
    
    @staticmethod
    def crossIoU(masks):
        """Compute a pairwise similarity matrix based on IoU for binary masks.

        The returned matrix is scaled from [-1, 1] by mapping IoU values to
        the range [-1, 1] via (I/U - 0.5) * 2.

        Args:
            masks (numpy.ndarray): Array of shape (N, ...) containing binary masks.

        Returns:
            numpy.ndarray: Pairwise similarity matrix of shape (N, N).
        """
        N = masks.shape[0]
        x = np.reshape(masks.copy(), (N, -1))
        a = np.repeat(np.expand_dims(x, axis=1), N, axis=1)
        b = np.repeat(np.expand_dims(x, axis=0), N, axis=0)

        U = np.sum(np.max(np.stack([a,b], axis=-1), axis=-1), axis=-1)
        I = np.sum((a+b)/2, axis=-1)
        return (I/U-0.5)*2
    
    @staticmethod
    def match_idxs_pair(a,b):
        """Check whether two index arrays are identical.

        Args:
            a, b (array-like): Index arrays to compare.

        Returns:
            bool: True if arrays have equal length and identical elements, else False.
        """
        if len(a)==len(b):
            return np.all(a==b)
        else:
            return False

    def concatenate_masks(self, ct, masks, th=0.4):
        '''
            ct: contingency table of shape (N,N) with similarity scores
            masks: clusters masks of shape (N, ...)
            th: threshold for combining masks
        '''
        N = ct.shape[0]

        collections = [np.where(r>th)[0] for r in ct]

        unique_collections = [collections[0]]
        for i, coll in enumerate(collections[1:]):
            if not np.any([self.match_idxs_pair(coll, x) for x in collections[:i+1]]):
                unique_collections.append(coll)

        out_masks = np.stack([np.max(masks[coll], axis=0) for coll in unique_collections], axis=0)

        return out_masks, unique_collections
    
    def gen_smoothed_clusters(self,
                              img,
                              avg_pools=3,
                              avg_pool_ksize=3,
                              avg_pool_th=0.4,
                              max_pools=3,
                              max_pool_ksize=3,
                              max_pool_th=0.7,
                              return_scores=False
                              ):
        """Generate smoothed and combined cluster masks from an input image.

        The method clusters pixels, produces binary masks per cluster, applies
        average pooling to merge similar nearby clusters, computes similarity
        scores, then applies max pooling and a second merge stage.

        Args:
            img (tf.Tensor or array-like): Input image tensor or array.
            avg_pools (int): Number of average-pooling steps to apply.
            avg_pool_ksize (int): Kernel size for average pooling.
            avg_pool_th (float): Similarity threshold for combining after avg pooling.
            max_pools (int): Number of max-pooling steps to apply.
            max_pool_ksize (int): Kernel size for max pooling.
            max_pool_th (float): Similarity threshold for combining after max pooling.
            return_scores (bool): If True, also return the avg and max similarity matrices.

        Returns:
            numpy.ndarray or tuple: If return_scores is False returns combined masks
            of shape (M, img_size, img_size, 1). If True returns (masks, avg_similarity, max_similarity).
        """
        img = img.numpy()
        _, clusters = self.get_clusters(img)
        clusters_masks = self.get_cluster_masks(clusters)

        averaged_masks = tf.cast(tf.constant(clusters_masks), tf.float32)
        for _ in range(avg_pools):
            averaged_masks = tf.nn.avg_pool2d(averaged_masks, ksize=avg_pool_ksize, strides=2, padding='VALID')
        averaged_masks = averaged_masks.numpy()

        avg_similarity_scores = self.crossIoU(averaged_masks)

        cm, collections = self.concatenate_masks(avg_similarity_scores, clusters_masks, th=avg_pool_th)

        pooled_masks = tf.cast(tf.constant(cm), tf.float32)
        for _ in range(max_pools):
            pooled_masks = tf.nn.max_pool2d(pooled_masks, ksize=max_pool_ksize, strides=2, padding='VALID')
        pooled_masks = pooled_masks.numpy()
        pooled_masks.shape

        max_similarity_scores = self.crossIoU(pooled_masks)

        cm, collections = self.concatenate_masks(max_similarity_scores, cm, th=max_pool_th)

        if return_scores:
            return cm, avg_similarity_scores, max_similarity_scores
        return cm