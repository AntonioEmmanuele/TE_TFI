from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor
from multiprocessing import cpu_count, Pool
import numpy as np


def train_tree(dtr : DecisionTreeRegressor, X_train : np.ndarray, y_train : np.ndarray):
    dtr.fit(X_train, y_train)
    # Starmap works with a copy, so return the classifier.
    return dtr

def predict_tree(dtr, X):
    return dtr.predict(X)

class TE_TFI:
    __supported_clusters ={
        "KMeans" : KMeans
    }
    
    def __init__(   self, 
                    cluster_type    : str = "KMeans", 
                    n_clusters      : int = 2, 
                    cluster_cfg     : dict = None,
                    random_state    : int = 42,
                    tree_confs      : list = None):
        # Check the cluster type
        assert cluster_type in TE_TFI.__supported_clusters.keys(), "INSERT A SUPPORTED CLUSTER"
        assert n_clusters > 1, "Invalid number of clusters, please insert a value greater than 1"
        
        # Save input parameters
        self.cluster_type = cluster_type
        self.n_clusters = n_clusters
        self.cluster_cfg = cluster_cfg
        self.random_state = random_state
        self.tree_confs = tree_confs

        # Initialize the cluster
        if cluster_cfg is not None:
            self.cluster = TE_TFI.__supported_clusters[cluster_type](**cluster_cfg, n_clusters = n_clusters, random_state = random_state)
        else:
            self.cluster = TE_TFI.__supported_clusters[cluster_type](n_clusters = n_clusters, random_state = random_state)
        
        # Initialize the trees
        if tree_confs is not None:
            assert len(tree_confs) == n_clusters, "Insert a number of configurations equal to the number of clusters."
            self.trees = [DecisionTreeRegressor(**conf, random_state = random_state) for conf in tree_confs]
        else:
            self.trees = [DecisionTreeRegressor(random_state = random_state) for i in range(0, n_clusters)]
        self.pool = Pool(n_clusters)

    def fit_clust_ts(self, hyst_buffers_cl : np.ndarray, train_wins_tree : np.ndarray, train_target_tree : np.ndarray):
        assert hyst_buffers_cl.ndim == 2, "Cluster hystorical buffer should be a 2D array"
        assert train_wins_tree.ndim == 2, "Tree windows should be a 2D array"
        assert train_target_tree.ndim >= 1, "Target values for a single tree should be at least a 1D array"
        # Fit the clusters, obtaining the trees labels.
        labels = self.cluster.fit_predict(hyst_buffers_cl)
        # Now train the trees.
        args = [[self.trees[i], train_wins_tree[labels == i], train_target_tree[labels == i]] for i in range(0, self.n_clusters)]
        # Starmap works with a copy, so reassign the trees.
        self.trees = self.pool.starmap(train_tree, args)
        
    def predict_clust_ts(self, hyst_buff_cl : np.ndarray, wins_tree : np.ndarray):
        assert hyst_buff_cl.ndim == 2, "Cluster hystorical buffer should be a 2D array"
        assert wins_tree.ndim == 2, "Tree windows should be a 2D array"
        # Initialize value to return 
        to_ret = np.zeros(len(wins_tree))
        # Classify each sample
        ids_tree = self.cluster.predict(hyst_buff_cl)
        # Divide samples per each tree
        args = [[self.trees[i], wins_tree[ids_tree == i]] for i in range(0, self.n_clusters)]
        # Divide the ids for sorting.
        # This vector contains for each tree the set of indexes in the final vector for each sample.
        final_vector_indexes = [np.where(ids_tree == i)[0] for i in range(0, self.n_clusters)]
        # Predict samples.
        preds_per_tree = self.pool.starmap(predict_tree, args)
        # Sort values for each different tree.
        for single_tree_preds, indexes in zip(preds_per_tree, final_vector_indexes):
            to_ret[indexes] = single_tree_preds
        return to_ret
    
    def __str__(self) -> str:
        internal_params     = f"N.ro Clusters: {self.n_clusters}\nC.Type: {self.cluster_type}\nRandomState: {self.random_state}\n"
        tree_confs          = f""
        tree_setted_confs   = f""
        for idx in range(0, self.n_clusters):
            tree_confs += f"Conf Tree {idx}: {self.tree_confs[idx]}\n"
            tree_setted_confs += f"Tree: {idx}\n   MaxDepth: {self.trees[idx].max_depth} MinSamplesSplit: {self.trees[idx].min_samples_split} MinSamplesLeaf: {self.trees[idx].min_samples_leaf} Criterion: {self.trees[idx].criterion}\n"
        return internal_params + tree_confs + tree_setted_confs
    
    def __del__(self):
        self.pool.close()