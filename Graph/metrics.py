import numpy as np


def local_clustering_coefficient(networks: np.ndarray,
                                 network_axes: list = None):
                                 
    if network_axes is None:
        network_axes = [-2, -1]

    shape = np.shape(networks)
    