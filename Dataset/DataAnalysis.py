import numpy as np
from Dataset.utils.basic import onehot2vector, hdf5_handler
from Dataset.utils.small_world import network_binary_by_sparsity, local_clustering_coefficient
from Dataset.DataBase import DataBase
from ops.matrix import matrix_significance_difference


def statistical_analysis_on_clustering_coefficient():
    db = DataBase(dataset_list='ABIDE_Initiative')
    sparsities = np.arange(start=0.01, stop=0.11, step=0.01)
    features = ['local clustering coefficient', 'clustering coefficient']
    label = onehot2vector(db.get_data(features='label')['label'])

    LCCs = []
    CCs = []
    for sparsity in sparsities:
        names = [
            'feature/aal90/{:s}/{:.2f}'.format(feature, sparsity) for feature in features]
        datas = db.get_data(features=names)
        LCCs.append(np.expand_dims(
            axis=-1, a=datas['feature/aal90/local clustering coefficient/{:.2f}'.format(sparsity)]))
        CCs.append(np.expand_dims(
            axis=-1, a=datas['feature/aal90/clustering coefficient/{:.2f}'.format(sparsity)]))
    LCCs = np.concatenate(LCCs, axis=-1)
    CCs = np.concatenate(CCs, axis=-1)
    
    results_LCCs = matrix_significance_difference(matrix=LCCs, label=label)
    results_CCs = matrix_significance_difference(matrix=CCs, label=label)
    pass


if __name__ == "__main__":
    statistical_analysis_on_clustering_coefficient()
