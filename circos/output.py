import numpy as np

from AAL.ROI import load_roi_info, load_parcellations
from ops.matrix import matrix_sort


def get_karyotype(output_path: str = None):
    parcellations = load_parcellations()

    for index, parce in enumerate(parcellations):
        print(parce.get_chr(color_index=index + 1))

    bands_list = []
    for parce in parcellations:
        bands_list.extend(parce.get_bands_list())

    for band_str in bands_list:
        print(band_str)


def get_label(output_path: str = None):
    parcellations = load_parcellations()

    label_list = []
    for parce in parcellations:
        label_list.extend(parce.get_label_list())

    for label_str in label_list:
        print(label_str)


def get_links(matrix: np.ndarray = None,
              edges: list = None,
              output_path: str = None,
              node_num: int = 90,
              top: int = None,
              if_symmetric: bool = None):
    if matrix is None:
        matrix = np.random.normal(size=[node_num, node_num])

    # Normalization to [-6, 6], [1, 11] corresponding to thickness and colormap
    data_min = np.min(matrix)
    data_max = np.max(matrix)
    max_abs = np.max((np.abs(data_min), np.abs(data_max)))
    norm_matrix_thickness = np.round(matrix / max_abs * 6)
    norm_matrix_colormap = np.round(-matrix / max_abs * 5 + 6)

    shape = np.shape(matrix)
    assert len(shape) == 2, 'The rank of input matrix must be to but get {:d}.'.format(
        len(shape))
    assert shape[0] == shape[1], 'The input matrix must be a square matrix but get shape {:}'.format(
        shape)

    parcellations = load_parcellations()

    rois_dict = {}
    for parce in parcellations:
        for roi in parce.rois:
            rois_dict[roi.index_AAL] = roi

    link_index = 0
    file = open(output_path, 'w+')

    if edges is None:
        if top is not None:
            edges = [element['coordinate']
                     for element in matrix_sort(matrix=matrix, top=top, if_symmetric=if_symmetric).values()]
        else:
            indices = np.triu_indices(shape[0], 1)
            edges = [[i, j] for i, j in zip(indices[0], indices[1])]

    for edge in edges:
        i, j = edge
        if matrix[i, j] == 0:
            continue

        band_width = 5000
        # ROI 1
        roi1 = rois_dict[i + 1]
        name1 = '{:s}{:s}'.format(roi1.parcellation, roi1.hemisphere)
        position1 = int((2 * roi1.index_of_parcellation + 1) / 2 * 1e6)

        # ROI 2
        roi2 = rois_dict[j + 1]
        name2 = '{:s}{:s}'.format(roi2.parcellation, roi2.hemisphere)
        position2 = int((2 * roi2.index_of_parcellation + 1) / 2 * 1e6)

        str = '{:s} {:d} {:d} {:s} {:d} {:d} ' \
            'color=rdbu-11-div-{:d},thickness={:d},' \
            'z={:f}\n'.format(name1,
                              position1 - band_width,
                              position1 + band_width,
                              name2,
                              position2 - band_width,
                              position2 + band_width,
                              int(norm_matrix_colormap[i, j]),
                              np.abs(int(norm_matrix_thickness[i, j])),
                              np.abs(norm_matrix_thickness[i, j])
                              )

        if output_path is None:
            print(str)
        else:
            file.write(str)

        link_index += 1

    if output_path is not None:
        file_name = output_path.split('.')[-2]
        dir_path = 'brain\\aal\\CNNSmallWorld\\new'
        conf_path = dir_path + '\\' + output_path

        print('bin\\circos -conf {conf_path} -outputdir {file_name}.conf -outputfile {file_name}.png'.format(
            conf_path=conf_path, dir_path=dir_path, file_name=file_name))
        file.close()
