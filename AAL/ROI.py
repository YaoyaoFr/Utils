import os
import numpy as np

parcellation_roi = {
    'FRONTAL':
        ['PreCG', 'SFGdor', 'ORBsup', 'MFG', 'ORBmid', 'IFGoperc', 'IFGtriang', 'ORBinf', 'ROL', 'SMA', 'SFGmed',
         'ORBmed', 'REC', 'PCL'],
    'PARIETAL':
        ['PoCG', 'SPG', 'IPL', 'SMG', 'ANG', 'PCUN'],
    'LIMBIC':
        ['OLF', 'INS', 'ACG', 'MCG', 'PCG', 'HIP', 'PHG', 'AMYG'],
    'TEMPORAL':
        ['FFG', 'HES', 'STG', 'TPOsup', 'MTG', 'TPOmid', 'ITG'],
    'OCCIPITAL':
        ['CAL', 'CUN', 'LING', 'SOG', 'MOG', 'IOG'],
    'SUB':
        ['CAU', 'PUT', 'PAL', 'THA'],
}
parcellation_names = ['FRONTAL', 'PARIETAL', 'LIMBIC', 'TEMPORAL', 'OCCIPITAL', 'SUB']
roi_parcellation = {}
for parcel in parcellation_names:
    for roi in parcellation_roi[parcel]:
        roi_parcellation[roi] = parcel


class Parcellation:

    def __init__(self,
                 name: str,
                 hemisphere: str,
                 rois: list,
                 length: int = None,
                 ):
        self.name = name
        self.hemisphere = hemisphere
        self.rois = rois
        if length is None:
            self.length = int(1e6 * len(rois))
        else:
            self.length = length

    def get_chr(self, color_index):
        name = '{:s}{:s}'.format(self.name,
                                 self.hemisphere)
        chr_str = 'chr - {:s} {:s} 0 {:d} chr{:d}'.format(name,
                                                          self.name,
                                                          self.length,
                                                          color_index)
        return chr_str

    def get_bands_list(self):
        band_list = []
        for index, roi in enumerate(self.rois):
            band_roi_str = roi.get_band_str(index=index)
            band_list.append(band_roi_str)
        return band_list

    def get_label_list(self):
        label_list = []
        for index, roi in enumerate(self.rois):
            label_roi_str = roi.get_label_str(index=index)
            label_list.append(label_roi_str)
        return label_list


class ROI:

    def __init__(self,
                 roi_name: str,
                 hemisphere: str,
                 abbreviation: str,
                 parcellation: str,
                 index_AAL: str or int,
                 ):
        self.roi_name = roi_name
        self.hemisphere = hemisphere
        self.abbreviation = abbreviation
        self.parcellation = parcellation

        if isinstance(index_AAL, str):
            self.index_AAL = int(index_AAL)
        else:
            self.index_AAL = index_AAL


class ROI2:

    def __init__(self,
                 index: str or int,
                 name: str,
                 abrv: str,
                 hemisphere: str,
                 parcellation: str):
        self.index = index
        self.name = name
        self.abrv = abrv
        self.hemisphere = hemisphere
        self.parcellation = parcellation
        self.index_of_parcellation = 0

    def get_info_str(self):
        info_str = 'Index: {:d}\t' \
                   'name: {:s}\r\n\t' \
                   'abrv: {:s}\t' \
                   'parcellation:{:s}'.format(self.index,
                                              self.name,
                                              self.abrv,
                                              self.parcellation)
        return info_str

    def get_band_str(self, index):
        name = '{:s}{:s}'.format(self.abrv, self.hemisphere)
        start_position = int(1e6 * index)
        end_position = int(1e6 * (index + 1))
        parcellation = '{:s}{:s}'.format(self.parcellation, self.hemisphere)
        band_str = 'band {:s} {:s} {:s} {:d} {:d} {:s}'.format(parcellation,
                                                               name, name,
                                                               start_position,
                                                               end_position,
                                                               name,
                                                               )
        return band_str

    def get_label_str(self, index):
        name = '{:s}'.format(self.abrv)
        start_position = int(1e6 * index)
        end_position = int(1e6 * (index + 1))
        parcellation = '{:s}{:s}'.format(self.parcellation, self.hemisphere)
        band_str = '{:s} {:d} {:d} {:s}'.format(parcellation,
                                                start_position,
                                                end_position,
                                                name,
                                                )
        return band_str


def load_rois(file_path: str = None):
    if file_path is None:
        file_path = '/'.join(__file__.split('/')[:-1]) + '/roi_names.txt'
    file = open(file_path)

    rois = []
    infos = file.read().split(' ')
    for info in infos:
        try:
            index = int(info)
            name = ''
        except:
            if '-' in info:
                abrv, hemisphere = info.split('-')
                rois.append(ROI(roi_name=name,
                                hemisphere=hemisphere,
                                abbreviation=abrv,
                                parcellation=roi_parcellation[abrv],
                                index_AAL=index,
                                )
                            )
            else:
                name += ' {:s}'.format(info)
            continue

    return rois


def load_roi_info(file_path: str = None):
    if file_path is None:
        file_path = '/'.join(__file__.split('/')[:-1]) + '/roi_names.txt'
    file = open(file_path)

    rois = []
    infos = file.read().split(' ')
    for info in infos:
        try:
            index = int(info)
            name = ''
        except:
            if '-' in info:
                abrv, hemisphere = info.split('-')
                parcelation = roi_parcellation[abrv]
                rois.append(ROI(index_AAL=index,
                                roi_name=name,
                                abbreviation=abrv,
                                hemisphere=hemisphere,
                                parcellation=parcelation,
                                )
                            )
            else:
                name += ' {:s}'.format(info)
            continue

    rois_dict = {}
    for roi in rois:
        name = '{:s}{:s}'.format(roi.abbreviation, roi.hemisphere)
        assert name not in rois_dict, '{:s} has exist.'.format(name)
        rois_dict[name] = roi

    return rois_dict


def load_parcellations(rois_dict: dict = None):
    if rois_dict is None:
        rois_dict = load_roi_info()

    parcellations = []
    # for i in range(len(parcellation_names)):
    for cor_index, cortical in enumerate(parcellation_names):
        rois = []
        roi_names_temp = parcellation_roi[cortical]

        for index, roi_name in enumerate(roi_names_temp):
            name = '{:s}R'.format(roi_name)
            roi = rois_dict[name]
            roi.index_of_parcellation = index
            rois.append(roi)

        parce = Parcellation(name=parcellation_names[cor_index],
                             hemisphere='R',
                             rois=rois,
                             )
        parcellations.append(parce)

    parcellation_names_reverse = list(parcellation_names)
    parcellation_names_reverse.reverse()
    # for i in np.arange(start=len(parcellation_names) - 1, stop=-1, step=-1):
    for cor_index, cortical in enumerate(parcellation_names_reverse):
        rois = []
        # roi_names_temp = list(parcellation_roi[i])
        roi_names_temp = parcellation_roi[cortical]
        roi_names_temp.reverse()

        for index, roi_name in enumerate(roi_names_temp):
            name = '{:s}L'.format(roi_name)
            roi = rois_dict[name]
            roi.index_of_parcellation = index
            rois.append(roi)

        parce = Parcellation(name=parcellation_names_reverse[cor_index],
                             hemisphere='L',
                             rois=rois,
                             )
        parcellations.append(parce)

    return parcellations
