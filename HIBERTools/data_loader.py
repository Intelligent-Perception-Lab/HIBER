'''
 # @ Author: Zhi Wu
 # @ Create Time: 2022-07-15 00:13:20
 # @ Modified by: Zhi Wu
 # @ Modified time: 2022-07-15 15:09:00
 # @ Description: Dataset class definition.
 '''
import glob
import os
import json
import numpy as np
from itertools import product

class HIBERDataset():
    """HIBER dataset load and visualize object"""

    def __init__(self, root, subsets=['ACTION', 'DARK', 'MULTI', 'OCCLUSION', 'WALK']) -> None:
        """HIBERDataset constructor

        Args:
            root (str): Root path of HIBER dataset directory.
            subsets (List[str], optional): Subsets to be processed. Defaults to ['ACTION', 'DARK', 'MULTI', 'OCCLUSION', 'WALK'].
        """
        super().__init__()
        self.subsets = ['ACTION', 'DARK', 'MULTI', 'OCCLUSION', 'WALK']
        assert set(subsets) <= set(self.subsets), \
            "subsets should in ['ACTION', 'DARK', 'MULTI', 'OCCLUSION', 'WALK']" 
        self.root = root
        self.chosen_subsets = subsets
        self.fpg = 590 # number of frames per group
        self.__statistics__()
        self.datas, self.cates = self.__prepare_data__() # index of data items and categories

    def __statistics__(self):
        """HIBERDataset statistics information, containing group lists of each subset and group lists of detailed subcategories.
        """
        statistics = {
            'WALK':{
                '1':list(range(1, 36)),
                '2':list(range(1, 21)),
                '3':list(range(1, 13)) + [38,],
                '4':list(range(1, 13)),
                '5':list(range(1, 13)),
                '6':list(range(1, 13)),
                '7':list(range(1, 13)),
                '8':list(range(1, 13)),
                '9':list(range(1, 13)),
                '10':list(range(4, 16)),
            },
            'MULTI':{
                '1':list(range(77, 89)),
                '2':list(range(65, 77)),
                '3':list(range(13, 22)) + list(range(59, 62)),
                '4':list(range(13, 25)),
                '5':list(range(13, 25)),
                '6':list(range(13, 25)),
                '7':list(range(13, 25)),
                '8':list(range(13, 25)),
                '9':list(range(13, 25)),
                '10':list(range(16, 28)),
            },
            'ACTION':{
                '1':list(range(37, 62)),
                '2':list(range(45, 65)),
                '3':list(range(47, 59)),
                '4':list(range(49, 61)),
                '5':list(range(49, 61)),
                '6':list(range(49, 63)),
                '7':list(range(49, 61)),
                '8':list(range(49, 61)),
                '9':list(range(49, 61)),
                '10':list(range(52, 64)),
            },
            'OCCLUSION':{
                '1':list(range(62, 68)) + list(range(73, 77)) + list(range(89, 95)),
                '2':list(range(21, 45)) + list(range(77, 89)),
                '3':list(range(22, 38)) + list(range(39, 47)),
                '4':list(range(25, 49)),
                '5':list(range(25, 49)),
                '6':list(range(25, 49)),
                '7':list(range(25, 49)),
                '8':list(range(25, 49)),
                '9':list(range(25, 49)),
                '10':list(range(28, 52)),
            },
            'DARK':{
                '1':list(range(0, 0)),
                '2':list(range(0, 0)),
                '3':list(range(0, 0)),
                '4':list(range(0, 0)),
                '5':list(range(0, 0)),
                '6':list(range(0, 0)),
                '7':list(range(0, 0)),
                '8':list(range(61, 71)),
                '9':list(range(61, 72)),
                '10':list(range(64, 74)),
            }
        }
        self.details = {
            'occlusion_styrofoam':[
                '01_65', '01_66', '01_67', '01_73', '01_74', '01_92', '01_93', '01_94', \
                '02_37', '02_38', '02_39', '02_40', '02_41', '02_42', '02_43', '02_44', '02_85', '02_86', '02_87', '02_88', \
                '03_22', '03_23', '03_24', '03_25', '03_26', '03_27', '03_28', '03_29', \
                '04_25', '04_26', '04_27', '04_28', '04_29', '04_30', '04_31', '04_32', \
                '05_25', '05_26', '05_27', '05_28', '05_29', '05_30', '05_31', '05_32', \
                '06_25', '06_26', '06_27', '06_28', '06_29', '06_30', '06_31', '06_32', \
                '07_25', '07_26', '07_27', '07_28', '07_29', '07_30', '07_31', '07_32', \
                '08_25', '08_26', '08_27', '08_28', '08_29', '08_30', '08_31', '08_32', \
                '09_25', '09_26', '09_27', '09_28', '09_29', '09_30', '09_31', '09_32', \
                '10_28', '10_29', '10_30', '10_31', '10_32', '10_33', '10_34', '10_35'],
            'occlusion_carton':[
                '01_62', '01_63', '01_64', '01_75', '01_76', '01_89', '01_90', '01_91', \
                '02_21', '02_22', '02_23', '02_24', '02_25', '02_26', '02_27', '02_28', '02_77', '02_78', '02_79', '02_80', \
                '03_30', '03_31', '03_32', '03_33', '03_34', '03_35', '03_36', '03_37', \
                '04_33', '04_34', '04_35', '04_36', '04_37', '04_38', '04_39', '04_40', \
                '05_33', '05_34', '05_35', '05_36', '05_37', '05_38', '05_39', '05_40', \
                '06_33', '06_34', '06_35', '06_36', '06_37', '06_38', '06_39', '06_40', \
                '07_33', '07_34', '07_35', '07_36', '07_37', '07_38', '07_39', '07_40', \
                '08_33', '08_34', '08_35', '08_36', '08_37', '08_38', '08_39', '08_40', \
                '09_33', '09_34', '09_35', '09_36', '09_37', '09_38', '09_39', '09_40', \
                '10_36', '10_37', '10_38', '10_39', '10_40', '10_41', '10_42', '10_43'],
            'occlusion_yoga':[
                '02_29', '02_30', '02_31', '02_32', '02_33', '02_34', '02_35', '02_36', '02_81', '02_82', '02_83', '02_84', \
                '03_39', '03_40', '03_41', '03_42', '03_43', '03_44', '03_45', '03_46', \
                '04_41', '04_42', '04_43', '04_44', '04_45', '04_46', '04_47', '04_48', \
                '05_41', '05_42', '05_43', '05_44', '05_45', '05_46', '05_47', '05_48', \
                '06_41', '06_42', '06_43', '06_44', '06_45', '06_46', '06_47', '06_48', \
                '07_41', '07_42', '07_43', '07_44', '07_45', '07_46', '07_47', '07_48', \
                '08_41', '08_42', '08_43', '08_44', '08_45', '08_46', '08_47', '08_48', \
                '09_41', '09_42', '09_43', '09_44', '09_45', '09_46', '09_47', '09_48', \
                '10_44', '10_45', '10_46', '10_47', '10_48', '10_49', '10_50', '10_51'],
            'occlusion_multi':[
                '01_73', '01_74', '01_75', '01_76', \
                '02_77', '02_78', '02_79', '02_80', '02_81', '02_82', '02_83', '02_84', '02_85', '02_86', '02_87', '02_88', \
                '03_26', '03_27', '03_28', '03_29', '03_34', '03_35', '03_36', '03_37', '03_43', '03_44', '03_45', '03_46', \
                '04_29', '04_30', '04_31', '04_32', '04_37', '04_38', '04_39', '04_40', '04_45', '04_46', '04_47', '04_48', \
                '05_29', '05_30', '05_31', '05_32', '05_37', '05_38', '05_39', '05_40', '05_45', '05_46', '05_47', '05_48', \
                '06_29', '06_30', '06_31', '06_32', '06_37', '06_38', '06_39', '06_40', '06_45', '06_46', '06_47', '06_48', \
                '07_29', '07_30', '07_31', '07_32', '07_33', '07_34', '07_35', '07_36', '07_45', '07_46', '07_47', '07_48', \
                '08_29', '08_30', '08_31', '08_32', '08_37', '08_38', '08_39', '08_40', '08_45', '08_46', '08_47', '08_48', \
                '09_29', '09_30', '09_31', '09_32', '09_37', '09_38', '09_39', '09_40', '09_45', '09_46', '09_47', '09_48', \
                '10_32', '10_33', '10_34', '10_35', '10_40', '10_41', '10_42', '10_43', '10_48', '10_49', '10_50', '10_51'],
            'occlusion_action':['01_62', '01_63', '01_64', '01_65', '01_66', '01_67'],
            'dark_multi':[
                '08_65', '08_66', '08_67', '08_68', \
                '09_65', '09_66', '09_67', '09_68', \
                '10_68', '10_69', '10_70', '10_71'],
            'dark_light_change':['08_69', '08_70', '09_69', '09_70', '09_71', '10_72', '10_73'],
        }
        for key, value in statistics.items():
            statistics[key] = ['%02d_%02d' % (int(k), g)  for k, v in value.items() for g in v ]
        self.statistics = statistics

    def info(self):
        """Display HIBER dataset information.

        Returns:
            str: Description and statistics of HIBER dataset.
        """
        heads = [
        'HIBERDataset statistic info:\n',
        'Main categories and their corresponding number of groups:',
        'Each group contains 590 optical frames (annotations) and 1180 RF frames.',
        'If you want to know precise group list of each categories, please access datasetobj.statistics attribute.\n']

        detail_heads = [
        '\nDetailed categories and their corresponding number of groups:',
        'These detailed categories are all included in main categories.',
        'If you want to know precise group list of each detailed categories, please access datasetobj.details attribute.\n']

        dataset = {}
        for k, v in self.statistics.items():
            dataset[k] = len(v)
        details = {}
        for k, v in self.details.items():
            details[k] = len(v)
        dataset_info = json.dumps(dataset, indent=4)
        details_info = json.dumps(details, indent=4)
        info = '\n'.join([*heads, dataset_info, *detail_heads, details_info])
        return info

    def __prepare_data__(self):
        """Construct indexes of data items and categories.

        Returns:
            Tuple(np.ndarray, str): indexes of data items and tokens of categories. 'A', 'M', 'O', 'D', 'W' stand for 'ACTION', 'MULTI', 'OCCLUSION', 'DARK', 'WALK'.
        """
        datas = []
        cates = []
        for k in self.chosen_subsets:
            datas.extend(self.statistics[k])
            cates.extend([k[0]] * len(self.statistics[k]) * self.fpg)
        cates = ''.join(cates)
        datas = list(product(datas, range(590)))
        datas = [[int(x[:2]), int(x[3:]), y]for (x, y) in datas]
        datas = np.array(datas)
        return datas, cates
            
    def __len__(self):
        return sum([len(self.statistics[k]) for k in self.chosen_subsets]) * self.fpg

    def __getitem__(self, index):
        """Load data item and corresponding annotations.

        Args:
            index (int): Index of data item.

        Returns:
            Tuple(np.ndarray, ..., np.ndarray): Data item and corresponding annotations, [hor, ver, pose2d, pose3d, hbox, vbox, silhouette].
        """
        v, g, f = self.datas[index]
        print(v, g, f)
        subset = [x for x in self.subsets if x.startswith(self.cates[index])][0]
        prefix = os.path.join(self.root, subset)
        hor = os.path.join(prefix, 'HORIZONTAL_HEATMAPS', '%02d_%02d' % (v, g), '%04d.npy' % f)
        ver = os.path.join(prefix, 'VERTICAL_HEATMAPS', '%02d_%02d' % (v, g), '%04d.npy' % f)
        anno_prefix = os.path.join(self.root, subset, 'ANNOTATIONS')
        pose2d = os.path.join(anno_prefix, '2DPOSE', '%02d_%02d.npy' % (v, g))
        pose3d = os.path.join(anno_prefix, '3DPOSE', '%02d_%02d.npy' % (v, g))
        hbox = os.path.join(anno_prefix, 'BOUNDINGBOX', 'HORIZONTAL', '%02d_%02d.npy' % (v, g))
        vbox = os.path.join(anno_prefix, 'BOUNDINGBOX', 'VERTICAL', '%02d_%02d.npy' % (v, g))
        silhouette = os.path.join(anno_prefix, 'SILHOUETTE', '%02d_%02d' % (v, g), '%04d.npy' % f)
        hor, ver = np.load(hor), np.load(ver)
        pose2d, pose3d = np.load(pose2d)[f], np.load(pose3d)[f]
        hbox, vbox = np.load(hbox)[f], np.load(vbox)[f]
        silhouette = np.load(silhouette)
        return hor, ver, pose2d, pose3d, hbox, vbox, silhouette



