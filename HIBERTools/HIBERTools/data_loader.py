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
import lmdb
from pathlib import Path
from itertools import product
from tqdm import tqdm


class HIBERDataset():
    """HIBER dataset load and visualize object"""

    def __init__(self, root=None, categories=['ACTION', 'DARK', 'MULTI', 'OCCLUSION', 'WALK'], 
                    mode='train', data_file=None, channel_first=True, complex=False) -> None:
        """HIBERDataset constructor.

        Args:
            root (str, optional): Root path of HIBER dataset directory. Defaults to None.
            categories (list[str], optional): Categories to be processed. Defaults to ['ACTION', 'DARK', 'MULTI', 'OCCLUSION', 'WALK'].
            mode (str, optional): Train/Validation/Test set, candidate values are ['train', 'val', 'test']. Defaults to 'train'
            data_file (str, optional): The directory of custom dataset split file (JSON format).
            channel_first (bool, optional): If true the radar frame shape will be (2, 160, 200), else (160, 200, 2). 
                                            This is determined by downloaded dataset file version. Defaults to 'True'.
            complex (bool, optional): If true, complex radar frame will be loaded and saved, 
                                            else real radar frame will be loaded and saved. Defaults to 'False'
        """
        super().__init__()
        self.categories = ['ACTION', 'DARK', 'MULTI', 'OCCLUSION', 'WALK']
        assert set(categories) <= set(self.categories), \
            "categories should in ['ACTION', 'DARK', 'MULTI', 'OCCLUSION', 'WALK']" 
        self.root = root
        self.chosen_cates = categories
        self.fpg = 590 # number of frames per group
        assert mode in ['train', 'val', 'test'], 'invalid dataset mode'
        self.mode = mode
        self.__statistics__(self.mode, data_file)
        self.datas, self.cates = self.__prepare_data__() # index of data items and categories
        self.total_keys = None # lmdb data keys
        self.channel_first = channel_first
        self.complex = complex

    def __statistics__(self, mode, data_file=None):
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
        if data_file is not None:
            with open(data_file, 'r') as f:
                db = json.load(f)
            val_set = db['Val']
            test_set = db['Test']
            train_set = db['Train']
        else:
            val_set = {
                'WALK': [
                            '07_01', '05_02', '06_02', '10_13', '07_05', '09_06', 
                            '03_12', '05_09', '06_09', '09_04', '09_03'], 
                'MULTI': [
                            '04_15', '08_19', '10_26', '05_23', '09_20', '04_23', 
                            '06_24', '08_21', '06_15', '04_19'], 
                'ACTION': [
                            '10_52', '10_56', '07_59', '08_51', '02_53', '05_57', 
                            '07_60', '08_52', '04_60', '05_50', '08_59'], 
                'OCCLUSION': [
                            '03_33', '05_30', '04_36', '10_44', '03_42', '09_47', 
                            '02_22', '08_48', '06_46', '10_41', '07_31', '06_28', 
                            '05_40', '09_36', '05_25', '02_41', '09_46', '06_25', 
                            '02_43', '10_46', '04_40', '08_28'], 
                'DARK': []
            }
            test_set = {
                'WALK': [
                            '01_01', '01_02', '01_03', '01_04', '01_05', '01_06', 
                            '01_07', '01_08', '01_09', '01_10', '01_11', '01_12', 
                            '01_13', '01_14', '01_15', '01_16', '01_17', '01_18', 
                            '01_19', '01_20', '01_21', '01_22', '01_23', '01_24', 
                            '01_25', '01_26', '01_27', '01_28', '01_29', '01_30', 
                            '01_31', '01_32', '01_33', '01_34', '01_35', '09_05', 
                            '02_03', '06_06', '10_09', '05_11', '09_11'], 
                'MULTI': [
                            '01_77', '01_78', '01_79', '01_80', '01_81', '01_82', 
                            '01_83', '01_84', '01_85', '01_86', '01_87', '01_88', 
                            '08_13', '06_23', '02_69', '08_16', '05_21', '07_16'], 
                'ACTION': [
                            '01_37', '01_38', '01_39', '01_40', '01_41', '01_42', 
                            '01_43', '01_44', '01_45', '01_46', '01_47', '01_48', 
                            '01_49', '01_50', '01_51', '01_52', '01_53', '01_54', 
                            '01_55', '01_56', '01_57', '01_58', '01_59', '01_60', 
                            '01_61', '06_50', '06_51', '08_55', '05_54', '07_55', 
                            '07_54'], 
                'OCCLUSION': [
                            '01_62', '01_63', '01_64', '01_65', '01_66', '01_67', 
                            '01_73', '01_74', '01_75', '01_76', '01_89', '01_90', 
                            '01_91', '01_92', '01_93', '01_94', '04_38', '06_42', 
                            '04_48', '07_47', '06_43', '02_38', '07_44', '04_35', 
                            '05_44', '04_29', '08_40', '07_40'], 
                'DARK': [
                            '08_61', '08_62', '08_63', '08_64', '08_65', '08_66', 
                            '08_67', '08_68', '08_69', '08_70', '09_61', '09_62', 
                            '09_63', '09_64', '09_65', '09_66', '09_67', '09_68', 
                            '09_69', '09_70', '09_71', '10_64', '10_65', '10_66', 
                            '10_67', '10_68', '10_69', '10_70', '10_71', '10_72', 
                            '10_73']}
            train_set = {}
            for k in val_set.keys():
                train_set[k] = train_set.get(k, [])
                train_set[k] = list(set(statistics[k]) - set(val_set[k]) - set(test_set[k]))
        
        self.__train_set__ = train_set
        self.__test_set__ = test_set
        self.__val_set__ = val_set
        __datas__ = {}
        for k, v  in eval(mode + '_set').items():
            if k in self.chosen_cates:
                __datas__[k] = v
        self.__datas__ = __datas__

    def info(self):
        """Display current HIBER dataset information.

        Returns:
            str: Description and statistics of HIBER dataset.
        """
        total_grps = sum([len(v) for k, v in self.__datas__.items()])
        heads = [
            '%s subset of HIBER Dataset.' % self.mode.upper(),
            '%d groups, %d samples in total.\n' % (total_grps, total_grps * self.fpg),
            'Detailed information is as follows.',
            '\nYou can save to lmdb format by calling datastobj.save_as_lmdb function.'
        ]
        dataset = {}
        for k, v in self.__datas__.items():
            if k in self.chosen_cates:
                dataset[k] = len(v)

        dataset_info = json.dumps(dataset, indent=4)
        info = '\n'.join([*heads[:-1], dataset_info, heads[-1]])
        return info

    def complete_info(self):
        """Display complete HIBER dataset information.

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

        train_head = '\nTrain/Validation/Test set are splited as follows.\nTrain set'
        val_head = '\nValidation set'
        test_head = '\nTest set'

        dataset = {}
        for k, v in self.statistics.items():
            dataset[k] = len(v)
        details = {}
        for k, v in self.details.items():
            details[k] = len(v)

        train_set = {}
        for k, v in self.__train_set__.items():
            train_set[k] = len(v)
        val_set = {}
        for k, v in self.__val_set__.items():
            val_set[k] = len(v)
        test_set = {}
        for k, v in self.__test_set__.items():
            test_set[k] = len(v)

        dataset_info = json.dumps(dataset, indent=4)
        details_info = json.dumps(details, indent=4)
        train_set_info = json.dumps(train_set, indent=4)
        val_set_info = json.dumps(val_set, indent=4)
        test_set_info = json.dumps(test_set, indent=4)

        split_info = [train_head, train_set_info, val_head, val_set_info, test_head, test_set_info]
        info = '\n'.join([*heads, dataset_info, *detail_heads, details_info, *split_info])
        return info

    def __prepare_data__(self):
        """Construct indexes of data items and categories.

        Returns:
            Tuple(np.ndarray, str): indexes of data items and tokens of categories. 'A', 'M', 'O', 'D', 'W' stand for 'ACTION', 'MULTI', 'OCCLUSION', 'DARK', 'WALK'.
        """
        datas = []
        cates = []
        for k in self.chosen_cates:
            datas.extend(self.__datas__[k])
            cates.extend([k[0]] * len(self.__datas__[k]) * self.fpg)
        cates = ''.join(cates)
        datas = list(product(datas, range(self.fpg)))
        datas = [[int(x[:2]), int(x[3:]), y]for (x, y) in datas]
        datas = np.array(datas)
        return datas, cates
            
    def __len__(self):
        return sum([len(self.__datas__[k]) for k in self.chosen_cates]) * self.fpg

    def __getitem__(self, index):
        """Load data item and corresponding annotations.

        Args:
            index (int): Index of data item.

        Returns:
            Tuple(np.ndarray, ..., np.ndarray): Data item and corresponding annotations, [hor, ver, pose2d, pose3d, hbox, vbox, silhouette].
        """
        assert not self.root is None, 'You do not pass "root" parameter to HIBERDataset object.'
        
        v, g, f = self.datas[index]
        categories = {
            'A': 'ACTION',
            'W': 'WALK',
            'M': 'MULTI',
            'O': 'OCCLUSION',
            'D': 'DARK'
        }
        category = categories[self.cates[index]]
        prefix = Path(self.root) / category

        if self.complex:
            hor = prefix / 'HORIZONTAL_HEATMAPS_COMPLEX' / f'{v:02d}_{g:02d}' / f'{f:04d}.npy'
            ver = prefix / 'VERTICAL_HEATMAPS_COMPLEX' / f'{v:02d}_{g:02d}' / f'{f:04d}.npy'

            assert hor.exists() and ver.exists(), 'Complex files does not exists.'

            # hor = hor.view(dtype=np.complex128).squeeze()
            # ver = ver.view(dtype=np.complex128).squeeze()
        else:
            hor = prefix / 'HORIZONTAL_HEATMAPS' / f'{v:02d}_{g:02d}' / f'{f:04d}.npy'
            ver = prefix / 'VERTICAL_HEATMAPS' / f'{v:02d}_{g:02d}' / f'{f:04d}.npy'

        hor, ver = np.load(hor), np.load(ver)

        if self.channel_first and hor.shape[0] != 2:
            print(f'Expected RF frame is (2, 160, 200), but got {hor.shape}, please maybe you should channel_first=False')
        elif not self.channel_first and hor.shape[-1] != 2:
            print(f'Expected RF frame is (160, 200, 2), but got {hor.shape}, please maybe you should channel_first=True')
        
        if hor.shape[-2] == 2:
            hor, ver = hor.transpose((2, 0, 1)), ver.transpose((2, 0, 1))
            hor, ver = np.ascontiguousarray(hor), np.ascontiguousarray(ver)

        anno_prefix = prefix / 'ANNOTATIONS'

        if category == 'DARK':
            # category DARK has no pose and boundingbox annotations
            pose2d = np.zeros((0, 14, 2))
            pose3d = np.zeros((0, 14, 3))
            hbox = np.zeros((0, 4))
            vbox = np.zeros((0, 4))
        else:
            pose2d = anno_prefix / '2DPOSE' / f'{v:02d}_{g:02d}.npy'
            pose3d = anno_prefix / '3DPOSE' / f'{v:02d}_{g:02d}.npy'
            hbox = anno_prefix / 'BOUNDINGBOX/HORIZONTAL' / f'{v:02d}_{g:02d}.npy'
            vbox = anno_prefix / 'BOUNDINGBOX/VERTICAL' / f'{v:02d}_{g:02d}.npy'
            pose2d, pose3d = np.load(pose2d)[f], np.load(pose3d)[f]
            hbox, vbox = np.load(hbox)[f], np.load(vbox)[f]
        silhouette = anno_prefix / 'SILHOUETTE' / f'{v:02d}_{g:02d}' / f'{f:04d}.npy'
        silhouette = np.load(silhouette)
        return hor, ver, pose2d, pose3d, hbox, vbox, silhouette

    def save_as_lmdb(self, out_path, complex=None):
        """Save dataset as lmdb format.

        Args:
            out_path (str): Ouput file path.
        """
        assert not self.root is None, 'You do not pass "root" parameter to HIBERDataset object.'
        assert self.__len__() != 0, 'At least one subset should be assigned.'
        if not complex is None:
            self.complex = complex

        data_item = self.__getitem__(0)
        total_samples = self.__len__()

        instance_byte = sum([x.nbytes for x in data_item]) * total_samples
        env = lmdb.open(out_path, map_size=instance_byte * 2)
        txn = env.begin(write=True)

        total_keys = self.get_lmdb_keys()

        for i in tqdm(range(total_samples)):
            data_item = self.__getitem__(i)
            keys = total_keys[i]
            for d, k in zip(data_item, keys):
                txn.put(k.encode('ascii'), d)
            txn.commit()
            txn = env.begin(write=True)
        env.close()

    def get_lmdb_keys(self, regenerate=False):
        """Return lmdb keys for loading from lmdb format.

        Args:
            regenerate (bool, optional): Whether to recalculate keys. Defaults to False.

        Returns:
            list: list of sample keys, each element contains 7 keys.
        """
        if self.total_keys is None or regenerate:
            prefixes = ['h_', 'v_', 'p2d_', 'p3d_', 'hb_', 'vb_', 'm_']
            total_keys = []
            for i in range(self.__len__()):
                v, g, f = self.datas[i]
                keys = [x + '%02d_%02d_%04d' % (v, g, f) for x in prefixes]
                total_keys.append(keys)
            self.total_keys = total_keys
        else:
            total_keys = self.total_keys
        return total_keys


