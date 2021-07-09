"""
    Author: Knoxliu (dengkailiu@whu.edu.cn)
    All rights reserved.
"""
import torch
import collections
import os
import soundfile as sf
# from scipy.io import wavfile
from torch.utils.data import DataLoader, Dataset
import numpy as np
from joblib import Parallel, delayed
import h5py
import random

LOGICAL_DATA_ROOT = '../data_logical/LA'
PHISYCAL_DATA_ROOT = '../data_physical'
SELECTED_METHOD = 'A01'

ASVFile = collections.namedtuple('ASVFile',
    ['speaker_id', 'file_name', 'path', 'sys_id', 'key'])

# sz_dataset = 4000

class ASVDataset(Dataset):
    """ Utility class to load  train/dev datatsets """
    def __init__(self, transform=None, is_train=True, is_logical=True,
                 feature_name=None, is_test=False, dataset_sz=0,
                 multi_class=False, interval=64000):
        if is_logical:
            data_root = LOGICAL_DATA_ROOT
            track = 'LA'
        else:
            data_root = PHISYCAL_DATA_ROOT
            track = 'PA'
        # if is_test:
        #     data_root = os.path.join('eval_data', data_root)
        assert feature_name is not None, 'must provide feature name'
        self.track = track
        self.is_logical = is_logical
        self.dataset_sz = dataset_sz
        self.prefix = 'ASVspoof2019_{}'.format(track)
        v1_suffix = ''
        # if is_test and track == 'PA':
        #     v1_suffix='_v1'
        self.sysid_dict = {
            '-': 0,  # bonafide speech
            'A01': 1, # Wavenet vocoder, A01->SS_1
            'A02': 2, # Conventional vocoder WORLD, A02->SS_2
            'A03': 3, # Conventional vocoder MERLIN, A03->SS_4
            'A04': 4, # Unit selection system MaryTTS, A04->US_1
            'A05': 5, # Voice conversion using neural networks, A05->VC_1
            'A06': 6, # transform function-based voice conversion, A06->VC_4
            'A07': 7,
            'A08': 8,
            'A09': 9,
            'A10': 10,
            'A11': 11,
            'A12': 12,
            'A13': 13,
            'A14': 14,
            'A15': 15,
            'A16': 16,
            'A17': 17,
            'A18': 18,
            'A19': 19
            # For PA:
            # 'AA':7,
            # 'AB':8,
            # 'AC':9,
            # 'BA':10,
            # 'BB':11,
            # 'BC':12,
            # 'CA':13,
            # 'CB':14,
            # 'CC': 15
        }
        self.is_test = is_test
        self.is_selected = 'selected' if SELECTED_METHOD != '' else 'full'
        self.is_train = is_train
        self.sysid_dict_inv = {v: k for k, v in self.sysid_dict.items()}
        self.data_root = data_root
        self.dset_name = 'eval' if is_test else 'train' if is_train else 'dev'
        self.protocols_fname = 'eval.trl' if is_test else 'train.trn' if is_train else 'dev.trl'
        self.protocols_dir = os.path.join(self.data_root,
            '{}_cm_protocols/'.format(self.prefix))
        self.audio_files_dir = os.path.join(self.data_root, '{}_{}'.format(
            self.prefix, self.dset_name )+v1_suffix, 'flac')
        self.protocols_fname = os.path.join(self.protocols_dir,
            'ASVspoof2019.{}.cm.{}.txt'.format(track, self.protocols_fname))
        self.cache_fname = 'cache_{}_{}_{}_{}_{}.npy'.format(self.dset_name, track, feature_name, dataset_sz, self.is_selected)
        self.cache_matlab_fname = 'cache_{}_{}_{}_{}.mat'.format(
            self.dset_name, track, feature_name, dataset_sz)
        self.transform = transform
        self.interval = interval
        if os.path.exists(self.cache_fname):
            self.data_x, self.data_y, self.files_meta = torch.load(self.cache_fname)
            print('Dataset loaded from cache {}\n'.format(self.cache_fname))
        elif feature_name == 'cqcc':
            if os.path.exists(self.cache_matlab_fname):
                self.data_x, self.data_y, self.sys_id = self.read_matlab_cache(self.cache_matlab_fname)
                files_meta = self.parse_protocols_file(self.protocols_fname)
                # need to solve !!!
                self.files_meta = files_meta

                print('Dataset loaded from matlab cache {}\n'.format(self.cache_matlab_fname))
                torch.save((self.data_x, self.data_y, self.files_meta),
                           self.cache_fname, pickle_protocol=4)
                print('Dataset saved to cache {}\n'.format(self.cache_fname))
            else:
                print("Matlab cache for cqcc feature do not exist.")
        else:
            files_meta = self.parse_protocols_file(self.protocols_fname)
            data = list(map(self.read_file, files_meta))
            # self.data_x, self.data_y, self.data_sysid, self.file_name = map(list, zip(*data))
            self.data_x, self.data_y, self.files_meta = self.preprocess(data, interval)
            if self.transform:
                # self.data_x = list(map(self.transform, self.data_x))
                self.data_x = Parallel(n_jobs=4, prefer='threads')(delayed(self.transform)(x) for x in self.data_x)
            torch.save((self.data_x, self.data_y, self.files_meta), self.cache_fname)
            print('Dataset saved to cache {}\n'.format(self.cache_fname))

        self.length = len(self.data_x)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.data_x[idx]
        y = self.data_y[idx]
        return x, y, self.files_meta[idx]

    def read_file(self, meta):
        data_x, sample_rate = sf.read(meta.path)
        # data_x, sample_rate = wavfile.read(meta.path)
        data_y = meta.key
        return data_x, float(data_y), meta

    def _parse_line(self, line):
        tokens = line.strip().split(' ')
        return ASVFile(speaker_id=tokens[0],
            file_name=tokens[1],
            path=os.path.join(self.audio_files_dir, tokens[1] + '.flac'),
            sys_id=self.sysid_dict[tokens[3]],
            key=int(tokens[4] == 'bonafide'))

    def _select_lines(self, lines):
        new_lines = []
        for line in lines:
            tokens = line.strip().split(' ')
            if tokens[3] == SELECTED_METHOD:
                new_lines.append(line)
        return new_lines

    def parse_protocols_file(self, protocols_fname):
        lines = open(protocols_fname).readlines()
        # Selected one spoofing method for training:
        if SELECTED_METHOD != '' and self.is_train:
            lines = self._select_lines(lines)

        # training or evaluating.
        if self.is_test and self.dataset_sz:

        # use subset. debug only !!!
        # if self.dataset_sz:
            random.seed(1)
            random.shuffle(lines)
            files_meta = map(self._parse_line, lines[:self.dataset_sz])
        else:
            files_meta = map(self._parse_line, lines)
        return list(files_meta)

    def read_matlab_cache(self, filepath):
        f = h5py.File(filepath, 'r')
        # filename_index = f["filename"]
        # filename = []
        data_x_index = f["data_x"]
        sys_id_index = f["sys_id"]
        data_x = []
        data_y = f["data_y"][0]
        sys_id = []
        for i in range(0, data_x_index.shape[1]):
            idx = data_x_index[0][i]  # data_x
            temp = f[idx]
            data_x.append(np.array(temp).transpose())
            # idx = filename_index[0][i]  # filename
            # temp = list(f[idx])
            # temp_name = [chr(x[0]) for x in temp]
            # filename.append(''.join(temp_name))
            idx = sys_id_index[0][i]  # sys_id
            temp = f[idx]
            sys_id.append(int(list(temp)[0][0]))
        data_x = np.array(data_x)
        data_y = np.array(data_y)
        return data_x.astype(np.float32), data_y.astype(np.int64)


    def preprocess(self, data, interval):
        # data_x, data_y, data_sysid, file_name = map(list, zip(*data))
        recon_data = []
        for elem in data:
            x = elem[0]
            s_length = x.shape[0]
            if s_length < interval:
                num_repeats = (interval // s_length) + 1
                x_repeat = np.repeat(x, num_repeats)
                padding_x = x_repeat[:interval]
                tmp_list = [padding_x, elem[1], elem[2]]
                recon_data.append(tmp_list)
            else:
                # frame division with no overlap
                if s_length % interval:
                    num_repeats = (s_length // interval) + 1
                    padding_x = np.pad(x, (0, num_repeats*interval-s_length//interval*interval),
                                       'constant', constant_values=(0, 0))
                else:
                    num_repeats = s_length // interval
                    padding_x = x
                # frame division with 50% overlap
                # hopping = int(0.5 * interval)
                # num_repeats = (s_length - interval) // hopping + 1
                for i in range(num_repeats):
                    fn = '{}-{}'.format(elem[2].file_name, str(i))
                    meta_tmp = ASVFile(speaker_id=elem[2].speaker_id,
                                        file_name=fn,
                                        path=elem[2].path,
                                        sys_id=elem[2].sys_id,
                                        key=elem[2].key)
                    # frame division with 50% overlap
                    # tmp_list = [x[i*hopping:(i*hopping+interval)], elem[1], meta_tmp]

                    # frame division with no overlap
                    tmp_list = [padding_x[i*interval:(i+1)*interval], elem[1], meta_tmp]
                    recon_data.append(tmp_list)

        return map(list, zip(*recon_data))


# if __name__ == '__main__':
#    train_loader = ASVDataset(feature_name='spect',
#                              is_train=True, dataset_sz=10)
#    data_loader1 = DataLoader(train_loader, batch_size=2, shuffle=False)
#    for i, (batch_x, batch_y, batch_meta) in enumerate(data_loader1):
#        print(batch_meta)
#        print('\n')
#        print(type(batch_x), type(batch_y))
#        print('\n')
#    print(len(train_loader))

