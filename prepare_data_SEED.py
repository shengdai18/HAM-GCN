# This is the processing script of DEAP dataset

import _pickle as cPickle

import numpy as np
import pandas as pd
from train_model import *
import os
from scipy.io import loadmat


class PrepareData:
    def __init__(self, data_format, data_path, data_save_path):
        # init all the parameters here
        # arg contains parameter settings
        self.data = None
        self.label = None
        self.model = None
        self.data_type = data_format
        self.data_path = data_path
        self.data_save_path = data_save_path
        self.save_path = self.get_save_path()
        # 原始电极顺序
        # channel_order = pd.read_excel('D:\\数据\\情绪数据集\\SEED\\SEED\\channel-order.xlsx', header=None)
        # self.original_order = channel_order.iloc[:, 0]
        # self.original_order = ['Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4',
        #                        'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
        #                        'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
        #                        'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
        #                        'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
        #                        'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
        #                        'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'Oz', 'O2', 'CB2']
        #
        # self.graph_fro_DEAP = [['Fp1', 'AF3'], ['Fp2', 'AF4'], ['F3', 'F5', 'F7', 'F1'], ['F2', 'F4', 'F6', 'F8'],
        #                        ['Fpz', 'Fz', 'FCz'],
        #                        ['FC5', 'FC3', 'FC1'], ['FC6', 'FC2', 'FC4'], ['C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6'],
        #                        ['CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6'],
        #                        ['P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8'],
        #                        ['PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8'],
        #                        ['CB1', 'O1', 'Oz', 'O2', 'CB2'],
        #                        ['FT7', 'T7', 'TP7'], ['FT8', 'T8', 'TP8']]
        # self.graph_gen_DEAP = [['Fp1', 'Fpz' 'Fp2'], ['AF3', 'AF4'],
        #                        ['F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8'],
        #                        ['FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6'],
        #                        ['C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6'],
        #                        ['CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6'],
        #                        ['P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8'],
        #                        ['PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8'], ['CB1', 'O1', 'Oz', 'O2', 'CB2'],
        #                        ['FT7', 'T7', 'TP7'], ['FT8', 'T8', 'TP8']]
        #
        # self.graph_type = graph_type

    def run(self, subject_list, split=False, expand=True):
        """
        Parameters
        ----------
        subject_list: the subjects need to be processed
        split: (bool) whether to split one trial's data into shorter segment
        expand: (bool) whether to add an empty dimension for CNN

        Returns
        -------
        The processed data will be saved './data_<data_format>_<dataset>_<label_type>/sub0.hdf'
        """
        for sub in subject_list:
            if self.data_type == 'DE':
                data_, label_ = self.get_DE_per_sub(sub)
            else:
                data_, label_ = self.load_data_per_subject(sub)
            # select label type here

            print('Data and label prepared!')
            print('data:' + str(data_.shape) + ' label:' + str(label_.shape))
            print('----------------------')


            self.save(data_, label_, sub)

    def load_data_per_subject(self, sub):
        """
        This function loads the target subject's original file
        Parameters
        ----------
        sub: which subject to load

        Returns
        -------
        data: (40, 32, 7680) label: (40, 4)
        """
        file_names = os.listdir(self.data_path)
        # all_label = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]
        all_label = [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 0, 1, 2, 0]
        all_data = []
        labels = []
        for file_name in file_names:
            sub_id = file_name.split('_')[0]
            if str(sub) == sub_id:
                file_path = os.path.join(self.data_path,file_name)
                sub_dict = loadmat(file_path)
                print(sub_dict.keys())
                for key in sub_dict.keys():
                    if 'eeg' in key:
                        data = sub_dict[key]
                        data = np.array(data)
                        trial_num = eval(key.split('_')[1][3:])
                        label = [all_label[trial_num-1]]
                        label = np.array(label)
                        data = np.expand_dims(data, axis=0)
                        data, label = self.split(data, label, segment_length=1)
                        data = self.reorder_channel(data=data, graph=self.graph_type)

                        all_data.append(data)
                        labels.append(label)
                all_data_array = np.concatenate(all_data, axis=0)
                all_label_array = np.concatenate(labels, axis=0)
        return all_data_array, all_label_array

    def get_DE_per_sub(self, sub):
        file_names = os.listdir(self.data_path)
        # all_label = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]
        all_label = [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 0, 1, 2]
        trial_len = []
        all_data = []
        labels = []
        for file_name in file_names:
            sub_id = file_name.split('_')[0]
            if str(sub) == sub_id:
                file_path = os.path.join(self.data_path,file_name)
                sub_dict = loadmat(file_path)
                print(sub_dict.keys())
                for key in sub_dict.keys():
                    if 'eeg' in key:
                        data = sub_dict[key]
                        data = np.array(data)
                        trial_num = eval(key.split('_')[1][3:])
                        label = [all_label[trial_num-1]]
                        label = np.array(label)
                        data = np.expand_dims(data, axis=0)
                        DE = decompose(data)
                        trial_len.append(len(DE))
                        data, label = self.split(data, label, segment_length=1)
                        # data = self.reorder_channel(data=data, graph=self.graph_type)
                        all_data.append(DE)
                        labels.append(label)
                trial_len = np.array(trial_len)
                # np.save('trial_len.npy', trial_len)
                all_data_array = np.concatenate(all_data, axis=0)
                all_label_array = np.concatenate(labels, axis=0)
        return all_data_array, all_label_array

    def reorder_channel(self, data, graph):
        """
        This function reorder the channel according to different graph designs
        Parameters
        ----------
        data: (trial, channel, data)
        graph: graph type

        Returns
        -------
        reordered data: (trial, channel, data)
        """
        if graph == 'fro':
            graph_idx = self.graph_fro_DEAP
        elif graph == 'gen':
            graph_idx = self.graph_gen_DEAP
        elif graph == 'hem':
            graph_idx = self.graph_hem_DEAP
        elif graph == 'BL':
            graph_idx = self.original_order
        elif graph == 'TS':
            graph_idx = self.TS

        idx = []
        if graph in ['BL', 'TS']:
            for chan in graph_idx:
                idx.append(self.original_order.index(chan))
        else:
            num_chan_local_graph = []
            for i in range(len(graph_idx)):
                num_chan_local_graph.append(len(graph_idx[i]))
                for chan in graph_idx[i]:
                    idx.append(self.original_order.index(chan))

            total_chan_num = sum(num_chan_local_graph)

            # save the number of channels in local graph for building the LGG model in utils.py
            dataset = h5py.File('num_chan_local_graph_{}.hdf'.format(graph), 'w')
            dataset['data'] = num_chan_local_graph
            dataset.close()
        return data[:, :, idx, :]

    def get_save_path(self):
        save_path = self.data_save_path
        data_type = 'data_{}_SEED'.format(self.data_type)
        save_path = osp.join(save_path, data_type)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            pass
        return save_path

    def save(self, data, label, sub):
        """
        This function save the processed data into target folder
        Parameters
        ----------
        data: the processed data
        label: the corresponding label
        sub: the subject ID
        Returns
        -------
        None
        """
        name = 'sub' + str(sub) + '.hdf'
        save_path = osp.join(self.save_path, name)
        dataset = h5py.File(save_path, 'w')
        dataset['data'] = data
        dataset['label'] = label
        dataset.close()

    def split(self, data, label, segment_length=1, overlap=0, sampling_rate=200):
        """
        This function split one trial's data into shorter segments
        Parameters
        ----------
        data: (f, channel, data)
        label: (1，)
        segment_length: how long each segment is (e.g. 1s, 2s,...)
        overlap: overlap rate
        sampling_rate: sampling rate

        Returns
        -------
        data:(num_segment, f, channel, segment_legnth)
        label:(num_segment,)
        """
        data_shape = data.shape
        step = int(segment_length * sampling_rate * (1 - overlap))
        data_segment = sampling_rate * segment_length
        data_split = []

        number_segment = int((data_shape[-1] - data_segment) // step)
        for i in range(number_segment + 1):
            data_split.append(data[:, :, (i * step):(i * step + data_segment)])
        data_split_array = np.stack(data_split, axis=0)
        label = np.repeat(label, number_segment+1)
        print("The data and label are split: Data shape:" + str(data_split_array.shape) + " Label:" + str(
            label.shape))
        data = data_split_array
        return data, label

if __name__ == '__main__':
    data_format = 'DE'
    data_path = 'D:\数据集\情绪数据集\SEED\Preprocessed_EEG'
    data_save_path = 'D:\数据集\LGGNet数据'

    pre_data = PrepareData(data_format, data_path, data_save_path)
    sub_list = np.arange(1, 15)
    pre_data.run(sub_list)