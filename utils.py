from torch.utils.data import Dataset
import torch
import numpy as np
import os.path as osp
import pickle
import random
from torch.utils.data import DataLoader
import csv
import os
import re
import h5py

import os
import time
import h5py
import numpy as np
import pprint
import random
from scipy.signal import butter, lfilter

from models.LGGNet import *
from eeg_dataset import *
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score


class EEGDataset(Dataset):
    # x_tensor: (sample, channel, datapoint(feature)) type = torch.tensor
    # y_tensor: (sample,) type = torch.tensor
    def __init__(self, x, y):
        self.x = x
        self.y = y

        assert self.x.size(0) == self.y.size(0)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.y)


def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)


def load_data_per_subject(load_path, sub):
    sub_code = 'sub' + str(sub) + '.hdf'
    path = osp.join(load_path, sub_code)
    # with open(path, 'rb') as file:
    # dataset = pickle.load(file)
    dataset = h5py.File(path, 'r')
    # data = dataset['data']
    # label = dataset['label']
    data = np.array(dataset['data'])
    label = np.array(dataset['label'])
    return data, label


def get_channel_info(load_path, graph_type):
    # path_info = osp.join(load_path, 'dataset_info.pkl')
    # with open(path_info, 'rb') as file:
    #     dataset_info = pickle.load(file)
    original_order = ['Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4',
                      'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
                      'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
                      'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
                      'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
                      'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
                      'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'Oz', 'O2', 'CB2']

    graph_fro_SEED = [['Fp1', 'AF3'], ['Fp2', 'AF4'], ['F3', 'F5', 'F7', 'F1'], ['F2', 'F4', 'F6', 'F8'],
                      ['Fpz', 'Fz', 'FCz'],
                      ['FC5', 'FC3', 'FC1'], ['FC6', 'FC2', 'FC4'], ['C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6'],
                      ['CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6'],
                      ['P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8'],
                      ['PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8'],
                      ['CB1', 'O1', 'Oz', 'O2', 'CB2'],
                      ['FT7', 'T7', 'TP7'], ['FT8', 'T8', 'TP8']]
    if graph_type == 'BL':
        graph_idx = original_order
    else:
        graph_idx = graph_fro_SEED
    # original_order = dataset_info['BL']
    input_subgraph = False
    for item in graph_idx:
        if isinstance(item, list):
            input_subgraph = True
    idx_new = []
    num_chan_local_graph = []
    if not input_subgraph:
        for chan in graph_idx:
            idx_new.append(original_order.index(chan))
    else:
        for i in range(len(graph_idx)):
            num_chan_local_graph.append(len(graph_idx[i]))
            for chan in graph_idx[i]:
                idx_new.append(original_order.index(chan))
    return idx_new, num_chan_local_graph


def load_data(load_path, load_idx, keep_subject=False, concat=True):
    data, label = [], []
    for i, idx in enumerate(load_idx):
        data_per_sub, label_per_sub = load_data_per_subject(load_path=load_path, sub=idx)
        if keep_subject:
            data.append(data_per_sub)
            label.append(label_per_sub)
            assert concat is not True, "Please set concat False is keep_subject is True"
        else:
            data.extend(data_per_sub)
            label.extend(label_per_sub)
    if concat:
        data = np.concatenate(data)  # --> seg, chan, data
        # label = np.concatenate(label)
        label = np.array(label)

    return data, label


def get_validation_set(train_idx, val_rate, shuffle):
    if shuffle:
        random.shuffle(train_idx)
    train = train_idx[:int(len(train_idx) * (1 - val_rate))]
    val = train_idx[int(len(train_idx) * (1 - val_rate)):]
    return train, val


def normalize(train, val, test):
    # input should be seg, chan, time/f
    # data: sample x channel x data
    for channel in range(train.shape[1]):
        mean = np.mean(train[:, channel, :])
        std = np.std(train[:, channel, :])
        train[:, channel, :] = (train[:, channel, :] - mean) / std
        val[:, channel, :] = (val[:, channel, :] - mean) / std
        test[:, channel, :] = (test[:, channel, :] - mean) / std
    return train, val, test


def numpy_to_torch(data, label):
    data = torch.from_numpy(data).float()
    label = torch.from_numpy(label).long()
    return data, label


def get_dataloader(data, label, batch_size, shuffle=True):
    # load the data
    dataset = EEGDataset(data, label)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)
    return loader


def prepare_data_for_training(data, label, idx, batch_size, shuffle):
    # reorder the data segment, chan, datapoint
    data = data[:, idx, :]
    # change to torch tensor
    data, label = numpy_to_torch(data=data, label=label)
    # prepare dataloader
    data_loader = get_dataloader(data=data, label=label, batch_size=batch_size, shuffle=shuffle)
    return data_loader


def get_task_chunk(subjects, step):
    return np.array_split(subjects, len(subjects) // step)


def log2txt(text_file, content):
    file = open(text_file, 'a')
    file.write(str(content) + '\n')
    file.close()


def log2csv(csv_file, content):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write the header row (column names)
        writer.writerow(["Metric", "Value"])

        # Write the data from the dictionary
        for key, value in content.items():
            writer.writerow([key, value])

    print(f"Data has been logged to {csv_file}")


def get_checkpoints(path):
    all_files = os.listdir(path)
    ckpt_files = [file for file in all_files if file.endswith(".ckpt")]
    return ckpt_files


def get_epoch_from_ckpt(ckpt_file):
    epoch_match = re.search(r'epoch=(\d+)', ckpt_file)
    epoch_number = int(epoch_match.group(1))
    return epoch_number


class LabelSmoothing(torch.nn.Module):
    """NLL loss with label smoothing.
       refer to: https://github.com/NVIDIA/DeepLearningExamples/blob/8d8b21a933fff3defb692e0527fca15532da5dc6/PyTorch/Classification/ConvNets/image_classification/smoothing.py#L18
    """

    def __init__(self, smoothing=0.0):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def set_gpu(x):
    torch.set_num_threads(1)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)


def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)


def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()


class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


_utils_pp = pprint.PrettyPrinter()


def pprint(x):
    _utils_pp.pprint(x)


def get_model(args):
    if args.model == 'LGGNet':
        idx_local_graph = list(np.array(h5py.File('num_chan_local_graph_{}.hdf'.format(args.graph_type), 'r')['data']))
        channels = sum(idx_local_graph)
        input_size = args.input_shape
        model = LGGNet(
            num_classes=args.num_class, input_size=input_size,
            sampling_rate=int(args.sampling_rate * args.scale_coefficient),
            num_T=args.T, out_graph=args.hidden,
            dropout_rate=args.dropout,
            pool=args.pool, pool_step_rate=args.pool_step_rate,
            idx_graph=idx_local_graph)
    return model


def get_dataloader(data, label, batch_size):
    # load the data
    dataset = eegDataset(data, label)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    return loader


def get_metrics(y_pred, y_true, classes=None):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='micro')
    if classes is not None:
        cm = confusion_matrix(y_true, y_pred, labels=classes)
    else:
        cm = confusion_matrix(y_true, y_pred)
    return acc, f1, cm


def get_trainable_parameter_num(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params


def L1Loss(model, Lambda):
    w = torch.cat([x.view(-1) for x in model.parameters()])
    err = Lambda * torch.sum(torch.abs(w))
    return err


def L2Loss(model, Lambda):
    w = torch.cat([x.view(-1) for x in model.parameters()])
    err = Lambda * torch.sum(w.pow(2))
    return err


class LabelSmoothing(nn.Module):
    """NLL loss with label smoothing.
       refer to: https://github.com/NVIDIA/DeepLearningExamples/blob/8d8b21a933fff3defb692e0527fca15532d a5dc6/PyTorch/Classification/ConvNets/image_classification/smoothing.py#L18
    """

    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def decompose(data, segment_length=1, sampling_rate=200):
    # trial*channel*sample
    frequency = 200
    decomposed_de = []
    num_sample = data.shape[2] // (segment_length * sampling_rate)

    for channel in range(62):
        trial_signal = data[0][channel]
        # 因为SEED数据没有基线信号部分
        temp_de = np.empty([0, num_sample])

        delta = butter_bandpass_filter(trial_signal, 1, 4, frequency, order=3)
        theta = butter_bandpass_filter(trial_signal, 4, 8, frequency, order=3)
        alpha = butter_bandpass_filter(trial_signal, 8, 14, frequency, order=3)
        beta = butter_bandpass_filter(trial_signal, 14, 31, frequency, order=3)
        gamma = butter_bandpass_filter(trial_signal, 31, 51, frequency, order=3)

        DE_delta = np.zeros(shape=[0], dtype=float)
        DE_theta = np.zeros(shape=[0], dtype=float)
        DE_alpha = np.zeros(shape=[0], dtype=float)
        DE_beta = np.zeros(shape=[0], dtype=float)
        DE_gamma = np.zeros(shape=[0], dtype=float)

        for index in range(num_sample):
            DE_delta = np.append(DE_delta, compute_DE(delta[index * 100:(index + 1) * 100]))
            DE_theta = np.append(DE_theta, compute_DE(theta[index * 100:(index + 1) * 100]))
            DE_alpha = np.append(DE_alpha, compute_DE(alpha[index * 100:(index + 1) * 100]))
            DE_beta = np.append(DE_beta, compute_DE(beta[index * 100:(index + 1) * 100]))
            DE_gamma = np.append(DE_gamma, compute_DE(gamma[index * 100:(index + 1) * 100]))
        temp_de = np.vstack([temp_de, DE_delta])
        temp_de = np.vstack([temp_de, DE_theta])
        temp_de = np.vstack([temp_de, DE_alpha])
        temp_de = np.vstack([temp_de, DE_beta])
        temp_de = np.vstack([temp_de, DE_gamma])
        decomposed_de.append(temp_de)
    decomposed_de = np.stack(decomposed_de, axis=0)
    decomposed_de = decomposed_de.transpose([2, 0, 1])

    print("trial_DE shape:", decomposed_de)
    return decomposed_de


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def compute_DE(signal):
    variance = np.var(signal, ddof=1)
    return math.log(2 * math.pi * math.e * variance) / 2
