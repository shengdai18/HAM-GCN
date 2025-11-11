from cross_validation import *
from preparedata.prepare_data_SEED import *
import argparse
from scipy.io import loadmat
import numpy as np


class HyperParameters:
    def __init__(self):
        ######## Data ########
        self.dataset = 'SEED'
        self.data_path = 'D:\数据集\情绪数据集\SEED\Preprocessed_EEG'
        self.data_save_path = 'D:\数据集\LGGNet数据'
        self.subjects = 15
        self.num_class = 3  # choices=[2, 3, 4]
        self.label_type = 'L'  # choices=['A', 'V', 'D', 'L']
        self.segment = 4
        self.overlap = 0
        self.sampling_rate = 200
        self.scale_coefficient = 1
        self.input_shape = (1, 62, 200)
        self.data_format = 'org'

        ######## Model Parameters ########
        self.model = 'ProgressiveMask'
        self.pool = 16
        self.pool_step_rate = 0.25
        self.T = 64
        self.hidden = 32

        ######## Reproduce the result using the saved model ######
        self.reproduce = False


if __name__ == '__main__':
    args = HyperParameters()
    sub_to_run = np.arange(1, args.subjects + 1)
    pd = PrepareData(args)
    pd.run(sub_to_run, split=True, expand=True)



