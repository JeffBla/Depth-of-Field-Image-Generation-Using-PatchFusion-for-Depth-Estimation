"""Dataset class template

You can specify '--dataset_mode rockXCT' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
You need to implement the following functions:
    -- <modify_commandline_options>: Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from pydicom import dcmread
import numpy as np
import pandas as pd
import torch

def Rescale0_1(imagePlusOneDim: torch.Tensor, CTG:float, WATER:float, AIR:float) -> torch.Tensor:
    return (imagePlusOneDim - AIR)/(3000-AIR)

class RockXCTDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument(
            '--isnorm',
            action='store_true',
            help='check whther the dataset is dcm files')
        parser.add_argument(
            '--isResetValAboveSoildCt',
            action='store_true',
            help='set all value higher than Solid Ct to Solid Ct'
        )
        parser.add_argument('--csvRefPath', type=str,
                             default='datasets/RockXCT_fractionalPorosity/fractionalPorosity.csv',
                               help='specify where to read the fractional porosity')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

        self.csvRefPath = opt.csvRefPath;
        self.csvPorosityRef = pd.read_csv(self.csvRefPath)

        self.solid_ct = opt.solid_ct
        self.water_ct = opt.water_ct
        self.air_ct = opt.air_ct

        self.isResetValAboveSoildCt = opt.isResetValAboveSoildCt
        self.isTrain = opt.isTrain

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        ds = dcmread(AB_path)
        # 提取像素數據
        px_arr = np.array(ds.pixel_array)
        # CT value
        AB_Hu = px_arr * ds.RescaleSlope + ds.RescaleIntercept
        AB_Hu = torch.tensor(AB_Hu, dtype=torch.float).unsqueeze(0)

        if (self.isResetValAboveSoildCt):
            AB_Hu[AB_Hu>self.solid_ct] = self.solid_ct

        AB_Hu = Rescale0_1(AB_Hu, self.solid_ct, self.water_ct, self.air_ct)

        # split AB image into A and B
        w = AB_Hu.shape[2]
        w2 = int(w / 2)
        A = AB_Hu[:,:,:w2]
        B = AB_Hu[:,:,w2:]

        # rescale to -1 ~ 1
        A = 2 * A - 1
        B = 2 * B - 1

        if self.isTrain:
            # attach fractional porosity
            # resolve the target by file path
            dirname, filename = AB_path.split('/')[-2:]
            index_fractionalPorosity = int(filename.split('.')[0])
            core, section, group = dirname.split('_')
            section = int(section)
            group = int(group)
            fractionalPorosity = self.csvPorosityRef[(self.csvPorosityRef['Core ID'] == core) & 
                                (self.csvPorosityRef['Section (m)'] == section) & 
                                (self.csvPorosityRef['Group'] == group)].iloc[index_fractionalPorosity]['Fractional porosity']
            fractionalPorosity = torch.tensor(fractionalPorosity,dtype=torch.float)
            
            return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path, 'fractionalPorosity':fractionalPorosity}
        else:
            return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}
            

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
    
