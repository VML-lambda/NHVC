from typing import Optional

from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule

from set_lightning_dataset import Div2kDataset, VideoSet, UVGDataset


class LightningDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config.batch_size
        self.setup()


    def prepare_data(self):
        """
            This method is used to define the processes that are meant to be performed by only one GPU. 
            It’s usually used to handle the task of downloading the data. 
        """
        pass


    def setup(self, stage: Optional[str] = None):
        """
            This method is used to define the process that is meant to be performed by all the available GPU. 
            It’s usually used to handle the task of loading the data. 
        """
        if self.config.generation or self.config.image_compression:
            self.train_dataset = Div2kDataset(self.config, 'train')
            self.valid_dataset = Div2kDataset(self.config, 'valid')
            self.test_dataset =  Div2kDataset(self.config, 'valid')
            # self.test_dataset =  UVGDataset(self.config, 'test')
        else: 
            self.train_dataset = VideoSet(self.config, 'train')
            self.valid_dataset = VideoSet(self.config, 'valid')
            self.test_dataset =  UVGDataset(self.config, 'test')


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size,
            num_workers=16, 
            shuffle=True
        )


    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset, 
            batch_size=self.batch_size, 
            num_workers=16,
            shuffle=False
        )


    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False
        )


# if __name__ == "__main__":
#     import skimage.io as sio
#     import numpy as np

#     class config:
#         def __init__(self):
#             self.batch_size = 1

#             self.data_path = './data/DIV2K/'
#             self.channel = None
#             self.image_res = (1072, 1920)
#             self.homography_res = (880, 1600)
#             self.crop_to_homography = True

#     train_config = config()
#     data_module = LightningDataModule(train_config)
#     data_module.setup()
#     val_dataloader = data_module.val_dataloader()

#     for k, target in enumerate(val_dataloader):
#     # get target image
#         target_amp, target_res, target_filename = target
#         if k==0 or k==1 or k==2 or k==3:
#             target_amp = target_amp["target"].numpy().squeeze()
#             target_amp = np.transpose(target_amp, axes=(1, 2, 0))
#             print(target_amp.shape)
#             sio.imsave(f'lightning_test_{k}.png',target_amp)
