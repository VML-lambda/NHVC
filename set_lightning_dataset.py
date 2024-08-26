import os
import random
import numpy as np
from imageio import imread
from skimage.transform import resize

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms.functional as tf

import utils
from utils import im2float


class Div2kDataset(Dataset):
    """
        Loads images a folder with augmentation for generator training

        Class initialization parameters
        -------------------------------
        data_path:          folder containing images
        channel:            color channel to load (0, 1, 2 for R, G, B, None for all 3), default None
        batch_size:         number of images to pass each iteration, default 1
        image_res:          2d dimensions to pad/crop the image to for final output, default (1080, 1920)
        homography_res:     2d dimensions to scale the image to before final crop to image_res 
                            for consistent resolutions (crops to preserve input aspect ratio), default (880, 1600)
        crop_to_homography: if True, only crops the image instead of scaling to get to target homography resolution, default False
        flip_vert:          True to augment with vertical flipping, default True
        flip_horz:          True to augment with horizontal flipping, default True
    """
    def __init__(self, config, mode):
        super().__init__()

        data_path = os.path.join(config.data_path, mode)
        if not os.path.isdir(data_path):
            raise NotADirectoryError(f'Data folder: {data_path}')

        self.data_path = data_path
        self.channels = {'r': 0, 'g': 1, 'b': 2, 'n': None}
        self.channel = self.channels[config.channel]
        self.image_res = config.image_res
        self.homography_res = config.homography_res
        self.crop_to_homography = config.crop_to_homography
        self.flip_vert = True if mode == 'train' else False
        self.flip_horz = True if mode == 'train' else False

        # Get image filename
        self.img_names = self.get_image_filenames()
        self.img_names.sort()

        # Set augmentation status (Horizontal flip, Vertical flip)
        self.augments = []
        self.augments.append(self.augment_vert) if self.flip_vert else None
        self.augments.append(self.augment_horz) if self.flip_horz else None
        self.augments_states = [fn() for fn in self.augments]

        # Create list of Image IDs with augmentation state
        self.order = ((i, ) for i in range(len(self.img_names)))
        for aug_type in self.augments:
            states = aug_type()
            self.order = ((*prev_states, s) for prev_states in self.order for s in states)
    
        # [(0, T, T), (0, T, F), (0, F, T), (0, F, F), ...
        self.order = list(self.order)


    def __getitem__(self, idx):
        filenum = self.order[idx][0]
        augments_states = self.order[idx][1:]

        # Load image and convert int to double with normalization
        img = imread(self.img_names[filenum])
        img = img[..., :] if self.channel is None else img[..., self.channel, np.newaxis]
        img = utils.im2float(img, dtype=np.float64)

        # Linearize intensity and convert to amplitude
        low_val = img <= 0.04045
        img[low_val] = 25 / 323 * img[low_val]
        img[np.logical_not(low_val)] = ((200 * img[np.logical_not(low_val)] + 11) / 211) ** (12 / 5)
        img = np.sqrt(img)

        # Move channel dim to torch convention (C, H, W)
        img = np.transpose(img, axes=(2, 0, 1))

        # Apply data augmentation
        for fn, state in zip(self.augments, augments_states):
            img = fn(img, state)

        # Normalize resolution
        input_res = img.shape[-2:]
        img = self.pad_crop_to_res(img, self.homography_res) if self.crop_to_homography else self.resize_keep_aspect(img, self.homography_res)
        img = self.pad_crop_to_res(img, self.image_res)

        # Generate Mask
        mask = np.ones(img.shape[:])
        mask = self.pad_crop_to_res(mask, self.homography_res) if self.crop_to_homography else mask
        mask = self.pad_crop_to_res(mask, self.image_res)
        
        # Return (Img, Resolution, Filename)
        return {
            "target": torch.from_numpy(img).float(),
            "mask": torch.from_numpy(mask).float(),
            "resolution": input_res,
            "filename": os.path.splitext(self.img_names[filenum])[0]
        }


    def __len__(self):
        return len(self.order)


    def get_order(self):
        return self.order


    def get_image_filenames(self):
        image_types = ['png', 'bmp', 'gif', 'jpg', 'jpeg']
        files = os.listdir(self.data_path)   
        exts = (os.path.splitext(f)[1] for f in files)
        images = [os.path.join(self.data_path, f) for e, f in zip(exts, files) if e[1:] in image_types]
        return images


    def augment_vert(self, image=None, flip=False):
        if image is None:
            return (False, True) 
        return image[..., ::-1, :] if flip else image


    def augment_horz(self, image=None, flip=False):
        if image is None:
            return (False, True) 
        return image[..., ::-1] if flip else image


    def resize_keep_aspect(self, image, target_res, pad=False):
        im_res = image.shape[-2:]
        resized_res = (int(np.ceil(im_res[1] * target_res[0] / target_res[1])), int(np.ceil(im_res[0] * target_res[1] / target_res[0])))
        image = utils.pad_image(image, resized_res, pytorch=False) if pad else utils.crop_image(image, resized_res, pytorch=False)
        image = np.transpose(image, axes=(1, 2, 0))
        image = resize(image, target_res, mode='reflect')
        return np.transpose(image, axes=(2, 0, 1))


    def pad_crop_to_res(self, image, target_res):
        padded_img = utils.pad_image(image, target_res, pytorch=False)
        cropped_img = utils.crop_image(padded_img, target_res, pytorch=False)
        return cropped_img



class VideoSet(Dataset):
    def __init__(self, config, mode):
        super().__init__()
    # def __init__(self, data_root, channel = None, image_res=(1088, 1920),
    #              homography_res=(880, 1600), crop_to_homography=False, is_training=True):
        """
        Creates a Video Septuplet object.
        Inputs.
            data_root: Root path for the Vimeo dataset containing the sep tuples.
            is_training: Train/Test.
            input_frames: Which frames to input for frame interpolation network.
        """
        self.data_root = config.video_path # .\vimeo_septuplet
        self.channels = {'r': 0, 'g': 1, 'b': 2, 'n': None}
        self.channel = self.channels[config.channel]
        self.image_res = config.image_res
        self.homography_res = config.homography_res
        self.crop_to_homography = config.crop_to_homography
        train_fn = os.path.join(self.data_root, 'sep_trainlist.txt') #
        test_fn = os.path.join(self.data_root, 'sep_vallist.txt')
        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines() # ['00001/0001','00001/0002',...]
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()

        self.training = True if mode == 'train' else False
        self.transforms = self.transform_video
        self.gain = config.gain
        self.imglist1 = [1, 3, 5, 7]
        self.imglist2 = [1, 4, 7]
        self.imglist3 = [1, 7]

    def __getitem__(self, index):
        if self.training:
            imgpath = os.path.join(self.data_root, 'sequences', self.trainlist[index])
            p_frame_idx = random.randint(2, 7)
            imgpaths = [imgpath + f'/img{1}.png', imgpath + f'/img{p_frame_idx}.png']
        else:
            imgpath = os.path.join(self.data_root, 'sequences', self.testlist[index])
            if self.gain:
                imgpaths = [imgpath + f'/img{i}.png' for i in range(1, 3)]
            else:
                imgpaths = [imgpath + f'/img{i}.png' for i in range(1, 7)]

        # Load images
        images = [imread(pth) for pth in imgpaths]
        # Data augmentation
        if self.training:
            images = self.transforms(images, self.channel, self.crop_to_homography,
                                     self.homography_res, self.image_res)
            # Random Temporal Flip
            if random.random() >= 0.5:
                images = images[::-1]
                imgpaths = imgpaths[::-1]
        else:
            images = self.transforms(images, self.channel, self.crop_to_homography,
                                     self.homography_res, self.image_res)
        
        # Generate Mask
        mask = np.ones(images[0].shape[:])
        mask = self.pad_crop_to_res(mask, self.homography_res) if self.crop_to_homography else mask
        mask = self.pad_crop_to_res(mask, self.image_res)
        
        return {
            "images": images,
            "mask": torch.from_numpy(mask).float()
        }

    def __len__(self):
        if self.training:
            return len(self.trainlist)
        else:
            return len(self.testlist)
        
    def transform_video(self, images, channel, crop_to_homography, homography_res, image_res):
        images_list = []
        for image in images:
            if channel is not None:
                image = image[..., channel, np.newaxis]
            image = im2float(image, dtype=np.float32)
            low_val = image <= 0.04045
            image[low_val] = 25 / 323 * image[low_val]
            image[np.logical_not(low_val)] = ((200 * image[np.logical_not(low_val)] + 11)
                                            / 211) ** (12 / 5)
            image = np.sqrt(image)

            image = tf.to_tensor(image)
            if crop_to_homography:
                image = self.pad_crop_to_res(image, homography_res)
            else:
                image = self.pad_crop_to_res(image, homography_res)
            image = self.pad_crop_to_res(image, image_res)
            images_list.append(image)
        return images_list
    
    def pad_crop_to_res(self, image, target_res):
        padded_img = utils.pad_image(image, target_res, pytorch=False)
        cropped_img = utils.crop_image(padded_img, target_res, pytorch=False)
        return cropped_img


class UVGDataset(Dataset):
    def __init__(self, config, mode):
        super().__init__()
    # def __init__(self, data_root, channel = None, image_res=(1088, 1920),
    #              homography_res=(880, 1600), crop_to_homography=False, is_training=True):
        """
        Creates a UVG Dataset object.
        Inputs.
            data_root: Root path for the Vimeo dataset containing the sep tuples.
            is_training: Train/Test.
            input_frames: Which frames to input for frame interpolation network.
        """
        self.data_root = '../data/UVG' # .\vimeo_septuplet
        # self.seq_list = ["Beauty", "Bosphorus", "HoneyBee", "Jockey", "ReadySteadyGo", "ShakeNDry", "YachtRide"]
        self.seq_list = ["ReadySteadyGo"]
        self.GOP = 32
        self.seq_num = 0
        self.maxIndex = config.maxIndex
        self.intra_frame = True
        self.channels = {'r': 0, 'g': 1, 'b': 2, 'n': None}
        self.channel = self.channels[config.channel]
        self.image_res = config.image_res
        self.homography_res = config.homography_res
        self.crop_to_homography = config.crop_to_homography

        self.training = True if mode == 'train' else False
        self.transforms = self.transform_image

    def __getitem__(self, index):
        self.intra_frame = True if index % self.GOP == 0 else False
        frame_num = index % self.maxIndex
        if (index % self.maxIndex == 0) and (index > 0):
            self.seq_num = self.seq_num + 1
        seq_name = self.seq_list[self.seq_num]

        if self.training:
            imgpath = os.path.join(self.data_root, seq_name, f'im00{frame_num+1:03d}.png')
        else:
            imgpath = os.path.join(self.data_root, seq_name, f'im00{frame_num+1:03d}.png')

        # Load images
        images = imread(imgpath)
        # Data augmentation
        if self.training:
            images = self.transforms(images, self.channel, self.crop_to_homography,
                                     self.homography_res, self.image_res)
            # Random Temporal Flip
            if random.random() >= 0.5:
                images = images[::-1]
                imgpaths = imgpaths[::-1]
        else:
            images = self.transforms(images, self.channel, self.homography_res, self.image_res)
        
        # Generate Mask
        mask = np.ones(images[0].shape[:])
        mask = self.pad_crop_to_res(mask, self.homography_res)
        mask = self.pad_crop_to_res(mask, self.image_res)
        
        return {
            "images": images,
            "mask": torch.from_numpy(mask).float(),
            "intra_frame": self.intra_frame,
            "sequence_name": seq_name
        }

    def __len__(self):
        if self.training:
            return len(self.trainlist)
        else:
            return len(self.seq_list) * self.maxIndex
        
    def transform_image(self, image, channel, homography_res, image_res):
        if channel is not None:
            image = image[..., channel, np.newaxis]
        image = im2float(image, dtype=np.float32)
        low_val = image <= 0.04045
        image[low_val] = 25 / 323 * image[low_val]
        image[np.logical_not(low_val)] = ((200 * image[np.logical_not(low_val)] + 11)
                                        / 211) ** (12 / 5)
        image = np.sqrt(image)

        image = tf.to_tensor(image)
        image = self.pad_crop_to_res(image, homography_res)
        image = self.pad_crop_to_res(image, image_res)
        return image
    
    def pad_crop_to_res(self, image, target_res):
        padded_img = utils.pad_image(image, target_res, pytorch=False)
        cropped_img = utils.crop_image(padded_img, target_res, pytorch=False)
        return cropped_img


if __name__ == "__main__":
    class config:
        def __init__(self):
            self.video_path = '../data/VideoSet/'
            self.channel = 'r'
            self.batch_size = 1
            self.image_res = (1088, 1920)
            self.homography_res = (880, 1600)
            self.crop_to_homography = True

    train_config = config()

    dataset = VideoSet(train_config, 'train')
    dataloader = DataLoader(dataset, batch_size=train_config.batch_size, shuffle=False, pin_memory=True)

    for k, data in enumerate(dataloader):
    # get target image
        images = [img for img in data['images']]
        images = torch.stack(images, dim=2)
        if k==0 or k==1 or k==2 or k==3:
            print(len(images))
            import skimage.io as sio
            target_amp = target_amp["Target"].numpy().squeeze()
            target_amp = np.transpose(target_amp, axes=(1, 2, 0))
            print(target_amp.shape)
            sio.imsave(f'lightning_test_{k}.png',target_amp)
