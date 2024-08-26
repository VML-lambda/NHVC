import os
import argparse


class Config:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Training Configuration")
        parser.add_argument('--train',          default=False,              action='store_true', help="train mode")
        parser.add_argument('--stage_mode',     default='stage3',           type=str, choices=["stage1", "stage2", "stage3"])
        parser.add_argument('--channel',        default='r',                type=str, choices=["r", "g", "b"])
        parser.add_argument('--gain',           default=False,              action='store_true', help="Enable gain if used, otherwise False")
        parser.add_argument('--lmbda',          default=130,                type=int) # 18, 35, 67, 130, 250, 480, 2000
        parser.add_argument('--gain_mode',      default='',                 type=str, choices=["", "low", "mid", "high"])
        parser.add_argument('--data_path',      default='./data/DIV2K/',    type=str)

        # if video compression
        parser.add_argument('--level',          default=2,                  type=int) # 0, 1, 2, 3, 4, 5
        parser.add_argument('--intra_mode',     default=False,              action='store_true', help="Enable intra mode if used, otherwise False")
        parser.add_argument('--eval_s',         default=3,                  type=int)
        parser.add_argument('--video_path',     default='./data/VideoSet/', type=str, choices=["r", "g", "b"])

        config = parser.parse_args()
        
        self.stage_mode = config.stage_mode
        self.channel = config.channel
        
        self.gain = config.gain
        self.train = config.train
        # precision of 0.0001
        self.single_lmbda = config.lmbda if self.gain == False and not self.stage_mode == 'stage1' else 0

        # run_id
        self.model_name = 'IEEE_VR_NHVC' # NHVC-noQE
        self.load_model = False
        self.mode = config.gain_mode
        self.level = config.level  
        self.allIntra = config.intra_mode
        self.eval_s = config.eval_s
        # Seed
        self.seed = 1234

        # Devices
        self.device = "gpu"
        self.gpu = [0]

        # Data loader configuration
        self.batch_size = 1

        # Data set configuration
        self.data_path = config.data_path
        self.video_path = config.video_path
        self.image_res = (1088, 1920)
        self.homography_res = (880, 1600)
        self.crop_to_homography = True if self.train == False else False
        self.maxIndex = 96

        # SLM configuratin
        self.propdist = 20
        self.featuresize = 6.4
        
        self.generation = True if self.stage_mode == 'stage1' else False
        self.image_compression = True if self.stage_mode == 'stage2' else False
        self.video_compression = True if self.stage_mode == 'stage3' else False
        
        if self.video_compression:
            if self.gain:
                model_ckpts = os.listdir(f'./experiment/{self.model_name}/{self.channel}/stage2-{self.mode}')
                self.model_ckpt = [os.path.join(f'./experiment/{self.model_name}/{self.channel}/stage2-{self.mode}', model_ckpt) for model_ckpt in model_ckpts if 'loss' in model_ckpt][0]
            else:
                model_ckpts = os.listdir(f'./experiment/{self.model_name}/{self.channel}/stage2/lmbda_{self.single_lmbda}')
                self.model_ckpt = [os.path.join(f'./experiment/{self.model_name}/{self.channel}/stage2/lmbda_{self.single_lmbda}', model_ckpt) for model_ckpt in model_ckpts if 'loss' in model_ckpt][0]

        # Training configuration
        self.epoches = 50 if self.gain else 40
        self.lr = 1e-4
        self.lr_step_size = 5
        self.aux_lr = 1e-3
        
        

        