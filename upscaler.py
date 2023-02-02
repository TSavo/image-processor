import os
import cv2
import numpy as np
import glob
import argparse

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact


class Upscaler:
    def __init__(self, model_name="RealESRGAN_x4plus", model_path=None, denoise_strength=1, tile=512, tile_pad=10, pre_pad=0, fp32=False,
                 gpu_id=0, face_enhance=False, outscale=1, ext='auto', suffix=''):
        self.model_name = model_name
        self.model_path = model_path
        self.denoise_strength = denoise_strength
        self.tile = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.fp32 = fp32
        self.gpu_id = gpu_id
        self.face_enhance = face_enhance
        self.outscale = outscale
        self.ext = ext
        self.suffix = suffix
        self.upsampler = None
        self.face_enhancer = None
    
    
        # determine models according to model names
        self.model_name = self.model_name.split('.')[0]
        if self.model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            self.netscale = 4
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
        elif self.model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            self.netscale = 4
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
        elif self.model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            self.netscale = 4
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
        elif self.model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            self.netscale = 2
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
        elif self.model_name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
            self.netscale = 4
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
        elif self.model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
            self.netscale = 4
            file_url = [
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
            ]

        # determine model paths
        if self.model_path is not None:
            model_path = self.model_path
        else:
            model_path = os.path.join('weights', self.model_name + '.pth')
            if not os.path.isfile(model_path):
                ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
                for url in file_url:
                    # model_path will be updated
                    model_path = load_file_from_url(
                        url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

        # use dni to control the denoise strength
        dni_weight = None
        if self.model_name == 'realesr-general-x4v3' and self.denoise_strength != 1:
            wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
            model_path = [model_path, wdn_model_path]
            dni_weight = [self.denoise_strength, 1 - self.denoise_strength]

        # restorer
        self.upsampler = RealESRGANer(
            scale=self.netscale,
            model_path=model_path,
            dni_weight=dni_weight,
            model=model,
            tile=self.tile,
            tile_pad=self.tile_pad,
            pre_pad=self.pre_pad,
            half=not self.fp32,
            gpu_id=self.gpu_id)

        if self.face_enhance:  # Use GFPGAN for face enhancement
            from gfpgan import GFPGANer
            self.face_enhancer = GFPGANer(
                model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                upscale=self.netscale,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=self.upsampler)
    def __call__(self, img):

        print('Testing')
        try:
            if self.face_enhance:
                _, _, output = self.face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            else:
                output, _ = self.upsampler.enhance(img, outscale=self.netscale)
        except RuntimeError as error:
            print('Error', error)
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
            return None
        else:
            return output

class TargetUpscaler:
    def __init__(self, target_size):
        self.target_size = target_size
        self.up2x = Upscaler(model_name=f'RealESRGAN_x2plus', tile=256)
        self.up4x = Upscaler(model_name=f'RealESRGAN_x4plus', tile=512)

    def __call__(self, image):
        while image.shape[0] < self.target_size and image.shape[1] < self.target_size:
            if image.shape[0] >= 4096 or image.shape[1] >= 4096:
                image = self.up2x(image)
            else:
                image = self.up4x(image)
            #if the image is less than 4096 on the long side, use the 2x upscaler
            if image is None:
                return None
        #if the longest edge is > target_sizze  resize it to be tarrget_ssizze along that edge
        if image.shape[0] > self.target_size or image.shape[1] > self.target_size:
            if image.shape[0] > image.shape[1]:
                image = cv2.resize(image, (int(image.shape[1] * 8192 / image.shape[0]), 8192))
            else:
                image = cv2.resize(image, (8192, int(image.shape[0] * 8192 / image.shape[1])))
        return image
