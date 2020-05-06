from PIL import Image
import torch
from torchvision import transforms
from math import log10
from torchvision.utils import save_image as imwrite
from torch.autograd import Variable
import os
import re

def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


class Converter_eval:
    def __init__(self, input_dir):
        self.im_list # записать список имен папок с данными


class Converter_other:

    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.transform = transforms.Compose([transforms.ToTensor()])
	
    def Test(self, model, output_dir, logfile=None):
        file_list = sorted(os.listdir(self.input_dir))
        for i in range(1, len(file_list) - 1):
            in_filename, in_ext = re.split('\.', file_list[i])
            out_filename = in_filename + 'a' + in_ext
            
            im1 = Image.open(self.input_dir + '/' + file_list[i])
            im2 = Image.open(self.input_dir + '/' + file_list[i+1])
            img1 = to_variable(self.transform(im1).unsqueeze(0))
            img2 = to_variable(self.transform(im2).unsqueeze(0))
            
            frame_out = model(img1, img2)
            imwrite(frame_out, output_dir + '/' + out_filename)
            im1.close()
            im2.close()
