from PIL import Image
import torch
from torchvision import transforms
from math import log10
from torchvision.utils import save_image as imwrite
from torch.autograd import Variable
import os


def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


class Middlebury_eval:
    def __init__(self, input_dir):
        self.im_list = ['Army', 'Backyard', 'Basketball', 'Dumptruck', 'Evergreen', 'Grove', 'Mequon', 'Schefflera', 'Teddy', 'Urban', 'Wooden', 'Yosemite']

class Generic_other:
    def __init__(self, input_dir, gt_dir):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.transform = transforms.Compose([transforms.ToTensor()])
    
    
    def Test(self, model, output_dir, logfile=None, output_name='output.png'):
        av_psnr = 0
        print('{:<7s}{:<3d}'.format('Epoch: ', model.epoch.item()) + '\n')
        
        sample_list = sorted(os.listdir(self.input_dir))
        for idx in range(len(sample_list)):
            item = sample_list[idx]
            #if not os.path.exists(output_dir + '/' + item):
            #    os.makedirs(output_dir + '/' + item)
            im_1 = Image.open(self.input_dir + '/' + item + '/frame1.png')
            im_2 = Image.open(self.gt_dir + '/' + item + '/frame2.png')
            im_3 = Image.open(self.input_dir + '/' + item + '/frame3.png')
            frame_in_1 = to_variable(self.transform(im_1).unsqueeze(0))
            frame_in_3 = to_variable(self.transform(im_3).unsqueeze(0))
            gt = to_variable(self.transform(im_2).unsqueeze(0))
            frame_out = model(frame_in_1, frame_in_3)
            psnr = -10 * log10(torch.mean((gt - frame_out) * (gt - frame_out)).item())
            av_psnr += psnr
            #imwrite(frame_out, output_dir + '/' + item + '/' + output_name, range=(0, 1))
            #msg = '{:<15s}{:<20.16f}'.format(item + ': ', psnr) + '\n'
            #print(msg, end='')
            if logfile is not None:
                logfile.write(msg)
            im_1.close()
            im_2.close()
            im_3.close()
            
        av_psnr /= len(sample_list)
        msg = '{:<15s}{:<20.16f}'.format('Average: ', av_psnr) + '\n'
        print(msg, end='')
        if logfile is not None:
            logfile.write(msg)
            
    
class Middlebury_other:
    def __init__(self, input_dir, gt_dir):
        self.im_list = ['Beanbags', 'Dimetrodon', 'DogDance', 'Grove2', 'Grove3', 'Hydrangea', 'MiniCooper', 'RubberWhale', 'Urban2', 'Urban3', 'Venus', 'Walking']
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.input0_list = []
        self.input1_list = []
        self.gt_list = []
        for item in self.im_list:
            self.input0_list.append(to_variable(self.transform(Image.open(input_dir + '/' + item + '/frame10.png')).unsqueeze(0)))
            self.input1_list.append(to_variable(self.transform(Image.open(input_dir + '/' + item + '/frame11.png')).unsqueeze(0)))
            self.gt_list.append(to_variable(self.transform(Image.open(gt_dir + '/' + item + '/frame10i11.png')).unsqueeze(0)))

    def Test(self, model, output_dir, logfile=None, output_name='output.png'):
        av_psnr = 0
        if logfile is not None:
            logfile.write('{:<7s}{:<3d}'.format('Epoch: ', model.epoch.item()) + '\n')
        for idx in range(len(self.im_list)):
            if not os.path.exists(output_dir + '/' + self.im_list[idx]):
                os.makedirs(output_dir + '/' + self.im_list[idx])
            frame_out = model(self.input0_list[idx], self.input1_list[idx])
            gt = self.gt_list[idx]
            psnr = -10 * log10(torch.mean((gt - frame_out) * (gt - frame_out)).item())
            av_psnr += psnr
            imwrite(frame_out, output_dir + '/' + self.im_list[idx] + '/' + output_name, range=(0, 1))
            msg = '{:<15s}{:<20.16f}'.format(self.im_list[idx] + ': ', psnr) + '\n'
            print(msg, end='')
            if logfile is not None:
                logfile.write(msg)
        av_psnr /= len(self.im_list)
        msg = '{:<15s}{:<20.16f}'.format('Average: ', av_psnr) + '\n'
        print(msg, end='')
        if logfile is not None:
            logfile.write(msg)
            
