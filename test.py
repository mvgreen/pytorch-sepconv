import argparse
from generate import Converter_other
from TestModule import Generic_other
from model import SepConvNet
import torch

parser = argparse.ArgumentParser(description='SepConv Pytorch')

# parameters
parser.add_argument('--input', type=str, default='./Interpolation_testset/input')
parser.add_argument('--gt', type=str, default='./Interpolation_testset/gt')
parser.add_argument('--output', type=str, default='./output_sepconv_pytorch_0/result')
parser.add_argument('--checkpoint', type=str, default='./output_sepconv_pytorch_0/checkpoint/model_epoch010.pth')
parser.add_argument('--cpfolder', type=str, default='./output_sepconv_pytorch_0/checkpoint/')
parser.add_argument('--generate', type=bool, default=False)


def generate_images(input_dir, output_dir, ckpt):
    print("Reading Test DB...")
    TestDB = Converter_other(input_dir)
    print("Loading the Model...")
    checkpoint = torch.load(ckpt)
    kernel_size = checkpoint['kernel_size']
    model = SepConvNet(kernel_size=kernel_size)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    model.epoch = checkpoint['epoch']
    model.cuda()

    print("Test Start...")
    TestDB.Test(model, output_dir)


def test_images(input_dir, gt_dir, output_dir, ckpt_folder):
    print("Reading Test DB...")
    TestDB = Generic_other(input_dir, gt_dir)
    print("Loading the Model...")
    
    ckpt_list = sorted(os.listdir(ckpt_folder))
    
    for i in range(len(ckpt_list)):
        checkpoint = torch.load(ckpt_folder + ckpt_list[i])
        kernel_size = checkpoint['kernel_size']
        model = SepConvNet(kernel_size=kernel_size)
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)
        model.epoch = checkpoint['epoch']
        model.cuda()

        print("Test Start...")
        TestDB.Test(model, output_dir)


def main():
    args = parser.parse_args()
    input_dir = args.input
    gt_dir = args.gt
    output_dir = args.output
    ckpt = args.checkpoint
    generate = args.generate
    ckpt_folder = args.cpfolder
    
    if generate == True:
        generate_images(input_dir, output_dir, ckpt)
    else:
        test_images(input_dir, gt_dir, output_dir, ckpt_folder)

    
if __name__ == "__main__":
    main()
