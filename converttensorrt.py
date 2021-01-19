import torch
from models.with_mobilenet import PoseEstimationWithMobileNet #my particular net architecture
from modules.load_state import load_state
from torch2trt import torch2trt #import library
import time

checkpoint_path='checkpoint_iter_370000.pth' #your trained weights path

net = PoseEstimationWithMobileNet()#my particular net istance
checkpoint = torch.load(checkpoint_path, map_location='cuda')
load_state(net, checkpoint)#load your trained weights path
net.cuda().eval()

data = torch.rand((1, 3, 256, 344)).cuda()#initialize a random tensor with the shape of your input data

model_trt = torch2trt(net, [data]) #IT CREATES THE COMPILED VERSION OF YOUR MODEL, IT TAKES A WHILE

torch.save(model_trt.state_dict(), 'net_trt.pth') #TO SAVE THE WEIGHTS OF THE COMPILED MODEL WICH ARE DIFFERENT FROM THE PREVIOUS ONES


#HERE IT IS HOW TO UPLOAD THE MODEL ONCE YOU HAVE COMPILED IT LIKE IN MY CASE THAT I HAVE ALREADY COMPILED IT