'''
Real measurement recovery
---------------------------------------------------------------------------------
Real measurement size in our paper: 768*1024
Use segmentation function to transform the measurement into 12 blocks.(12*256*256)
Use block2img concatenate blocks to a whole HSI
---------------------------------------------------------------------------------
'''
from Net import Unet_3d
from utils import *
from segmentation import *
import torch.optim as optim
import torch.nn as nn
import torch
import scipy.io as scio
import time
import datetime
import os
import numpy as np
from torch.autograd import Variable
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if not torch.cuda.is_available():
    raise Exception('NO GPU!')

test_path = "../../Data/real_data_Cheng/2022_1.5/data_fruit.mat"
mask_3d_shift, meas_real = Load_real_data(test_path)
mask_use = mask_3d_shift
meas_use = meas_real
meas_block = meas_segmentation(meas_use) 
mask_block_batch = mask_segmentation(mask_use)
last_train = 88                      
model_save_filename = '2021_11_10_17_50_30'         
batch_size = 12
model = Unet_3d(1, 1).cuda()

if last_train != 0:
    model = torch.load('./model/' + model_save_filename + '/model_epoch_{}.pth'.format(last_train))    

def test(epoch):
    test_PhiTy = gen_real_phity(meas_block, mask_block_batch)
    model.eval()
    begin = time.time()
    with torch.no_grad():
        model_out = model(test_PhiTy)
    end = time.time()
    model_out = torch.squeeze(model_out)
    out = block2image(model_out)
    pred = np.transpose(out.detach().cpu().numpy(), (1, 2, 0)).astype(np.float32)
    print('===> time: {:.2f}'.format((end - begin)))
    return pred
    
     
def main():
    if model_save_filename == '':
        date_time = str(datetime.datetime.now())
        date_time = time2file_name(date_time)
    else:
        date_time = model_save_filename
    result_path = 'recon' + '/' + date_time
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    epoch = last_train
    pred = test(epoch)
    
    name = result_path + '/' + 'Test_real_{}'.format(last_train) + '.mat'
    scio.savemat(name, {'pred': pred})
        
if __name__ == '__main__':
    main()    
    

