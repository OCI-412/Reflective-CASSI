import torch
import numpy
'''
-------------------------------------------------------------------------
Function: meas_segmentation(), mask_segmentation()
segmentation of big size measurement and mask,
transform big size meas and mask to several blocks with size of 256 * 256.
-------------------------------------------------------------------------
Function: block2image()
concatenate recovered several blocks to a whole image,
which have the same size of original measurement.
-------------------------------------------------------------------------
'''

def meas_segmentation(meas):
    meas_use = meas[40:-40,45:-45]# 768-80, 1024-90
    meas_block = torch.zeros(12, 256, 256).cuda().float()
    for i in range(3):
        meas_block[i*4+0,:,:] = meas_use[i*216:i*216+256,0:256]
        meas_block[i*4+1,:,:] = meas_use[i*216:i*216+256,226:482]
        meas_block[i*4+2,:,:] = meas_use[i*216:i*216+256,452:708]
        meas_block[i*4+3,:,:] = meas_use[i*216:i*216+256,678:934]

    return meas_block  ##12,256,256

def mask_segmentation(mask_3d_shift):
    mask_use = mask_3d_shift[:,40:-40,45:-45] #27, 768-80, 1024-90
    mask_block = torch.zeros(12, 27, 256, 256).cuda().float()
    for i in range(3):
        mask_block[i*4+0,:,:,:] = mask_use[:,i*216:i*216+256,0:256]
        mask_block[i*4+1,:,:,:] = mask_use[:,i*216:i*216+256,226:482]
        mask_block[i*4+2,:,:,:] = mask_use[:,i*216:i*216+256,452:708]
        mask_block[i*4+3,:,:,:] = mask_use[:,i*216:i*216+256,678:934]

    return mask_block  ##12,27,256,256

def block2image(recon_block):
    HSI_img = torch.zeros(27, 768-80, 1024-90)
    #row 1#
    HSI_img[:, 0:236, 0:241] = recon_block[0, :, 0:236, 0:241]
    HSI_img[:, 0:236, 241:467] = recon_block[1, :, 0:236, 15:241]
    HSI_img[:, 0:236, 467:693] = recon_block[2, :, 0:236, 15:241]
    HSI_img[:, 0:236, 693:934] = recon_block[3, :, 0:236, 15:256]
    #row 2#
    HSI_img[:, 236:452, 0:241] = recon_block[4, :, 20:236, 0:241]
    HSI_img[:, 236:452, 241:467] = recon_block[5, :, 20:236, 15:241]
    HSI_img[:, 236:452, 467:693] = recon_block[6, :, 20:236, 15:241]
    HSI_img[:, 236:452, 693:934] = recon_block[7, :, 20:236, 15:256]
    #row 3#
    HSI_img[:, 452:688, 0:241] = recon_block[8, :, 20:256, 0:241]
    HSI_img[:, 452:688, 241:467] = recon_block[9, :, 20:256, 15:241]
    HSI_img[:, 452:688, 467:693] = recon_block[10, :, 20:256, 15:241]
    HSI_img[:, 452:688, 693:934] = recon_block[11, :, 20:256, 15:256]

    return HSI_img#27, 668, 934
