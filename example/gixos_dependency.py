from tiled.client import from_profile
c = from_profile('opls')
import numpy as np
import matplotlib.pyplot as plt
from databroker import Broker
import copy

import pyFAI, pyFAI.detectors, pyFAI.azimuthalIntegrator

def testfunction(a,b):
    c = a+b
    print(c)

def loadgixos_ai(sample_id, chamber_id=False, mode = "single", roi_y = 100, roi_dy=2, roi_x = 6.5, pxsize = 172e-6, sdd = 627/1000, image_label = 'pilatus100k_image'):
    # this is to load 1d line cut from a p100k
    # sample dataset
    db = Broker(c)
    h_sample = db[sample_id]
    gixos_3d_sample = list(h_sample.data(image_label)) # these images are 3d data while the 1st dimention is the stack of the 2d tiff, 
    if chamber_id:
        h_chamber = db[chamber_id]
        gixos_3d_chamber = list(h_chamber.data(image_label))
    else:
        gixos_3d_chamber = [np.zeros((gixos_3d_sample[0].shape[0],gixos_3d_sample[0].shape[1],gixos_3d_sample[0].shape[2]))] *len(gixos_3d_sample)
        print('no chamber data')

    # prepare
    result = {'id': sample_id, 'bkg_id': chamber_id, 'energy': h_sample.start['energy'], 'wavelength': 12.39842 / (h_sample.start['energy']/ 1000), 'alpha': 0, 'beta': 0, 'tth': 0, 'gixos_1d': [], 'gixos_2d':[]}
    ai_lst = []
    
    if mode == "single":
        print("process individual frame")
        result['tth'] = np.array(list(h_sample.data('geo_stth')))
        result['alpha'] = np.array(list(h_sample.data('geo_alpha')))
        result['beta'] = np.array(list(h_sample.data('geo_beta')))
        #print('energy = ', result['energy'])
        #print('wavelength = ', result['wavelength'])
        #print('alpha = ', result['alpha'])
        #print('2theta = ', result['tth'])
        #print('beta = ', result['beta'])
        #print(sdd)
        for i,(sample_image,chamber_image) in enumerate(zip(gixos_3d_sample,gixos_3d_chamber)):
            gixos_3d_sample[i] = np.sum(sample_image, axis = 0) # these images are 3d data while the 1st dimention is the stack of the 2d tiff, squeeze into 2d now
            gixos_3d_chamber[i] =  np.sum(chamber_image, axis = 0)      
            result['gixos_2d'].append(gixos_3d_sample[i] - gixos_3d_chamber[i])
            # gixos 1d linecut as 2d array, n_d1 = 1
            result['gixos_1d'].append(np.sum(result['gixos_2d'][i][roi_y-roi_dy:roi_y+roi_dy+1, :], axis = 0, keepdims = True))
            #print(result['gixos_1d'][i].ndim)
            #print(result['gixos_1d'][i].shape)
            # set azimuthal integrator
            det = pyFAI.detectors.Detector(pxsize*(roi_dy*2+1), pxsize)
            det.max_shape=(result['gixos_1d'][i].shape[0],result['gixos_1d'][i].shape[1])
            ai_lst.append(pyFAI.azimuthalIntegrator.AzimuthalIntegrator(dist=sdd, detector=det, wavelength=result['wavelength']/10**10))
            ai_lst[i].poni1 = 0  # poni1 is along dim1
            ai_lst[i].poni2 = roi_x*pxsize # poni2 is along dim2
            ai_lst[i].rot1 = (result['alpha'][i]+result['beta'][i])/180*np.pi # rot1 around dim1 axis, towards direction of dim2
            ai_lst[i].rot2 = -result['tth'][i]/180*np.pi # rot2 is around dim2 axis, towards the negative of dim1 (does not matter here since it is one dimension)
            #print(ai_lst[i])
    elif mode == "sum":
        print("sum frames of each scan")
        result['tth'] = np.mean(np.array(list(h_sample.data('geo_stth'))))
        result['alpha'] = np.mean(np.array(list(h_sample.data('geo_alpha'))))
        result['beta'] = np.mean(np.array(list(h_sample.data('geo_beta'))))
        #print(np.array(gixos_3d_sample).shape)
        gixos_3d_sample = np.sum(np.array(gixos_3d_sample), axis = 1)
        #print(np.array(gixos_3d_sample).shape)
        gixos_3d_chamber = np.sum(np.array(gixos_3d_chamber), axis = 1)
        result['gixos_2d'].append(np.sum(gixos_3d_sample, axis = 0) - np.sum(gixos_3d_chamber, axis = 0))
        result['gixos_1d'].append(np.sum(result['gixos_2d'][0][roi_y-roi_dy:roi_y+roi_dy+1, :], axis = 0, keepdims = True))
        # set azimuthal integrator
        det = pyFAI.detectors.Detector(pxsize*(roi_dy*2+1), pxsize)
        det.max_shape=(result['gixos_1d'][0].shape[0],result['gixos_1d'][0].shape[1])
        ai_lst.append(pyFAI.azimuthalIntegrator.AzimuthalIntegrator(dist=sdd, detector=det, wavelength=result['wavelength']/10**10))
        ai_lst[0].poni1 = 0  # poni1 is along dim1
        ai_lst[0].poni2 = roi_x*pxsize # poni2 is along dim2
        ai_lst[0].rot1 = (result['alpha']+result['beta'])/180*np.pi # rot1 around dim1 axis, towards direction of dim2
        ai_lst[0].rot2 = -result['tth']/180*np.pi # rot2 is around dim2 axis, towards the negative of dim1 (does not matter here since it is one dimension)
        #print(ai_lst[0])
    else:
        print("please define mode between single and sum")
    result['qxy'] = 4*np.pi*np.sin(np.deg2rad(result['tth'])/2) / result['wavelength']
    
    print("loading done")
    return result, ai_lst


def pixel2th(inputmap, D_ph, D_ph_det, px_h_0, px_v_0_est, v_0_mode = "fit", energy = 14400, det = 'p100k', orient = [-1, -1], D_rot2 = 0, D_rot3 = 0, **kwargs):
    """
    convert pixel into tth and tt
    D_ph    [mm]
    D_ph_det [mm]
    center_h
    center_v_est
    v_0_mode = "fix" or "fit"
    det = 'p100k', or lambda, eiger, perk
    orient = [-1, -1] (0,0 pixel up, larger detrot (pos>neg)), [0] for up/down, [1] for detrot direction
    D_rot2 = rotation2 [deg] detector 2theta angle from the horizontal plane (rot2 is pyFAI definition)
    D_rot3 = rotation3 [deg]  detector rotation arount the incident beam when at detrot=0, clockwise for rot3 up
    optional: mask
    output: n*m matrix for intensity, qxy and qz
    """
    # constant preparation
    planck = 12398.4
    wv = planck / energy
    qc = 0.0218
            
    px_size_pool = {'p100k': 0.172, 'lambda': 0.055, 'eiger1m': 0.075, 'perk': 0.200}
    px_size = px_size_pool[det]
    D_rot2_rad = np.radians(D_rot2)
    D_rot3_rad = np.radians(D_rot3)
    inputmap['rotmat'] = np.array([[np.cos(-D_rot3_rad), -np.sin(-D_rot3_rad)],[np.sin(-D_rot3_rad), np.cos(-D_rot3_rad)]])
    
    # find Vineyard peak, idx_V is the index of the peak in the chv array
    if v_0_mode == "fit":
        px_v_V = inputmap['chv'][np.argmax(np.sum(inputmap['mat'], axis = 1)[px_v_0_est-20:px_v_0_est+20])+px_v_0_est-20]
        #print(px_v_V)
        px_v_0 = px_v_V - orient[0] * ((D_ph_det+D_ph)*np.tan(np.arcsin(qc*wv/4./np.pi) - D_rot2_rad)/px_size)
        #print(px_v_0)
    else:
        px_v_0 = px_v_0_est
        px_v_V = int(px_v_0 + orient[0] * ((D_ph_det+D_ph)*np.tan(np.arcsin(qc*wv/4./np.pi) - D_rot2_rad)/px_size))
        
    # matrix for horizontal and vertical distance of a pixel to the beam center, in the detector frame: :,:,0 for horizontal, :,:,1 for vertical 
    px_pos_det = np.zeros([inputmap['mat'].shape[0],inputmap['mat'].shape[1],2])
    
    for i in range(px_pos_det.shape[0]):
        px_pos_det[i,:,0] = orient[1] * px_size*(inputmap['chh']-px_h_0)
    for i in range(px_pos_det.shape[1]):
        px_pos_det[:,i,1] = orient[0] * px_size*(inputmap['chv']-px_v_0)
    
    #calculate the coordinate of the pixels at the frame of the vertical slit, for rotated detector
    px_pos = np.zeros([inputmap['mat'].shape[0],inputmap['mat'].shape[1],2])

    px_pos[:,:,0] = inputmap['rotmat'][0,0]*px_pos_det[:,:,0] + inputmap['rotmat'][0,1]*px_pos_det[:,:,1]   # horizontal pos
    px_pos[:,:,1] = inputmap['rotmat'][1,0]*px_pos_det[:,:,0] + inputmap['rotmat'][1,1]*px_pos_det[:,:,1]   # vertical pos
    
    # distance projection of the pixels at the same height position as the poni pixel from the pinhole position
    D_px = np.zeros([inputmap['mat'].shape[0],inputmap['mat'].shape[1],2])
    # projection on the horizontal plane
    D_px[:,:,0] = (D_ph + D_ph_det)*np.cos(D_rot2_rad) - px_pos[:,:,1]*np.sin(D_rot2_rad) - D_ph
    # height of the pixel from the horizontal plane
    D_px[:,:,1] = (D_ph + D_ph_det)*np.sin(D_rot2_rad) + px_pos[:,:,1]*np.cos(D_rot2_rad)
    # calculate the tth and tt based on the distance projections
    inputmap['tth']=np.degrees(np.arctan(px_pos[:,:,0]/D_px[:,:,0]))+inputmap['detrot'];
    inputmap['tth'] = np.where(np.abs(inputmap['tth'])<1e-8, 1e-8, inputmap['tth'])
    tt_rad=np.arctan(D_px[:,:,1]/(D_ph*np.sin(np.radians(inputmap['detrot']))/np.sin(np.radians(inputmap['tth'])) + D_px[:,:,0]/np.cos(np.radians(inputmap['tth']-inputmap['detrot']))));

    inputmap['tt'] = np.degrees(tt_rad)
    inputmap['px_v_V'] = px_v_V
    
    return inputmap


#%% angular rebin
def tthrebin(inputmap, dtth = 0.0065, **kwargs):
    """
    rebin tth:
        with pixel splitting
    tth limit is the closest value from the limit in the original matrix
    input: n*m intensity matrix and tth matrix, tt-array
    argument: dtth = 0.008
    unit: degree
    output: n'*m' intensity matrix, m'-array for tth, n'-array for tt
    note: nan value for the tth region near specular
    note2: nan value for the tth/tt outside of the limit (limit either created or getting from the result of the ttrebin)
    do it by getting tt broder mask from ttrebin and multiply it to the mask, then temporarily set nan intensity to -1, in the end set everything outside the tth border to nan
    """
    # dtt = 0.0065A^-1 at 14.4keV, tth = 0 for Pilatus at 1500mm
    settings = { 'mask': [] }
    settings.update(kwargs)
    
    outputmap = copy.deepcopy(inputmap)
    #mask
    if settings['mask']==[]:
        mask = np.ones([inputmap['mat'].shape[0],inputmap['mat'].shape[1]])
        print('no mask')                
    else:
        mask = settings['mask']
        print('apply mask')
    
    #tt border mask from the tt-rebin: outside the border weight is zero
    if inputmap['ttborderMask']!=[]:
        mask = mask*inputmap['ttborderMask']
        print('apply tt border mask')
    
    # start
    outputmap = copy.deepcopy(inputmap)
    
    # getting tth limits of the input
    outputmap['tthlim_raw'] = np.zeros((inputmap['mat'].shape[0],2))
    for idx in range(outputmap['tthlim_raw'].shape[0]):
        finite_idx = np.where(np.isfinite(inputmap['mat'][idx,:]))
        outputmap['tthlim_raw'][idx,0] = np.min(inputmap['tth'][idx, finite_idx[0]])
        outputmap['tthlim_raw'][idx,1] = np.max(inputmap['tth'][idx, finite_idx[0]])
    
    # replace the nan into -1, their weight will be 0
    np.place(inputmap['mat'],np.isnan(inputmap['mat']), -1.0)
    np.place(outputmap['mat'],np.isnan(outputmap['mat']), 0.0)
                
    # GISAXS specular preparation: getting tth boundary around the specular
    tth_boundary = np.zeros((inputmap['mat'].shape[0],2))
    for idx in range(tth_boundary.shape[0]):
        pos_idx = np.where(inputmap['tth'][idx,:]>0)
        tth_boundary[idx,0] = np.min(inputmap['tth'][idx, pos_idx[0]])
        neg_idx = np.where(inputmap['tth'][idx,:]<0)
        if neg_idx[0].size != 0:
            tth_boundary[idx,1] = np.max(inputmap['tth'][idx, neg_idx[0]])
        else:
            tth_boundary[idx,1] = tth_boundary[idx,0] 
    
    # create rebinned tth array from the lower and the upper lim of the initial tth
    tthlim = [np.floor(np.min(inputmap['tth'])/dtth)*dtth, np.ceil(np.max(inputmap['tth'])/dtth)*dtth]
    outputmap['tth'] = np.array([np.arange(tthlim[0]-2*dtth, tthlim[1]+dtth*2, dtth)])
    # prepare for pixel splitting: idx, stat and the rebinned intensity matrix
    # idx :,:,0 is the index of the upper edge of the tth bin where each pixel belongs to; 
    # idx :,:,1 the fraction of the pixel into the upper edge of the tth bin, the rest goes to the lower edge
    outputmap['idx'] = np.zeros([inputmap['mat'].shape[0],inputmap['mat'].shape[1],2])
    # stat: statistics matrix for the rebinning: accumulated fraction into each rebined pixel, to be used for normalization, all values set to zero
    outputmap['stat'] = np.zeros([inputmap['mat'].shape[0],outputmap['tth'].shape[1]])
    # result matrix for intensity, all values set to 0
    outputmap['mat'] = np.zeros([inputmap['mat'].shape[0],outputmap['tth'].shape[1]])
    
    # pixel splitting:
    #   1. use np.digitize find the upper edge index in the rebinned matrix for each original pixel
    #   2. for each original pixel, compute its fraction for the upper edge index
    #   3. split the fraction of each pixel into the upper and the lower edge in the rebinning statistics matrix; the values in the rebinning statistics matrix starts accumulation
    #   4. split the intensity of each pixel into the upper and the lower edge in the rebinned matrix; the intensity in the rebinned matrix starts accumulation
    # prepare function
    create_digitize = np.digitize
    for i in range(inputmap['tth'].shape[0]):
        outputmap['idx'][i,:,0] = create_digitize(inputmap['tth'][i,:], outputmap['tth'][0,:], right = True)
        #print(outputmap['idx'][i,:,0])
        for j in range(inputmap['tth'].shape[1]):
            idx_bins = int(outputmap['idx'][i,j,0])
            #print([i, j, idx_bins])
            outputmap['idx'][i,j,1] = 1-(outputmap['tth'][0,idx_bins]-inputmap['tth'][i,j])/dtth
            outputmap['stat'][i,idx_bins] = outputmap['stat'][i,idx_bins] + outputmap['idx'][i,j,1]*mask[i,j]
            outputmap['stat'][i,idx_bins-1] = outputmap['stat'][i,idx_bins-1] + (1-outputmap['idx'][i,j,1])*mask[i,j]
            outputmap['mat'][i,idx_bins] = outputmap['mat'][i,idx_bins] + inputmap['mat'][i,j]*outputmap['idx'][i,j,1]*mask[i,j]
            outputmap['mat'][i,idx_bins-1] = outputmap['mat'][i,idx_bins-1] + inputmap['mat'][i,j]*(1-outputmap['idx'][i,j,1])*mask[i,j]
            #print([i, outputmap['stat'][i,213]])

    # prepare function
    create_argwhere = np.argwhere
    # normalization by the statistics, everything with no pixel in is counted as negative (-1e6)
    np.place(outputmap['stat'], outputmap['stat']==0, -1)    
    empty_cell = create_argwhere(outputmap['stat']<0)
    for y,x in empty_cell:
        outputmap['mat'][y,x] = 1e6

    outputmap['mat'] = outputmap['mat'] / outputmap['stat']    
    # test = copy.deepcopy(outputmap['mat'])
    # prepare function
    create_zeros = np.zeros
    create_interp = np.interp
    # interpolation for all these negative values
    for i in range(outputmap['mat'].shape[0]):
        empty_cell = create_argwhere(outputmap['mat'][i,:]<-1e3)[:,0]
        filled_cell = create_argwhere(outputmap['mat'][i,:]>-1e3-(1e-5))[:,0]
        filled_value = create_zeros(len(filled_cell))
        for j in range(len(filled_cell)):
            filled_value[j] = outputmap['mat'][i,filled_cell[j]]
        miss_values = create_interp(empty_cell, filled_cell, filled_value)        
        for j in range(len(empty_cell)):
            outputmap['mat'][i,empty_cell[j]]=miss_values[j]       
        # set mat element within tth boundary to nan for GISAXS
        #np.place(outputmap['mat'][i,:],(outputmap['tth'] - tth_boundary[i,0])*(outputmap['tth'] - tth_boundary[i,1])<0, np.nan)
        # set mat element beyond tth limit to nan
        np.place(outputmap['mat'][i,:],(outputmap['tth'] - outputmap['tthlim_raw'][i,0])*(outputmap['tth'] - outputmap['tthlim_raw'][i,1])>0, np.nan)
    
    del empty_cell, miss_values, filled_cell, filled_value, i, j 
    return outputmap

def ttrebin(inputmap, dtt = 0.0065, **kwargs):
    """
    rebin tt, when tt is a n*m-matrix:
        with pixel splitting
    tt limit is the closest value from the limit in the original matrix
    input: n*m matrix for intensity, tth, tt
    argument: dtt = 0.0065
    unit: A^-1
    output: n*m matrix for intensity and tth, n-array for tt
    works with the same mechanism as the tth rebin
    also output the tt limit into the output field and set the int value outside the tt border to nan for each column
    output also a tt border mask for the next step
    """
    settings = { 'mask': [] }
    settings.update(kwargs)
    
    outputmap = copy.deepcopy(inputmap)
    #mask
    if settings['mask']==[]:
        mask = np.ones([inputmap['mat'].shape[0],inputmap['mat'].shape[1]])
        print('no mask')                
    else:
        mask = settings['mask']
        print('apply mask')
    
    # getting tt limits of the original input
    outputmap['ttlim_raw'] = np.zeros((2,inputmap['mat'].shape[1]))
    for idx in range(outputmap['ttlim_raw'].shape[1]):
        unmask_idx = np.where(mask[:,idx]>0.5)
        if unmask_idx[0]!=[]:
            outputmap['ttlim_raw'][0,idx] = np.max(inputmap['tt'][unmask_idx[0],idx])
            outputmap['ttlim_raw'][1,idx] = np.min(inputmap['tt'][unmask_idx[0],idx])
        else:
            outputmap['ttlim_raw'][0,idx] = np.nan
            outputmap['ttlim_raw'][1,idx] = np.nan
    
    ttlim = [np.floor(np.min(inputmap['tt'])/dtt)*dtt, np.ceil(np.max(inputmap['tt'])/dtt)*dtt]
    #print('ttlim:\n%f\n%f\n' %(ttlim[0], ttlim[1]))
    outputmap['tt'] = np.arange(ttlim[0]-2*dtt, ttlim[1]+2*dtt, dtt)
    #print(len(outputmap['tt']))
    # prepare for pixel splitting: idx, stat and the rebinned intensity matrix
    # idx :,:,0 is the index of the upper edge of the tt bin where each pixel belongs to; 
    # idx :,:,1 the fraction of the pixel into the upper edge of the tt bin, the rest goes to the lower edge
    outputmap['idx'] = np.zeros([inputmap['mat'].shape[0],inputmap['mat'].shape[1],2])
    # stat: statistics matrix for the rebinning: accumulated fraction into each rebined pixel, to be used for normalization, all values set to zero
    outputmap['stat'] = np.zeros([outputmap['tt'].shape[0],inputmap['mat'].shape[1]])
    tth_stat = np.zeros([outputmap['tt'].shape[0],inputmap['mat'].shape[1]])
    # result matrix for intensity, all values set to 0
    outputmap['mat'] = np.zeros([outputmap['tt'].shape[0],inputmap['mat'].shape[1]])
    outputmap['tth'] = np.zeros([outputmap['tt'].shape[0],inputmap['mat'].shape[1]])    
    # pixel splitting:
    #   1. use np.digitize find the upper edge index in the rebinned matrix for each original pixel
    #   2. for each original pixel, compute its fraction for the upper edge index
    #   3. split the fraction of each pixel into the upper and the lower edge in the rebinning statistics matrix; the values in the rebinning statistics matrix starts accumulation
    #   4. split the intensity of each pixel into the upper and the lower edge in the rebinned matrix; the intensity in the rebinned matrix starts accumulation
    # prepare function
    create_digitize = np.digitize
    for i in range(inputmap['tt'].shape[1]):
        outputmap['idx'][:,i,0] = create_digitize(inputmap['tt'][:,i], outputmap['tt'], right = True)
        #print(outputmap['idx'][i,:,0])
        for j in range(inputmap['tt'].shape[0]):
            idx_bins = int(outputmap['idx'][j,i,0])
            #print([j, i, idx_bins])
            outputmap['idx'][j,i,1] = 1-(outputmap['tt'][idx_bins]-inputmap['tt'][j,i])/dtt
            outputmap['stat'][idx_bins,i] = outputmap['stat'][idx_bins,i] + outputmap['idx'][j,i,1]*mask[j,i]
            outputmap['stat'][idx_bins-1,i] = outputmap['stat'][idx_bins-1,i] + (1-outputmap['idx'][j,i,1])*mask[j,i]
            outputmap['mat'][idx_bins,i] = outputmap['mat'][idx_bins,i] + inputmap['mat'][j,i]*outputmap['idx'][j,i,1]*mask[j,i]
            outputmap['mat'][idx_bins-1,i] = outputmap['mat'][idx_bins-1,i] + inputmap['mat'][j,i]*(1-outputmap['idx'][j,i,1])*mask[j,i]
            tth_stat[idx_bins,i] = tth_stat[idx_bins,i] + outputmap['idx'][j,i,1]
            tth_stat[idx_bins-1,i] = tth_stat[idx_bins-1,i] + (1-outputmap['idx'][j,i,1])          
            outputmap['tth'][idx_bins,i] = outputmap['tth'][idx_bins,i] + inputmap['tth'][j,i]*outputmap['idx'][j,i,1]
            outputmap['tth'][idx_bins-1,i] = outputmap['tth'][idx_bins-1,i] + inputmap['tth'][j,i]*(1-outputmap['idx'][j,i,1])
            #print([i, outputmap['stat'][i,213]])
    
    # prepare function
    create_argwhere = np.argwhere
    # normalization by the statistics, everything with no pixel in is counted as negative (-1e6)
    np.place(outputmap['stat'], outputmap['stat']==0, -1)    
    np.place(tth_stat, tth_stat==0, -1)  
    empty_cell = create_argwhere(outputmap['stat']<0)
    for y,x in empty_cell:
        outputmap['mat'][y,x] = 1e6
    outputmap['mat'] = outputmap['mat'] / outputmap['stat']    
    outputmap['tth'] = outputmap['tth'] / tth_stat
    #test = copy.deepcopy(outputmap['mat'])
    
    # remove the columns without the values
    empty_col = []
    empty_row_limit = outputmap['mat'].shape[0]-2 
    for idx_value in range(outputmap['mat'].shape[1]):
        empty_cell = create_argwhere(outputmap['mat'][:,idx_value]<-1e3)[:,0]
        if len(empty_cell)>empty_row_limit:
            empty_col.append(idx_value)
    
    outputmap['mat'] = np.delete(outputmap['mat'], empty_col, axis = 1)
    outputmap['tth'] = np.delete(outputmap['tth'], empty_col, axis = 1)
    outputmap['ttlim_raw'] = np.delete(outputmap['ttlim_raw'], empty_col, axis = 1)
    #outputmap['stat'] = np.delete(outputmap['stat'], empty_col, axis = 1)
    #test = np.delete(test, empty_col, axis = 1)

    #empty_col_limt = outputmap['mat'].shape[1]*3/4
    empty_col_limt = outputmap['mat'].shape[1]-2
    # remove the 1st rows without the values
    for idx_value in range(outputmap['mat'].shape[0]):
        empty_cell = create_argwhere(outputmap['mat'][idx_value,:]<-1e3)[:,0]
        if len(empty_cell)<empty_col_limt:
            break
    #print(empty_cell)
    #print(idx_value)    
    outputmap['mat'] = np.delete(outputmap['mat'], range(0, idx_value+2), axis = 0)
    outputmap['tth'] = np.delete(outputmap['tth'], range(0, idx_value+2), axis = 0)
    outputmap['tt'] = np.delete(outputmap['tt'], range(0, idx_value+2), axis = 0)
    #outputmap['stat'] = np.delete(outputmap['stat'], range(0, idx_value+2), axis = 0)
    #test = np.delete(test, range(0, idx_value+2), axis = 0)
    
    # remove the last rows without the values
    idx_last = outputmap['mat'].shape[0]-1
    for idx_value in range(idx_last+1):
        empty_cell = create_argwhere(outputmap['mat'][idx_last-idx_value,:]<-1e3)[:,0]
        if len(empty_cell)<empty_col_limt:
            break
    #print(idx_last-idx_value-1)    
    outputmap['mat'] = np.delete(outputmap['mat'], range(idx_last-idx_value-1, idx_last+1), axis = 0)
    outputmap['tth'] = np.delete(outputmap['tth'], range(idx_last-idx_value-1, idx_last+1), axis = 0)
    outputmap['tt'] = np.delete(outputmap['tt'], range(idx_last-idx_value-1, idx_last+1), axis = 0)
    #outputmap['stat'] = np.delete(outputmap['stat'], range(idx_last-idx_value-1, idx_last+1), axis = 0)
    #test = np.delete(test, range(idx_last-idx_value-1, idx_last+1), axis = 0)
    
    # prepare function
    create_zeros = np.zeros
    create_interp = np.interp  
    # interpolation for all these negative values
    for i in range(outputmap['mat'].shape[1]):
        empty_cell = create_argwhere(outputmap['mat'][:,i]<-1e3)[:,0]
        filled_cell = create_argwhere(outputmap['mat'][:,i]>=-1e3-(1e-5))[:,0]
        filled_value = create_zeros(len(filled_cell))
        filled_tth = create_zeros(len(filled_cell))
        for j in range(len(filled_cell)):
            filled_value[j] = outputmap['mat'][filled_cell[j],i]
            filled_tth[j] = outputmap['tth'][filled_cell[j],i]
        miss_values = create_interp(empty_cell, filled_cell, filled_value)
        #print(miss_values)
        miss_tth = create_interp(empty_cell, filled_cell, filled_tth)                
        for j in range(len(empty_cell)):
            outputmap['mat'][empty_cell[j],i]=miss_values[j]
            outputmap['tth'][empty_cell[j],i]=miss_tth[j]
        
        # set int value outside of tt limit raw to nan for every tth column
        if np.isnan(outputmap['ttlim_raw'][0,i]):
            outputmap['mat'][:,i] = np.nan
        else:
            np.place(outputmap['mat'][:,i],(outputmap['tt'] - outputmap['ttlim_raw'][0,i])*(outputmap['tt'] - outputmap['ttlim_raw'][1,i])>0, np.nan)
    
    # create a border mask: outside of the tt border the weight is zero
    outputmap['ttborderMask'] = np.ones([outputmap['mat'].shape[0],outputmap['mat'].shape[1]])
    nan_cell = create_argwhere(np.isnan(outputmap['mat']))
    for y,x in nan_cell:
        outputmap['ttborderMask'][y,x] = 0
    
    del empty_cell, miss_values, filled_cell, filled_value, miss_tth, filled_tth, nan_cell, i, j 
    return outputmap


def export(inputmap, filename, exp_path, axis = ['Qxy', 'Qz']):
    """
    export the map into three ascii files: _I.dat, _[axis0].dat, _[axis1].dat
    """
    np.savetxt(exp_path+filename+"_I.dat",inputmap['mat'],fmt='%f')
    np.savetxt(exp_path+filename+"_"+axis[0]+".dat",inputmap[axis[0]],fmt='%f')
    np.savetxt(exp_path+filename+"_"+axis[1]+".dat",inputmap[axis[1]],fmt='%f')
