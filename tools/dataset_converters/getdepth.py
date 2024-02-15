import cv2, sys, os
import argparse
import scipy.io
import scipy.signal
# import open3d as o3d
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import random
import torch
from mmcv.ops.ball_query import ball_query
import copy
import skimage.filters


sys.path.append('../../../spatio-temporal-csph/')
from csph_layers import CSPH3DLayer 

BASE = "/srv/home/bgoyal2/Documents/mmdetection3d/data/sunrgbd/sunrgbd_trainval/"
OUTFOLDER = BASE + '../points_min2/'
#OUTFOLDER = '/scratch/bhavya/points_baseline/3dcnndenoise-argmax/'
GEN_FOLDER = 'processed_lowfluxlowsbr_min2/SimSPADDataset_nr-576_nc-704_nt-1024_tres-586ps_dark-0_psf-0'
SUNRGBDMeta = '../OFFICIAL_SUNRGBD/SUNRGBDMeta3DBB_v2.mat'
NUM_PEAKS=3 # upto NUM_PEAKS peaks are selected
NUM_PEAKS_START = 110
CORRECTNESS_THRESH = 25
SAMPLED_POINTS=50000

scenes = open(BASE + 'all_data_idx.txt').readlines()
scenes = [x.strip() for x in scenes]

metadata = scipy.io.loadmat(BASE + SUNRGBDMeta)['SUNRGBDMeta'][0]

C = 3e8
def tof2depth(tof):
    return tof * C / 2.

def random_sampling(points, num_points, p=None):
    replace = (points.shape[0] < num_points)
    choices = np.random.choice(points.shape[0], num_points, replace=replace, p=p)
    return points[choices], choices

pulse = [[[0.0000, 0.0000, 0.0000, 0.0000, 0.0001, 0.0013, 0.0105, 0.0520, 0.1528, 0.2659, 0.2743, 0.1676, 0.0607, 0.0130, 0.0017, 0.0001, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]]

#def use_o3d(pts, write_text):
#    pcd = o3d.geometry.PointCloud()
#
#    # the method Vector3dVector() will convert numpy array of shape (n, 3) to Open3D format.
#    # see http://www.open3d.org/docs/release/python_api/open3d.utility.Vector3dVector.html#open3d.utility.Vector3dVector
#    pcd.points = o3d.utility.Vector3dVector(pts)
#
#    # http://www.open3d.org/docs/release/python_api/open3d.io.write_point_cloud.html#open3d.io.write_point_cloud
#    o3d.io.write_point_cloud("my_pts.ply", pcd, write_ascii=write_text)
#
#    # read ply file
#    pcd = o3d.io.read_point_cloud('my_pts.ply')
#
#    # visualize
#    o3d.visualization.draw_geometries([pcd])


# the arg parser
def parse_args():
    parser = argparse.ArgumentParser(description='.mat simulation file to point cloud')
    parser.add_argument(
        '--method',
        choices=['denoise', 'argmax-filtering', 'argmax-filtering-sbr', 'gaussfilter-argmax-filtering-sbr', 'argmax-filtering-conf', 'peaks-confidence', 'decompressed-peaks-confidence', 'decompressed-argmax', 'peakswogtvalid-confidence', 'gaussfilter-peaks-confidence'],
        default='peaks-confidence',
        help='Method used for converting histograms to point clouds')
    parser.add_argument(
        '--sbr',
        choices=['5_1', '5_50', '5_100', '1_50', '1_100'],
        default='1_50',
        help='SBR')
    parser.add_argument('--num_peaks', default=None, type=int,
                    help='num peaks for each pixel')
    parser.add_argument('--threshold', default=None, type=float,
                    help='threshold for spad filtering')
    parser.add_argument('--outfolder_prefix', default=None, type=str,
                    help='add prefix to output folder')
    parser.add_argument('--start', default=None, type=int,
                    help='start index for datalist')
    parser.add_argument('--end', default=None, type=int,
                    help='end index for datalist')
    args = parser.parse_args()
    return args


def camera_params(K):
    cx, cy = K[0,2], K[1,2]
    fx, fy = K[0,0], K[1,1]
    return cx, cy, fx, fy

nt_compression = 1024
csph1D_obj = None

# Does compression and decompression
def decompress(spad):
    assert spad.shape[2]==nt_compression
    spad_out = spad.transpose(2,0,1)
    spad_out = torch.from_numpy(spad_out[None,None,...])
    spad_out = csph1D_obj(spad_out)[0,0,...].numpy()
    spad_out = spad_out.transpose(1,2,0)
    return spad_out


# Find peaks and converts to point cloud
def peakpoints(nr, nc, K, bin_size, spad, gtvalid, Rtilt, rbins, intensity, peaks_post_processing=True, decompressed=False, gaussian_filter_pulse=False):
    xx = np.linspace(1, nc, nc)
    yy = np.linspace(1, nr, nr)
    x, y = np.meshgrid(xx, yy)
    cx, cy, fx, fy = camera_params(K)
    xa = (x - cx)/fx
    ya = (y - cy)/fy
    xa, ya = xa[:,:,np.newaxis], ya[:,:,np.newaxis]
    nt = spad.shape[2]
    # Removing first few bins
    spad[:,:,:5] = 0

    if(decompressed):
        # compress and decompress using truncated fourier
        spad_copy = copy.deepcopy(spad)
        spad = decompress(spad)
    elif(gaussian_filter_pulse):
        gf_pulse = np.zeros((5,5,22))
        gf_pulse[2,2,:] = pulse[0][0]
        gf_pulse = skimage.filters.gaussian(gf_pulse,sigma=1.0)
        gf_pulse = gf_pulse/gf_pulse.sum()
        spad = scipy.signal.convolve(spad, gf_pulse, mode='same')
    else:
        spad = scipy.signal.convolve(spad, pulse, mode='same')

    allpeaks = np.zeros((nr, nc, NUM_PEAKS_START))
    if(True):
        for ii in range(1, nr+1):
            for jj in range(1, nc+1):
                #peaks = scipy.signal.find_peaks(spad[ii-1, jj-1,:], distance=10, height=2)[0][:NUM_PEAKS_START]
                if(peaks_post_processing):
                    peaks = scipy.signal.find_peaks(spad[ii-1, jj-1,:], distance=10, height=0.3)[0][:NUM_PEAKS_START]
                else:
                    # Use height 0 as after convolve, spad can have very small negative numbers instead of 0
                    # it uses fourier transform for fast calculation
                    peaks = scipy.signal.find_peaks(spad[ii-1, jj-1,:], distance=10, height=0.)[0][:NUM_PEAKS_START]
                allpeaks[ii-1,jj-1,:len(peaks)]=peaks
    
        allpeaks = allpeaks.astype(int)
        density = spad[np.arange(nr)[:, np.newaxis, np.newaxis], np.arange(nc)[np.newaxis, :, np.newaxis], allpeaks]
    
        dp = np.stack([density, allpeaks])
        dpindex = dp[0,:,:,:].argsort(axis=-1)
        dpindex = dpindex[:,:,::-1]
     
        density = dp[0,:,:,:]
        density = density[np.arange(nr)[:, np.newaxis, np.newaxis], np.arange(nc)[np.newaxis, :, np.newaxis], dpindex]
        allpeaks = dp[1,:,:,:]
        allpeaks = allpeaks[np.arange(nr)[:, np.newaxis, np.newaxis], np.arange(nc)[np.newaxis, :, np.newaxis], dpindex]
    
    density = density[:,:,:NUM_PEAKS]
    allpeaks = allpeaks[:,:,:NUM_PEAKS].astype(int)
    #print('All Peaks ', np.count_nonzero(allpeaks==0))

    if(peaks_post_processing):
        maxdensity = density.max(axis=-1, keepdims=True)
        removepeaks = density<(maxdensity-0.5)
        density[removepeaks]=0.
        allpeaks[removepeaks]=0

    total_sampling_prob = density.sum(-1, keepdims=True)
    total_sampling_prob[total_sampling_prob<1e-9]=1
    sampling_prob = density/total_sampling_prob
    # we might drop a few points later, if they are farther than 65.356 depth
    # which would make sum non 1, but ignoring that for now.


    # Can remove points that are too close to camera
    # Few examples that I saw, 58 was the min bin count
    #removepeaks = allpeaks<50
    #density[removepeaks]=0.
    #allpeaks[removepeaks]=0



    #for ii in range(nr//2+10, nr//2+12):
    #    for jj in range(1, nc+1):
    #        rbin = rbins[ii-1,jj-1]-1
    #        peaks = allpeaks[ii-1, jj-1,:]
    #        plt.close()
    #        plt.figure().set_figwidth(24)

    #        plt.subplot(2,1,1)
    #        plt.bar(range(nt), spad[ii-1, jj-1,:], width=0.9)
    #        plt.scatter([rbin], [spad[ii-1, jj-1, rbin]], c='g', alpha=0.3)
    #        plt.scatter(peaks, [spad[ii-1, jj-1, peaks]], c='r', alpha=0.7)
    #        plt.text(100, 0, str(rbin) + " " + str(spad[ii-1, jj-1].max()) + " " + str(peaks) + " " + str(density[ii-1,jj-1]) )
    #        plt.subplot(2,1,2)
    #        plt.bar(range(nt), spad_copy[ii-1, jj-1,:], width=0.9)

    #        plt.savefig('plots_compress32_filteringpeaks_1_50/fig_' + str(ii-1) + '_' + str(jj-1)+ '_hist.png', dpi=500)

    #        plt.close()
    #        inten = intensity.copy()
    #        inten[ii-1, :]=1
    #        inten[:, jj-1]=1
    #        plt.imshow(inten)
    #        plt.savefig('plots_compress32_filteringpeaks_1_50/fig_' + str(ii-1) + '_' + str(jj-1)+ '_depth.png')


    # Only using this for visualization. These points are approximately correct depth
    correct = abs(np.repeat(rbins[:,:,np.newaxis], NUM_PEAKS, axis=-1) - allpeaks)<=CORRECTNESS_THRESH

    dists = tof2depth(allpeaks*bin_size)
    depths = dists/(xa**2 + ya**2 + 1)**0.5

    #print(allpeaks[73,313,:])
    #print(dists[73,313,:])
    #print(depths[73,313,:])
    #print(np.count_nonzero(depths==0))

    depths = depths*1000.
    depths = depths.astype(np.uint16)
    depths = (depths>>3)<<3
    depths = depths*gtvalid[:,:,np.newaxis]

    depths = (depths>>3 | np.uint16(depths<<13))
    depths = depths.astype('float32')/1000.
    depths[depths>8]=8
    X = xa*depths
    Y = ya*depths
    Z = depths

    #AA = set(zip(*np.nonzero(Z==0)))
    #AA2 = set(zip(*np.nonzero(X==0)))
    #AA3 = set(zip(*np.nonzero(Y==0)))
    #BB = set(zip(*np.nonzero(dists==0)))
    #gtvalidthree = np.tile(gtvalid[:,:,None],3)
    #CC = set(zip(*np.nonzero(gtvalidthree==0)))
    #DD = AA - (CC|BB)
    #DD2 = AA2 - (CC|BB)
    #DD3 = AA3 - (CC|BB)
    #print(len(BB), sorted(list(BB)))
    #print(len(AA), len(BB), len(CC))
    #print(len(DD), sorted(list(DD)))
    #print(len(DD2), sorted(list(DD2)))
    #print(len(DD3), sorted(list(DD3)))
    #print(depths[73,313,:])

    points3d = np.stack([X, Z, -Y])
    points3d, density, sampling_prob = points3d.reshape((3,-1)), density.flatten(), sampling_prob.flatten()
    points3d = np.matmul(Rtilt, points3d)
    return points3d, density, sampling_prob, correct, np.tile(xa, NUM_PEAKS).flatten(), np.tile(ya, NUM_PEAKS).flatten()

# Convert dist to depth
def finaldepth(nr, nc, K, dist, gtvalid):
    xx = np.linspace(1, nc, nc)
    yy = np.linspace(1, nr, nr)
    x, y = np.meshgrid(xx, yy)
    cx, cy, fx, fy = camera_params(K)
    x = (x - cx)/fx
    y = (y - cy)/fy
    depthmap = dist/(x**2 + y**2 + 1)**0.5
    depthmap = depthmap*1000.
    depthmap = depthmap.astype(np.uint16)
    # Not sure why SUNRGBD code for converting to point cloud (read3dPoints.m) shifts last 3 bits, but I am zeroing it out for now
    # This removes points that are farther than 65.535 because np.uint16 would have wrapped for those numbers
    depthmap = (depthmap>>3)<<3
    depthmap = depthmap*gtvalid
    return depthmap


# Convert depth to point cloud
def depth2points(nr, nc, K, depthmap, Rtilt):
    depthmap = (depthmap>>3 | np.uint16(depthmap<<13))
    depthmap = depthmap.astype('float32')/1000.
    depthmap[depthmap>8]=8

    xx = np.linspace(1, nc, nc)
    yy = np.linspace(1, nr, nr)
    x, y = np.meshgrid(xx, yy)
    cx, cy, fx, fy = camera_params(K)
    x = (x - cx)*depthmap/fx
    y = (y - cy)*depthmap/fy
    z = depthmap

    points3d = np.stack([x, z, -y])
    points3d = points3d.reshape((3,-1))
    points3d = np.matmul(Rtilt, points3d)
    return points3d

# density is just height of each bin, can normalize it in inference code
def argmaxfiltering(spad):
    spaddensity = scipy.signal.convolve(spad, pulse, mode='same')
    return spaddensity.argmax(-1), spaddensity.max(-1)


def argmaxfilteringsbr(spad, decompressed=False, gaussian_filter_pulse=False):
    spad[:,:,:20] = 0
    if(decompressed):
        # compress and decompress using truncated fourier
        spad_copy = copy.deepcopy(spad)
        spad = decompress(spad)
    elif(gaussian_filter_pulse):
        gf_pulse = np.zeros((5,5,22))
        gf_pulse[2,2,:] = pulse[0][0]
        gf_pulse = skimage.filters.gaussian(gf_pulse,sigma=1.0)
        gf_pulse = gf_pulse/gf_pulse.sum()
        spad = scipy.signal.convolve(spad, gf_pulse, mode='same')
    else:
        spad = scipy.signal.convolve(spad, pulse, mode='same')

    spadargmax, spadmax = spad.argmax(-1), spad.max(-1)
    return spadargmax, spadmax, spad.sum(-1)

# density is just height of each bin, can normalize it in inference code
def argmaxdecompressed(spad):
    spad = decompress(spad)
    return spad.argmax(-1)


# random tie breaker for argmax on raw histogram bins
def argmaxrandomtie(spad):
    maxval = spad.max(axis=-1, keepdims=True)
    maxmatrix = spad == maxval
    
    spadmax = np.zeros(spad.shape[:2], dtype=np.int32)
    for i in range(spad.shape[0]):
        for j in range(spad.shape[1]):
            spadmax[i,j] = np.random.choice(np.flatnonzero(maxmatrix[i,j,:]))
    return spadmax

def main(args):
    
    start, end = 0, len(scenes)
    if(args.start is not None):
        start = args.start
    if(args.end is not None):
        end = args.end
    if(args.num_peaks):
        global NUM_PEAKS
        NUM_PEAKS = args.num_peaks
    if(args.outfolder_prefix):
        global OUTFOLDER
        OUTFOLDER = OUTFOLDER + args.outfolder_prefix + '/'

    outfolder = OUTFOLDER + args.method + '/' + args.sbr + '/'
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    if('decompress' in args.method):
        csph1D_obj = CSPH3DLayer(k=32, num_bins=nt_compression, tblock_init='TruncFourier', optimize_codes=False, encoding_type='csph1d', zero_mean_tdim_codes=True)
        csph1D_obj.to(device='cpu')

    all_correct_cf, all_incorrect_cf = [], []
    all_correct_sp, all_incorrect_sp = [], []
    all_correct_neighcount, all_incorrect_neighcount = [], []
    all_correct_neighcf, all_incorrect_neighcf = [], []
    all_correct_neighsp, all_incorrect_neighsp = [], []
    all_correct_neighcfweighted, all_incorrect_neighcfweighted = [], []
    all_correct_neighspweighted, all_incorrect_neighspweighted = [], []
    cfmax = 0

    #scenes_selected = random.sample(scenes, 10) # for vis 
    scenes_selected = scenes[start:end]
    for scene in scenes_selected:
        print(scene)
        OUTFILE = outfolder + scene.zfill(6) +'.bin'
        if(os.path.exists(OUTFILE)):
            continue
        mat_file = BASE + GEN_FOLDER + '_' + args.sbr + '/spad_' + scene.zfill(6) + '_' + args.sbr +'.mat'
        data = scipy.io.loadmat(mat_file)
    
        nr, nc = data['intensity'].shape
        nt = data['num_bins'][0,0]
        Rtilt = metadata[int(scene)-1][1]
        K = metadata[int(scene)-1][2]
    
        depthpath = '../OFFICIAL_SUNRGBD' + metadata[int(scene)-1][3][0][16:]
        rgbpath = '../OFFICIAL_SUNRGBD' + metadata[int(scene)-1][4][0][16:]
        # Using Depth map to remove points that are NAN in original depth image, using gtvalid
        # Simulation script for histograms inpaints NAN depths
        # but I am ignoring those points as SUNRGB dataset processing ignores it too.
        gtdepth = cv2.imread(BASE + depthpath, cv2.IMREAD_UNCHANGED)
        if(gtdepth is None):
            print('could not load depth image')
            exit(0)
        gtvalid = gtdepth>0 
        rgb = cv2.imread(BASE + rgbpath, cv2.IMREAD_UNCHANGED)/255.
        rgb = rgb[:, :, ::-1]  # BGR -> RGB
        rgb = rgb.transpose(2,0,1) # HWC -> CHW
        density = None
        sampling_prob = None
    
        # Subtract 1 from range bins to get the right bin index in python
        # as matlab indexes it from 1
        # for distance calculation, this is fine
        #range_bins = data['range_bins']
        #dist = tof2depth(range_bins*data['bin_size'])
    
        spad = data['spad'].toarray()
        spad = spad.reshape((nr, nc, nt), order='F')
        #spadcopy = scipy.signal.convolve(spad, pulse, mode='same')
        #spadcopy = spad.copy()
        #spad = spad.argmax(-1)
        #spad = argmaxrandomtie(spad)
        if(args.method=='argmax-filtering-sbr'):
            spad, density, densitysum = argmaxfilteringsbr(spad)
            #thresh_mask = densitysum>=5.0
            #spad, density, densitysum = spad*thresh_mask, density*thresh_mask, densitysum*thresh_mask
            if(args.threshold is not None):
                thresh_mask = density>=args.threshold
                spad, density, densitysum = spad*thresh_mask, density*thresh_mask, densitysum*thresh_mask
            density, densitysum = density.reshape(-1), densitysum.reshape(-1)

            correct = abs(data['range_bins']-spad)<=CORRECTNESS_THRESH
            dist = tof2depth(spad*data['bin_size'])
            depthmap = finaldepth(nr, nc, K, dist, gtvalid)
            points3d = depth2points(nr, nc, K, depthmap, Rtilt)
        elif(args.method=='gaussfilter-argmax-filtering-sbr'):
            spad, density, densitysum = argmaxfilteringsbr(spad, gaussian_filter_pulse=True)
            if(args.threshold is not None):
                thresh_mask = density>=args.threshold
                spad, density, densitysum = spad*thresh_mask, density*thresh_mask, densitysum*thresh_mask
            density, densitysum = density.reshape(-1), densitysum.reshape(-1)

            correct = abs(data['range_bins']-spad)<=CORRECTNESS_THRESH
            dist = tof2depth(spad*data['bin_size'])
            depthmap = finaldepth(nr, nc, K, dist, gtvalid)
            points3d = depth2points(nr, nc, K, depthmap, Rtilt)
        elif(args.method=='argmax-filtering-conf'):
            spad, density = argmaxfiltering(spad)
            if(args.threshold is not None):
                thresh_mask = density>=args.threshold
                spad, density = spad*thresh_mask, density*thresh_mask
            density = density.reshape(-1)
            correct = abs(data['range_bins']-spad)<=CORRECTNESS_THRESH
    
            #for ii in range(nr//2+10, nr//2+15):
            #    for jj in range(1, nc+1):
            #        plt.close()
            #        plt.figure().set_figwidth(24)
            #        plt.bar(range(nt), spadcopy[ii-1, jj-1,:], width=0.9)
            #        rbin = data['range_bins'][ii-1,jj-1]-1
            #        selected = spad[ii-1, jj-1]
            #        plt.scatter([rbin], [spadcopy[ii-1, jj-1, rbin]], c='g', alpha=0.3)
            #        plt.scatter([selected], [spadcopy[ii-1, jj-1, selected]], c='r', alpha=0.7)
            #        plt.text(100, 0.4, str(rbin) + " " + str(spadcopy[ii-1, jj-1, :].max()) + " " + str(selected) + str(spadcopy[ii-1, jj-1, selected]) + " " + str(gtvalid[ii-1,jj-1]) + " " + str(data['intensity'][ii-1,jj-1]))
            #        plt.savefig('plots_argmax_filtering_' + args.sbr + '/fig' + str(ii-1) + '_' + str(jj-1)+ '_hist.png', dpi=500)
            #        plt.close()
            #        inten = data['intensity'].copy()
            #        inten[ii-1, :]=1
            #        inten[:, jj-1]=1
            #        plt.imshow(inten)
            #        plt.savefig('plots_argmax_filtering_' + args.sbr + '/fig' + str(ii-1) + '_' + str(jj-1)+ '_depth.png')
        
            dist = tof2depth(spad*data['bin_size'])
        
            depthmap = finaldepth(nr, nc, K, dist, gtvalid)
            points3d = depth2points(nr, nc, K, depthmap, Rtilt)
        elif(args.method=='decompressed-argmax'):
            spad = argmaxdecompressed(spad)
            correct = abs(data['range_bins']-spad)<=CORRECTNESS_THRESH

            dist = tof2depth(spad*data['bin_size'])
        
            depthmap = finaldepth(nr, nc, K, dist, gtvalid)
            points3d = depth2points(nr, nc, K, depthmap, Rtilt)
        elif(args.method=='denoise'):
            denoised = np.load(mat_file+'.npz')
            denoised = denoised['spad_denoised_argmax']
            correct = abs(data['range_bins']-denoised)<=CORRECTNESS_THRESH

            dist = tof2depth(denoised*data['bin_size'])
            depthmap = finaldepth(nr, nc, K, dist, gtvalid)
            points3d = depth2points(nr, nc, K, depthmap, Rtilt)

        elif(args.method == 'peaks-confidence'):
            points3d, density, sampling_prob, correct, xa, ya = peakpoints(nr, nc, K, data['bin_size'], spad, gtvalid, Rtilt, data['range_bins'], data['intensity'])
            rgb = np.repeat(rgb[:,:,:,np.newaxis], NUM_PEAKS, axis=-1)    
        elif(args.method == 'gaussfilter-peaks-confidence'):
            points3d, density, sampling_prob, correct, xa, ya = peakpoints(nr, nc, K, data['bin_size'], spad, gtvalid, Rtilt, data['range_bins'], data['intensity'], gaussian_filter_pulse=True, peaks_post_processing=False)
            rgb = np.repeat(rgb[:,:,:,np.newaxis], NUM_PEAKS, axis=-1)    
        elif(args.method == 'peakswogtvalid-confidence'):
            points3d, density, sampling_prob, correct, xa, ya = peakpoints(nr, nc, K, data['bin_size'], spad, np.ones_like(gtvalid), Rtilt, data['range_bins'], data['intensity'], peaks_post_processing = False)
            rgb = np.repeat(rgb[:,:,:,np.newaxis], NUM_PEAKS, axis=-1)    
        elif(args.method == 'decompressed-peaks-confidence'):
            points3d, density, sampling_prob, correct, xa, ya = peakpoints(nr, nc, K, data['bin_size'], spad, gtvalid, Rtilt, data['range_bins'], data['intensity'], decompressed = True, peaks_post_processing = False)
            rgb = np.repeat(rgb[:,:,:,np.newaxis], NUM_PEAKS, axis=-1) 
        else:
            print('not implemented yet')
            exit(0)

        correct = correct.reshape(-1)
        rgb = rgb.reshape((3, -1))

        valid = np.all(points3d, axis=0) # only select points that have non zero locations    
        if(density is not None):
            density = density[valid]
            densitysum = densitysum[valid]
        correct = correct[valid]
        points3d, rgb = points3d.T, rgb.T
        points3d, rgb = points3d[valid,:], rgb[valid,:]
        
        print(points3d.shape[0], gtvalid.sum())
        num_points = SAMPLED_POINTS
        if(sampling_prob is not None):
            sampling_prob = sampling_prob[valid]
            xa,ya = xa[valid], ya[valid]

            # xa, ya is pixel cordinates
            # Sampling num_points pixels and then selecting all peak from those pixels.
            # so, this allows sampling number of pixels rather than points.
            xya = list( zip(xa,ya) )
            all_xya = list( set( xya ) )
            if(len(all_xya)<=num_points):
                selected_xy = set(all_xya)
            else:
                selected_xy = set(random.sample(all_xya, num_points))
            choices = []
            for xy_idx, xy in enumerate(xya):
                if(xy in selected_xy):
                    choices.append(xy_idx)
            assert len(choices)>=num_points
            points3d = points3d[choices]

            # Earlier I was sampling by sampling_prob, but this does not ensure all peaks from a pixel are included
            #points3d, choices = random_sampling(points3d, num_points, p=sampling_prob/sampling_prob.sum())
            #negprobs = -1*sampling_prob[choices]
            #newchoices = negprobs.argsort()
            #choices = choices[newchoices]
            #points3d = points3d[newchoices]
            sampling_prob = sampling_prob[choices]
        else:
            points3d, choices = random_sampling(points3d, num_points)

        if(density is not None):
            density = density[choices]
            densitysum = densitysum[choices]
        rgb = rgb[choices]
        correct = correct[choices]

        if(sampling_prob is not None):
            points3d_rgb = np.concatenate([points3d, density[:,np.newaxis], sampling_prob[:,np.newaxis], rgb], axis=1)
        elif(density is not None):
            points3d_rgb = np.concatenate([points3d, density[:,np.newaxis], density[:,np.newaxis]/densitysum[:,np.newaxis], rgb], axis=1)
        else:
            points3d_rgb = np.concatenate([points3d, rgb], axis=1)
        #points_xyz = torch.from_numpy(points3d_rgb[:,:3]).cuda()[None, :, :]
        #points_probs = torch.from_numpy(points3d_rgb[:,3]).cuda()[None, :]
        #points_sp = torch.from_numpy(points3d_rgb[:,4]).cuda()[None, :]
        ##points_probs = torch.ones(points3d_rgb.shape[0]).cuda()[None, :]
        ##points_sp = torch.ones(points3d_rgb.shape[0]).cuda()[None, :]

        #cfmax = max(cfmax, points_probs.max())
        ##points_sp = points_sp/points_sp.max()
    
        #all_correct_cf.extend(points_probs[0, correct].tolist())
        #all_incorrect_cf.extend(points_probs[0, ~correct].tolist())
        #all_correct_sp.extend(points_sp[0, correct].tolist())
        #all_incorrect_sp.extend(points_sp[0, ~correct].tolist())

        #MAX_BALL_NEIGHBORS = 64
        ## Ball query returns same index is neighbors are less than queried number of neighbors
        ## output looks like [3,56,74,2,44,3,3,3,3,3,3,3,3,3,3,3,3]
        #ball_idxs = ball_query(0, 0.2, MAX_BALL_NEIGHBORS, points_xyz, points_xyz).long()
        #
        ## first idx of the ball query is repeated if neighbors are fewer than MAX
        #ball_idxs_first = ball_idxs[:,:,0][:,:,None]
        #nonzero_ball_idxs = ((ball_idxs-ball_idxs_first)!=0)
        #nonzero_count = nonzero_ball_idxs.sum(-1).cpu().numpy()
    
        #all_correct_neighcount.extend(nonzero_count[0,correct].tolist())
        #all_incorrect_neighcount.extend(nonzero_count[0,~correct].tolist())
    
        #points_probs_tiled = points_probs[:,:,None].tile(MAX_BALL_NEIGHBORS)
        #points_sp_tiled = points_sp[:,:,None].tile(MAX_BALL_NEIGHBORS)
        #neighbor_probs = torch.gather(points_probs_tiled, 1, ball_idxs) 
        #neighbor_sp = torch.gather(points_sp_tiled, 1, ball_idxs) 
        #neighbor_probs = neighbor_probs*nonzero_ball_idxs
        #neighbor_sp = neighbor_sp*nonzero_ball_idxs
        ## average neighbor probability, would be less if neighbors are fewer than MAX
        #neighbor_probs = neighbor_probs.mean(-1)
        #neighbor_sp = neighbor_sp.mean(-1)
        #neighbor_probs_weighted = neighbor_probs*points_probs
        #neighbor_sp_weighted = neighbor_sp*points_sp
 
        #all_correct_neighcf.extend(neighbor_probs[0,correct].tolist())
        #all_incorrect_neighcf.extend(neighbor_probs[0,~correct].tolist())

        #all_correct_neighsp.extend(neighbor_sp[0,correct].tolist())
        #all_incorrect_neighsp.extend(neighbor_sp[0,~correct].tolist())
    
        #all_correct_neighcfweighted.extend(neighbor_probs_weighted[0,correct].tolist())
        #all_incorrect_neighcfweighted.extend(neighbor_probs_weighted[0,~correct].tolist())
    
        #all_correct_neighspweighted.extend(neighbor_sp_weighted[0,correct].tolist())
        #all_incorrect_neighspweighted.extend(neighbor_sp_weighted[0,~correct].tolist())
    
    
        # .bin file should be float 32 for mmdet3d
        points3d_rgb.astype(np.float32).tofile(OUTFILE)
        #cv2.imwrite(outfolder + scene.zfill(6) + '.png', depthmap)
    
    #UPPER = int((cfmax+0.5)*100)
    #bins = [x*0.01 for x in range(UPPER)]
    ##UPPER = int((cfmax+0.5))
    ##bins = [x for x in range(UPPER)]
    #IMAGE_DIR = 'figs_sbr_gauss'

    #plt.close()
    #plt.hist(all_correct_cf, bins, color='g', alpha=0.5)
    #plt.hist(all_incorrect_cf, bins, color='r', alpha=0.5)
    #plt.savefig(IMAGE_DIR + '/pointcf' + str(MAX_BALL_NEIGHBORS) + '_peaks_' + args.sbr + '.png', dpi=500)

    #plt.close()
    #plt.hist(all_correct_neighcf, bins, color='g', alpha=0.5)
    #plt.hist(all_incorrect_neighcf, bins, color='r', alpha=0.5)
    #plt.savefig(IMAGE_DIR + '/neighcf' + str(MAX_BALL_NEIGHBORS) + '_peaks_' + args.sbr + '.png', dpi=500)
    #
    #plt.close()
    #plt.hist(all_correct_neighcfweighted, bins, color='g', alpha=0.5)
    #plt.hist(all_incorrect_neighcfweighted, bins, color='r', alpha=0.5)
    #plt.savefig(IMAGE_DIR + '/neighcfweighted' + str(MAX_BALL_NEIGHBORS) + '_peaks_' + args.sbr + '.png', dpi=500)

    #plt.close()
    #bins = range(MAX_BALL_NEIGHBORS+2)
    #plt.hist(all_correct_neighcount, bins, color='g', alpha=0.5)
    #plt.hist(all_incorrect_neighcount, bins, color='r', alpha=0.5)
    #plt.savefig(IMAGE_DIR + '/neighcount' + str(MAX_BALL_NEIGHBORS) + '_peaks_' + args.sbr + '.png', dpi=500)

    #plt.close()
    #bins = [x*0.001 for x in range(101)]
    #plt.hist(all_correct_sp, bins, color='g', alpha=0.5)
    #plt.hist(all_incorrect_sp, bins, color='r', alpha=0.5)
    #plt.savefig(IMAGE_DIR + '/pointsp' + str(MAX_BALL_NEIGHBORS) + '_peaks_' + args.sbr + '.png', dpi=500)

    #plt.close()
    #bins = [x*0.001 for x in range(101)]
    #plt.hist(all_correct_neighsp, bins, color='g', alpha=0.5)
    #plt.hist(all_incorrect_neighsp, bins, color='r', alpha=0.5)
    #plt.savefig(IMAGE_DIR + '/neighsp' + str(MAX_BALL_NEIGHBORS) + '_peaks_' + args.sbr + '.png', dpi=500)

    #plt.close()
    #bins = [x*0.001 for x in range(101)]
    #plt.hist(all_correct_neighspweighted, bins, color='g', alpha=0.5)
    #plt.hist(all_incorrect_neighspweighted, bins, color='r', alpha=0.5)
    #plt.savefig(IMAGE_DIR + '/neighspweighted' + str(MAX_BALL_NEIGHBORS) + '_peaks_' + args.sbr + '.png', dpi=500)



if __name__ == '__main__':
  args = parse_args()
  print(args)
  main(args)
