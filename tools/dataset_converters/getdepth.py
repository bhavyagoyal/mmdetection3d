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


BASE = "/srv/home/bgoyal2/Documents/mmdetection3d/data/sunrgbd/sunrgbd_trainval/"
#OUTFOLDER = BASE + '../points_wp_cf/'
OUTFOLDER = '/scratch/bhavya/points_wsp_cf/'
GEN_FOLDER = 'processed_full_lowfluxlowsbr/SimSPADDataset_nr-576_nc-704_nt-1024_tres-586ps_dark-0_psf-0'
SUNRGBDMeta = '../OFFICIAL_SUNRGBD/SUNRGBDMeta3DBB_v2.mat'
NUM_PEAKS=3 # upto NUM_PEAKS peaks are selected
NUM_PEAKS_START = 110

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
        choices=['argmax-filtering', 'argmax-filtering-conf', 'peaks-confidence'],
        default='peaks-confidence',
        help='Method used for converting histograms to point clouds')
    parser.add_argument(
        '--sbr',
        choices=['5_50', '5_100', '1_50', '1_100', '5_500'],
        default='5_100',
        help='SBR')
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

# Find peaks and converts to point cloud
def peakpoints(nr, nc, K, bin_size, spad, gtvalid, Rtilt, rbins, intensity):
    xx = np.linspace(1, nc, nc)
    yy = np.linspace(1, nr, nr)
    x, y = np.meshgrid(xx, yy)
    cx, cy, fx, fy = camera_params(K)
    xa = (x - cx)/fx
    ya = (y - cy)/fy
    xa, ya = xa[:,:,np.newaxis], ya[:,:,np.newaxis]
    spad = scipy.signal.convolve(spad, pulse, mode='same')

    allpeaks = np.zeros((nr, nc, NUM_PEAKS_START))
    for ii in range(1, nr+1):
        for jj in range(1, nc+1):
            #peaks = scipy.signal.find_peaks(spad[ii-1, jj-1,:], distance=10, height=2)[0][:NUM_PEAKS_START]
            peaks = scipy.signal.find_peaks(spad[ii-1, jj-1,:], distance=10, height=0.3)[0][:NUM_PEAKS_START]
            allpeaks[ii-1,jj-1,:len(peaks)]=peaks

    spad[:,:,0] = 0
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

    maxdensity = density.max(axis=-1, keepdims=True)
    removepeaks = density<(maxdensity-0.5)
    density[removepeaks]=0.
    allpeaks[removepeaks]=0

    total_sampling_prob = density.sum(-1, keepdims=True)
    total_sampling_prob[total_sampling_prob<1e-9]=1
    sampling_prob = density/total_sampling_prob

    # Can remove points that are too close to camera
    # Few examples that I saw, 58 was the min bin count
    #removepeaks = allpeaks<50
    #density[removepeaks]=0.
    #allpeaks[removepeaks]=0

    # Convert it to probability
#    totaldensity = density.sum(-1, keepdims=true)
#    totaldensity[totaldensity<1e-9]=1
#    density = density/totaldensity


    #for ii in range(nr//2+10, nr//2+12):
    #    for jj in range(1, nc+1):
    #        plt.close()
    #        plt.figure().set_figwidth(24)
    #        plt.bar(range(nt), spad[ii-1, jj-1,:], width=0.9)
    #        rbin = rbins[ii-1,jj-1]-1
    #        peaks = allpeaks[ii-1, jj-1,:]
    #        plt.scatter([rbin], [spad[ii-1, jj-1, rbin]], c='g', alpha=0.3)
    #        plt.scatter(peaks, [spad[ii-1, jj-1, peaks]], c='r', alpha=0.7)
    #        plt.text(100, 0, str(rbin) + " " + str(spad[ii-1, jj-1].max()) + " " + str(peaks) + " " + str(density[ii-1,jj-1]) )
    #        plt.savefig('plots_filteringpeaks_' + sbr + '/fig' + str(ii-1) + '_' + str(jj-1)+ '_hist.png', dpi=500)
    #        plt.close()
    #        inten = intensity.copy()
    #        inten[ii-1, :]=1
    #        inten[:, jj-1]=1
    #        plt.imshow(inten)
    #        plt.savefig('plots_filteringpeaks_' + sbr + '/fig' + str(ii-1) + '_' + str(jj-1)+ '_depth.png')


    # Only using this for visualization. These points are approximately correct depth
    correct = abs(np.repeat(rbins[:,:,np.newaxis], NUM_PEAKS, axis=-1) - allpeaks)<=10 

    dists = tof2depth(allpeaks*bin_size)
    depths = dists/(xa**2 + ya**2 + 1)**0.5

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
    points3d = np.stack([X, Z, -Y])
    points3d, density, sampling_prob = points3d.reshape((3,-1)), density.flatten(), sampling_prob.flatten()
    points3d = np.matmul(Rtilt, points3d)
    return points3d, density, sampling_prob, correct

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

    outfolder = OUTFOLDER + args.method + '_' + args.sbr + '/'
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    all_correct, all_incorrect = [], []
    all_correct_neigh, all_incorrect_neigh = [], []
    all_correct_neighprobs, all_incorrect_neighprobs = [], []
    all_correct_neighprobsweighted, all_incorrect_neighprobsweighted = [], []
    probmax = 0
    for scene in scenes[start:end]:
        print(scene)
        OUTFILE = outfolder + scene.zfill(6) +'.bin'
        if(os.path.exists(OUTFILE)):
            continue
        data = scipy.io.loadmat(BASE + GEN_FOLDER + '_' + args.sbr + '/spad_' + scene.zfill(6) + '_' + args.sbr +'.mat')
    
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
        if(args.method=='argmax-filtering-conf'):
            spad, density = argmaxfiltering(spad)
            density = density.reshape(-1)
            correct = abs(data['range_bins']-spad)<=10
    
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
        elif(args.method == 'peaks-confidence'):
            points3d, density, sampling_prob, correct = peakpoints(nr, nc, K, data['bin_size'], spad, gtvalid, Rtilt, data['range_bins'], data['intensity'])
            rgb = np.repeat(rgb[:,:,:,np.newaxis], NUM_PEAKS, axis=-1)    
        else:
            print('not implemented yet')
            exit(0)

        correct = correct.reshape(-1)
        rgb = rgb.reshape((3, -1))

        valid = np.all(points3d, axis=0) # only select points that have non zero locations    
        density = density[valid]
        sampling_prob = sampling_prob[valid]
        correct = correct[valid]
        points3d, rgb = points3d.T, rgb.T
        points3d, rgb = points3d[valid,:], rgb[valid,:]
    
        points3d, choices = random_sampling(points3d, 50000, p=sampling_prob/sampling_prob.sum())
        density = density[choices]
        sampling_prob = sampling_prob[choices]
        rgb = rgb[choices]
        correct = correct[choices]

        all_correct.extend(density[correct])
        all_incorrect.extend(density[~correct])
   
        #density = density/density.sum()
    
        #points3d_rgb = np.concatenate([points3d, rgb], axis=1)
        points3d_rgb = np.concatenate([points3d, density[:,np.newaxis], sampling_prob[:,np.newaxis], rgb], axis=1)
 
        ##points3d_rgb = random_sampling(points3d_rgb, 50000, p=density)
        #points3d_rgb, choices = random_sampling(points3d_rgb, 50000)
        #
        #points_xyz = torch.from_numpy(points3d_rgb[:,:3]).cuda()[None, :, :]
        #points_probs = torch.from_numpy(points3d_rgb[:,3]).cuda()[None, :]
        #probmax = max(probmax, points_probs.max())
        ##points_probs = points_probs/points_probs.max()
    
        #MAX_BALL_NEIGHBORS = 64
        ## Ball query returns same index is neighbors are less than queried number of neighbors
        ## output looks like [3,56,74,2,44,3,3,3,3,3,3,3,3,3,3,3,3]
        #ball_idxs = ball_query(0, 0.2, MAX_BALL_NEIGHBORS, points_xyz, points_xyz).long()
        ##ball_idxs = ball_idxs[:,:,1:]
        #
        ## subtract original idx number and find non zero to find neighbors 
        #idxs = torch.arange(0,points_xyz.shape[1])[None, :, None].cuda()
        #nonzero_ball_idxs = ((ball_idxs-idxs)!=0)
        #nonzero_count = nonzero_ball_idxs.sum(-1).cpu().numpy()
    
        #all_correct_neigh.extend(nonzero_count[0,correct[choices]].tolist())
        #all_incorrect_neigh.extend(nonzero_count[0,~correct[choices]].tolist())
    
        #points_probs_tiled = points_probs[:,:,None].tile(MAX_BALL_NEIGHBORS)
        #neighbor_probs = torch.gather(points_probs_tiled, 1, ball_idxs) 
        #neighbor_probs = neighbor_probs*nonzero_ball_idxs
        ## average neighbor probability, would be less if neighbors are less
        #neighbor_probs = neighbor_probs.mean(-1)
        #neighbor_probs_weighted = neighbor_probs*points_probs
    
        #all_correct_neighprobs.extend(neighbor_probs[0,correct[choices]].tolist())
        #all_incorrect_neighprobs.extend(neighbor_probs[0,~correct[choices]].tolist())
    
    
        #all_correct_neighprobsweighted.extend(neighbor_probs_weighted[0,correct[choices]].tolist())
        #all_incorrect_neighprobsweighted.extend(neighbor_probs_weighted[0,~correct[choices]].tolist())
    
        ##        choices = (-1*neighbor_probs).argsort()
        ##        #choices = choices[:,:self.backbone.SA_modules[0].fps_sample_range_list[0]] 
        ##        stack_points = torch.gather(stack_points, 1, choices[:,:,None].tile(stack_points.shape[2]))
        ##        #stack_points = stack_points[:,:-10000]
    
    
    
        # .bin file should be float 32 for mmdet3d
        points3d_rgb.astype(np.float32).tofile(OUTFILE)
        #cv2.imwrite(outfolder + scene.zfill(6) + '.png', depthmap)
    
    #UPPER = int((probmax+0.5)*100)
    #plt.close()
    #bins = [x*0.01 for x in range(UPPER)]
    #plt.hist(all_correct, bins, color='g', alpha=0.5)
    #plt.hist(all_incorrect, bins, color='r', alpha=0.5)
    #plt.savefig('figs/pointprobs64_peaks_' + args.sbr + '.png', dpi=500)
    #
    #plt.close()
    #bins = [x*0.01 for x in range(UPPER)]
    #plt.hist(all_correct_neighprobs, bins, color='g', alpha=0.5)
    #plt.hist(all_incorrect_neighprobs, bins, color='r', alpha=0.5)
    #plt.savefig('figs/neighprobs64_peaks_' + args.sbr + '.png', dpi=500)
    #
    #plt.close()
    #bins = range(1025)
    #plt.hist(all_correct_neigh, bins, color='g', alpha=0.5)
    #plt.hist(all_incorrect_neigh, bins, color='r', alpha=0.5)
    #plt.savefig('figs/neighcount64_peaks_' + args.sbr + '.png', dpi=500)
    #
    #
    #plt.close()
    #bins = [x*0.01 for x in range(UPPER)]
    #plt.hist(all_correct_neighprobsweighted, bins, color='g', alpha=0.5)
    #plt.hist(all_incorrect_neighprobsweighted, bins, color='r', alpha=0.5)
    #plt.savefig('figs/neighweightedprobs64_peaks_' + args.sbr + '.png', dpi=500)


if __name__ == '__main__':
  args = parse_args()
  print(args)
  main(args)
