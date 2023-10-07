import cv2, sys
import scipy.io
# import open3d as o3d
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
BASE = "/srv/home/bgoyal2/Documents/mmdetection3d/data/sunrgbd/sunrgbd_trainval/"
GEN_FOLDER = 'processed_full_lowflux/SimSPADDataset_nr-576_nc-704_nt-1024_tres-586ps_dark-0_psf-0/'
SUNRGBDMeta = '../OFFICIAL_SUNRGBD/SUNRGBDMeta3DBB_v2.mat'
SBR = '5_5'
OUTFOLDER = BASE + '../py/points_' + SBR + '_gtdepth/'

scenes = open(BASE + 'train_data_idx.txt').readlines()
scenes = [x.strip() for x in scenes]

metadata = scipy.io.loadmat(BASE + SUNRGBDMeta)['SUNRGBDMeta'][0]

start, end = 0, len(scenes)
if(len(sys.argv)>1):
    start = int(sys.argv[1])
    end = int(sys.argv[2])

C = 3e8
def tof2depth(tof):
    return tof * C / 2.



def random_sampling(points, num_points, replace=None, return_choices=False):
    if replace is None:
        replace = (points.shape[0] < num_points)
    choices = np.random.choice(points.shape[0], num_points, replace=replace)
    if return_choices:
        return points[choices], choices
    else:
        return points[choices]

def use_o3d(pts, write_text):
    pcd = o3d.geometry.PointCloud()

    # the method Vector3dVector() will convert numpy array of shape (n, 3) to Open3D format.
    # see http://www.open3d.org/docs/release/python_api/open3d.utility.Vector3dVector.html#open3d.utility.Vector3dVector
    pcd.points = o3d.utility.Vector3dVector(pts)

    # http://www.open3d.org/docs/release/python_api/open3d.io.write_point_cloud.html#open3d.io.write_point_cloud
    o3d.io.write_point_cloud("my_pts.ply", pcd, write_ascii=write_text)

    # read ply file
    pcd = o3d.io.read_point_cloud('my_pts.ply')

    # visualize
    o3d.visualization.draw_geometries([pcd])


def camera_params(K):
    cx, cy = K[0,2], K[1,2]
    fx, fy = K[0,0], K[1,1]
    return cx, cy, fx, fy

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
    points3d = points3d.reshape((3,-1), order='F')
    points3d = np.matmul(Rtilt, points3d)
    return points3d

for scene in scenes[start:end]:
    print(scene)
    data = scipy.io.loadmat(BASE + GEN_FOLDER + 'spad_' + scene.zfill(6) + '_' + SBR +'.mat')

    nr, nc = data['intensity'].shape
    nt = data['num_bins'][0,0]
    Rtilt = metadata[int(scene)-1][1]
    K = metadata[int(scene)-1][2]

    depthpath = '../OFFICIAL_SUNRGBD' + metadata[int(scene)-1][3][0][16:]
    rgbpath = '../OFFICIAL_SUNRGBD' + metadata[int(scene)-1][4][0][16:]
    # Uisng Depth map to remove points that are NAN in original depth image
    # Simulation script for histograms inpaints such NAN depths
    # but I am ignoring those points
    gtdepth = cv2.imread(BASE + depthpath, cv2.IMREAD_UNCHANGED)
    gtvalid = gtdepth>0 
    rgb = cv2.imread(BASE + rgbpath, cv2.IMREAD_UNCHANGED)/255.
    rgb = rgb[:, :, ::-1]  # BGR -> RGB
    rgb = rgb.reshape((-1,3), order='F')


    range_bins = data['range_bins']
    dist = tof2depth(range_bins*data['bin_size'])

    #spad = data['spad'].toarray()
    #spad = spad.reshape((nr, nc, nt), order='F')
    #spad = spad.argmax(-1)
    #dist = tof2depth(spad*data['bin_size'])

    depthmap = finaldepth(nr, nc, K, dist, gtvalid)
    points3d = depth2points(nr, nc, K, depthmap, Rtilt)
    valid = np.all(points3d, axis=0)
    points3d = points3d.T
    points3d = points3d[valid,:]
    rgb = rgb[valid,:]
    points3d_rgb = np.concatenate([points3d, rgb], axis=1)
    # .bin file should be float 32 for mmdet3d
    pc_upright_depth_subsampled = random_sampling(points3d_rgb, 50000).astype(np.float32)
    pc_upright_depth_subsampled.tofile(OUTFOLDER + scene.zfill(6) +'.bin')
    #scipy.io.savemat(BASE + GEN_FOLDER + 'spad_' + scene.zfill(6) + '_' + SBR + '_gtdepthpy.mat', {"instance": points3d_rgb})
    #cv2.imwrite(BASE + GEN_FOLDER + 'spad_' + scene.zfill(6) + '_' + SBR + '_gtdepth.png', depthmap)
    # cv2.imwrite(BASE + GEN_FOLDER + 'spad_' + scene.zfill(6) + '_' + SBR + '_argmax.png', depthmap)




    # pts = []
#    for ii in range(nr//2, nr+1):
#        for jj in range(1, nc+1):
#            peaks = scipy.signal.find_peaks(spad[ii-1, jj-1,:], distance=10, height=2)[0]
#            print(peaks)
#            exit()
#            plt.clf()
#            plt.hist(list(range(nt)), nt, weights=spad[ii-1, jj-1,:])
#            plt.scatter(peaks, np.zeros_like(peaks), c='r')
#            plt.text(100,1, str(len(peaks)))
#            plt.savefig('plots_' + SBR + '/fig' + str(ii-1) + '_' + str(jj-1)+ '.png')
#            # x, y = jj-cx, ii-cy
#            # dd = dist[ii-1,jj-1]
#            # depth = dd / ( ( (x/fx)**2 + (y/fy)**2 + 1 )**0.5 )
#            # X = x*depth/fx
#            # Y = y*depth/fy
#            # depthmap[ii-1,jj-1] = depth
#            # pts.append((X, depth, -Y))
#
#    # cv2.imwrite('intensity.jpg', np.repeat(data['intensity'][:, :, np.newaxis], 3, axis=2)*255 )
#
#    # pts = np.array(pts)
#    # pts = pts[np.random.choice(pts.shape[0], 100000, replace=False), :]
#    # # print(pts[:10])
#    # write_text = True
#    # use_o3d(pts, write_text)



