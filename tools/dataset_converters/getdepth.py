import cv2, sys
import scipy.io
import open3d as o3d
import tof_utils
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

BASE = "/nobackup/bhavya/votenet/sunrgbd/sunrgbd_trainval/"
GEN_FOLDER = 'processed_full/SimSPADDataset_nr-576_nc-704_nt-1024_tres-586ps_dark-0_psf-0/'
SUNRGBDMeta = '../OFFICIAL_SUNRGBD/SUNRGBDMeta3DBB_v2.mat'
SBR = '50_1'

scenes = open(BASE + 'val_data_idx.txt').readlines()
scenes = [x.strip() for x in scenes]

metadata = scipy.io.loadmat(BASE + SUNRGBDMeta)['SUNRGBDMeta'][0]

start, end = 0, len(scenes)
if(len(sys.argv)>1):
    start = int(sys.argv[1])
    end = int(sys.argv[2])

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


def finaldepth(nr, nc, cx, cy, fx, fy, dist, gtvalid):
    xx = np.linspace(1, nc, nc)
    yy = np.linspace(1, nr, nr)
    x, y = np.meshgrid(xx, yy)
    x = (x - cx)/fx
    y = (y - cy)/fy
    depthmap = dist/(x**2 + y**2 + 1)**0.5
    depthmap = depthmap*1000.
    depthmap = depthmap.astype(np.uint16)
    # Not sure why SUNRGBD code for converting to point cloud (read3dPoints.m) shifts last 3 bits, but I am zeroing it out for now
    depthmap = (depthmap>>3)<<3
    depthmap = depthmap*gtvalid
    return depthmap


for scene in scenes[start:end]:
    print(scene)
    data = scipy.io.loadmat(BASE + GEN_FOLDER + 'spad_' + scene.zfill(6) + '_' + SBR +'.mat')

    nr, nc = data['intensity'].shape
    nt = data['num_bins'][0,0]
    K = data['K']
    cx, cy = K[1, 6], K[1, 7]
    fx, fy = K[1, 0], K[1, 4]
    # print(K, cx, cy, fx, fy)

    # gt = cv2.imread('/nobackup/bhavya/datasets/sunrgbd/SUNRGBD/realsense/sa/2014_10_21-18_19_07-1311000073/depth/0000063.png', cv2.IMREAD_UNCHANGED)
    gtpath = '../OFFICIAL_SUNRGBD' + metadata[int(scene)-1][3][0][16:]
    gt = cv2.imread(BASE + gtpath, cv2.IMREAD_UNCHANGED)
    gtvalid = gt>0


    # range_bins = data['range_bins']
    # dist = tof_utils.tof2depth(range_bins*data['bin_size'])
    # depthmap = finaldepth(nr, nc, cx, cy, fx, fy, dist, gtvalid)
    # cv2.imwrite(BASE + GEN_FOLDER + 'spad_' + scene.zfill(6) + '_' + SBR + '_gtdepth.png', depthmap)


    spad = data['spad'].toarray()
    spad = spad.reshape((nr, nc, nt), order='F')
    spad = spad.argmax(-1)
    dist = tof_utils.tof2depth(spad*data['bin_size'])
    depthmap = finaldepth(nr, nc, cx, cy, fx, fy, dist, gtvalid)
    cv2.imwrite(BASE + GEN_FOLDER + 'spad_' + scene.zfill(6) + '_' + SBR + '_argmax.png', depthmap)




    # # pts = []
    # for ii in range(nr//2, nr+1):
    #     for jj in range(1, nc+1):
    #         peaks = scipy.signal.find_peaks(spad[ii-1, jj-1,:])[0]
    #         plt.clf()
    #         plt.hist(list(range(nt)), nt, weights=spad[ii-1, jj-1,:])
    #         plt.scatter(peaks, np.zeros_like(peaks), c='r')
    #         plt.savefig('plots_' + SBR + '/fig' + str(ii-1) + '_' + str(jj-1)+ '.png')
    #         # x, y = jj-cx, ii-cy
    #         # dd = dist[ii-1,jj-1]
    #         # depth = dd / ( ( (x/fx)**2 + (y/fy)**2 + 1 )**0.5 )
    #         # X = x*depth/fx
    #         # Y = y*depth/fy
    #         # depthmap[ii-1,jj-1] = depth
    #         # pts.append((X, depth, -Y))

    # cv2.imwrite('intensity.jpg', np.repeat(data['intensity'][:, :, np.newaxis], 3, axis=2)*255 )

    # pts = np.array(pts)
    # pts = pts[np.random.choice(pts.shape[0], 100000, replace=False), :]
    # # print(pts[:10])
    # write_text = True
    # use_o3d(pts, write_text)



