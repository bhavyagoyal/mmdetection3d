import numpy as np

from mmdet3d.visualization import Det3DLocalVisualizer

#points = np.fromfile('demo/data/sunrgbd/000017.bin', dtype=np.float32)
points = np.fromfile('data/sunrgbd/points_5_5_gtdepth/005050.bin', dtype=np.float32)
#points = np.fromfile('data/sunrgbd/points_5_5_argmax/005050.bin', dtype=np.float32)
#points = np.fromfile('data/sunrgbd/points_bk/005050.bin', dtype=np.float32)
#points = np.fromfile('/nobackup/bhavya/votenet/sunrgbd/matlab/testing_depth.bin', dtype=np.float32)
print(points.shape)
points = points.reshape(-1, 6)[:,:3]
visualizer = Det3DLocalVisualizer()
mask = np.random.rand(points.shape[0], 3)
points_with_mask = np.concatenate((points, mask), axis=-1)
# Draw 3D points with mask
visualizer.set_points(points, pcd_mode=2, vis_mode='add')
visualizer.draw_seg_mask(points_with_mask)
visualizer.show()

