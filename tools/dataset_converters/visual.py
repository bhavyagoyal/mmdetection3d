import numpy as np

from mmdet3d.visualization import Det3DLocalVisualizer
import matplotlib
import seaborn as sns


#points = np.fromfile('demo/data/sunrgbd/000017.bin', dtype=np.float32)
#points = np.fromfile('../../data/sunrgbd/points_5_5_gtdepth/005050.bin', dtype=np.float32)
#points = np.fromfile('pointscheckclean/points0.bin', dtype=np.float32)
#points = np.fromfile('pointscheck/test_peaks-confidence-1-50-fps5000-sp03-1-0.bin', dtype=np.float32)
#points = np.fromfile('pointscheck/denoise-softmax_1_50_005050.bin', dtype=np.float32)
points = np.fromfile('pointscheck/5_50_score_denoise_005050.bin', dtype=np.float32)
#points = np.fromfile('pointscheck/5_50_argmax_filtering_conf_005050.bin', dtype=np.float32)
#points = np.fromfile('pointscheck/5_5_gtdepth_005050.bin', dtype=np.float32)
#points = np.fromfile('pointscheck/5_50_wsp_cf_005050.bin', dtype=np.float32)
#points = np.fromfile('pointscheck/velodyne_000001.bin', dtype=np.float32)
#points = np.fromfile('../../data/sunrgbd/points_bk/005050.bin', dtype=np.float32)
#points = np.fromfile('/nobackup/bhavya/votenet/sunrgbd/matlab/testing_depth.bin', dtype=np.float32)
print(points.shape)
points = points.reshape(-1,7)
#choices = np.random.choice(points.shape[0], 5000, replace=False)
#points = points[choices]
#points = points.reshape(-1,6)
#points = points.reshape(-1, 6)[:,:3]
#points_color = points[:,4]
points_color = points[:,4:7]
#points_color = points_color/points_color.max()
#points_color = sns.color_palette('coolwarm', as_cmap=True)(points_color)[:,:3]
points = points[:,:3]
#points = points.reshape(-1, 3)
visualizer = Det3DLocalVisualizer()
#mask = np.random.rand(points.shape[0], 3)
#points_with_mask = np.concatenate((points, mask), axis=-1)
# Draw 3D points with mask
#visualizer.set_points(points, pcd_mode=2, vis_mode='add')
visualizer.set_points(points, pcd_mode=2, vis_mode='add', points_color = points_color)
    
#visualizer.draw_seg_mask(points_with_mask)
visualizer.show()

