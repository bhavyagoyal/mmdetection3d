import numpy as np

from mmdet3d.visualization import Det3DLocalVisualizer
import matplotlib
import seaborn as sns
import sys


fname = sys.argv[1]
points = np.fromfile(fname, dtype=np.float32)
#points = np.fromfile('data/sunrgbd/points_min2/1.0/score-denoised_sunrgbd1000/argmax-filtering-sbr/5_50_testing/pcl/000001.bin', dtype=np.float32)
#points = np.fromfile('data/sunrgbd/points_gaussian/score-denoised/0.2/pcl/000001.bin', dtype=np.float32)
#points = np.fromfile('data/sunrgbd/points_gaussian/0.2/000001.bin', dtype=np.float32)
#points = np.fromfile('data/sunrgbd/points_min2/1.0/argmax-filtering-sbr/5_50/000001.bin', dtype=np.float32)
print(points.shape)
points = points.reshape(-1,8)

#choices = np.random.choice(points.shape[0], 5000, replace=False)
#points = points[choices]
#points = points.reshape(-1,6)
#points = points.reshape(-1, 6)[:,:3]

points_color = points[:,4]
#points_color = points[:,5:8]
points_color = points_color/points_color.max()
points_color = sns.color_palette('coolwarm', as_cmap=True)(points_color)[:,:3]

points = points[:,:3]
visualizer = Det3DLocalVisualizer()

#visualizer.set_points(points, pcd_mode=2, vis_mode='add')
visualizer.set_points(points, pcd_mode=2, vis_mode='add', points_color = points_color)
    
visualizer.show()

