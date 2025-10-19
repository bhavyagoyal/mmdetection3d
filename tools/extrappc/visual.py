import numpy as np

from mmdet3d.visualization import Det3DLocalVisualizer
import matplotlib
import seaborn as sns
import sys
import pandas
import open3d as o3d

fname = sys.argv[1]
num_points = int(sys.argv[2])
color_idx = int(sys.argv[3])
thresh = float(sys.argv[4])


points = np.fromfile(fname, dtype=np.float32)
print(points.shape)
points = points.reshape(-1,num_points)

#points = pandas.read_csv(fname) 
#points = points.to_numpy()

#choices = np.random.choice(points.shape[0], 5000, replace=False)
choices = points[:,3]>thresh
points = points[choices]

if(color_idx==-1):
    points_color = points[:,5:8]
else:
    points_color = points[:,color_idx]
    pt_cls = sorted(points_color)
    mn, mx = pt_cls[0], pt_cls[-200]
    points_color = np.clip(points_color, mn, mx)
    points_color = (points_color-mn)/(mx-mn)
    # points_color = points_color/points_color.max()
    points_color = sns.color_palette('coolwarm', as_cmap=True)(points_color)[:,:3]

points = points[:,:3]
print('X ', points[:,0].mean(), ' Y ', points[:,1].mean(), ' Z ', points[:,2].mean(), ' P ', points_color.mean())
#visualizer = Det3DLocalVisualizer()

#visualizer.set_points(points, pcd_mode=2, vis_mode='add')
#visualizer.set_points(points, pcd_mode=2, vis_mode='add', points_color = points_color)
    
#visualizer.show()


def visualize_points(points, colors=None):
    """
    points: numpy array of shape (N, 3)
    colors: numpy array of shape (N, 3) in range [0, 1] or None
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Visualize
    o3d.visualization.draw_geometries([pcd])


# 2. With colors
visualize_points(points, points_color)





