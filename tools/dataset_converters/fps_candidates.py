import numpy as np

from mmdet3d.visualization import Det3DLocalVisualizer
import matplotlib
import seaborn as sns

psize = 4
for idx, pref in enumerate(['clean', '5-50-sp0004', '5-50-sp0004-newfpscf10']):
    points = np.fromfile('centroids/'+pref+'-0-0-points.bin', dtype=np.float32)
    indices = np.fromfile('centroids/'+pref+'-1-0-indices.bin', dtype=np.int64)

    visualizer = Det3DLocalVisualizer()
    if(idx>=2):
        points = points.reshape(-1,10)
        #indices = np.random.choice(range(points.shape[0]), 10000, replace=False)
        points = points[indices]
        points_xyz = points[:,:3]
        points_color = points[:,4]
        #points_color = (points_color-points_color.min())/points_color.max()
        points_color = (points_color-0.3)/4.0
        points_color = sns.color_palette('coolwarm', as_cmap=True)(points_color)[:,:3]
        visualizer.set_points(points_xyz, pcd_mode=2, vis_mode='add', points_color = points_color, points_size=psize)
    elif(idx==1):
        points = points.reshape(-1,10)
        #indices = np.random.choice(range(points.shape[0]), 10000, replace=False)
        points = points[indices]
        points_xyz = points[:,:3]
        visualizer.set_points(points_xyz, pcd_mode=2, vis_mode='add', points_size=psize)
    else:
        points = points.reshape(-1,8)
        #indices = np.random.choice(range(points.shape[0]), 50000, replace=False)
        points = points[indices]
        points_xyz = points[:,:3]
        visualizer.set_points(points_xyz, pcd_mode=2, vis_mode='add', points_size=psize)
    with open('cam_selected/000001.json', 'rb') as f:
        visualizer.o3d_vis.set_view_status(f.read().strip())
    visualizer.o3d_vis.capture_screen_image(f"{idx}-center.jpg", True)
    visualizer.show()


#visualizer.o3d_vis.get_view_control().set_front([0,-1,0])
#visualizer.o3d_vis.get_view_control().set_lookat([0,4,0])
#visualizer.o3d_vis.get_view_control().set_up([0,0,1])
#visualizer.o3d_vis.get_view_control().set_zoom(0.5)

