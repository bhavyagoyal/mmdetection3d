import pickle
import copy
import numpy as np
import random
splits = ['train', 'val']
SBR='1_50'


## create clean point cloud with extra feature containing confidence/probability as one
BASE = '../../data/sunrgbd/points_clean8/'
OUTBASE = '../../data/sunrgbd/points_gaussian/'
#for i in range(10335):
#    fname = str(i+1).zfill(6) + '.bin'
#    print(fname)
#    points = np.fromfile(BASE + fname, dtype=np.float32)
#    points = points.reshape(-1,6)
#    ones = np.ones((points.shape[0], 1), dtype=np.float32)
#    points = np.concatenate([points[:,:3], ones*2., points[:,3:]], 1)
#    points.tofile(OUTBASE+fname)
    

for sp in splits:
    with open('../../data/sunrgbd/sunrgbd_infos_' + sp + '.pkl', 'rb') as f:
        data = pickle.load(f)
    data_list = data['data_list']

    #data_list = data_list[:1]
    #data_list[0]['instances']=[]
    #data['data_list'] = data_list

    #r = np.random.choice(range(len(data_list)), 100, replace=False).tolist()
    #print(r)
    #final_list = [data_list[x] for x in r]

    #np.savetxt(save_path, pcl_denoised.numpy(), fmt='%.8f')
    #SBR = ['1_100', '1_50', '5_100', '5_50', 'clean']
    #final_list = []
    noise_max = [0.01, 0.1, 1.0]
    for idx, _ in enumerate(data_list):
        #points = np.fromfile('../../data/sunrgbd/points_min2/0.3/argmax-filtering-sbr/5_50/'+ data_list[idx]['lidar_points']['lidar_path'], dtype=np.float32)
        points = np.fromfile(BASE+data_list[idx]['lidar_points']['lidar_path'], dtype=np.float32)
        points = points.reshape(-1,8)
        points3d = points[:,:3]
        for noise in noise_max:
            noise_std = random.uniform(0, noise)
            points3d = points3d + np.random.normal(size=points3d.shape) * noise_std
            pointsout = np.concatenate([points3d.astype(np.float32), points[:,3:]], axis=1)
            pointsout.tofile(OUTBASE + str(noise) + '/' + data_list[idx]['lidar_points']['lidar_path'])
        #np.savetxt('txt/points_clean8/' + sp + '/' + data_list[idx]['lidar_points']['lidar_path'][:-4]+'.xyz', points, fmt='%.8f')
        #for sbr in SBR:
        #    data_elem = copy.deepcopy(data_list[idx])
        #    data_elem['lidar_points']['lidar_path'] = sbr + '/' + data_elem['lidar_points']['lidar_path']
        #    final_list.append(data_elem)
    #print(len(final_list))
    
    #data['data_list'] = final_list
    #SBRstr = '_'.join(SBR)
    #with open('../../data/sunrgbd/sunrgbd_infos_' + sp + '_' + SBRstr + '_shortvis.pkl', 'wb') as f:
    #with open('../../data/sunrgbd/sunrgbd_infos_' + sp + '_one.pkl', 'wb') as f:
    #    pickle.dump(data, f)
   

