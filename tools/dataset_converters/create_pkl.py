import pickle
import copy
import numpy as np
splits = ['train', 'val']
SBR='1_50'


## create clean point cloud with extra feature containing confidence/probability as one
#BASE = '../../data/sunrgbd/points_clean/'
#OUTBASE = '../../data/sunrgbd/points_clean7/'
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
    r = np.random.choice(range(len(data_list)), 100, replace=False).tolist()
    print(r)
    final_list = [data_list[x] for x in r]

    #SBR = ['1_100', '1_50', '5_100', '5_50', 'clean']
    #final_list = []
    #for idx, _ in enumerate(data_list):
    #    for sbr in SBR:
    #        data_elem = copy.deepcopy(data_list[idx])
    #        data_elem['lidar_points']['lidar_path'] = sbr + '/' + data_elem['lidar_points']['lidar_path']
    #        final_list.append(data_elem)
    #
    #print(len(final_list))
    
    data['data_list'] = final_list
    SBRstr = '_'.join(SBR)
    #with open('../../data/sunrgbd/sunrgbd_infos_' + sp + '_' + SBRstr + '_shortvis.pkl', 'wb') as f:
    with open('../../data/sunrgbd/sunrgbd_infos_' + sp + '_shortvis.pkl', 'wb') as f:
        pickle.dump(data, f)
   

