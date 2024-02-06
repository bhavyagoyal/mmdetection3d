import pickle
import copy
import numpy as np
splits = ['train', 'val']
SBR='1_50'
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
    

