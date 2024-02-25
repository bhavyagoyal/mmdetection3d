import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
plt.style.use('ggplot')
matplotlib.use('Agg')
matplotlib.rcParams['text.color'] = 'black'
matplotlib.rcParams['axes.labelcolor'] = 'black'
matplotlib.rcParams['xtick.color'] = 'black'
matplotlib.rcParams['ytick.color'] = 'black'
#params = {'legend.fontsize': 'x-large',
#         'axes.labelsize': 'x-large',
#         'axes.titlesize':'x-large',
#         'xtick.labelsize':'x-large',
#         'ytick.labelsize':'x-large'}
#matplotlib.rcParams.update(params)
matplotlib.rcParams.update({'font.size': 12})

#fig, ax = plt.subplots()
x = ['0.1', '0.05', '0.02', '0.01']
p1 = [10, 10, 10, 10]
p2 = [20, 20., 20., 20]
p3 = [54.29, 52.46, 38.49, 29.42]
p4 = [52, 52, 52, 52]

#plt.plot(x, p1, marker='D', linewidth=3.0, markersize=7.0, label='PPC w/o NPD Filtering')
#plt.plot(x, p2, marker='D', linewidth=3.0, markersize=7.0, label='PPC w/o FPS Cap')
plt.plot(x, p4, marker='D', linewidth=3.0, markersize=7.0, label='PPC w/o Probability')
plt.plot(x, p3, marker='D', linewidth=3.0, markersize=7.0, label='PPC')
plt.xlabel('SBR')
plt.ylabel('mAP 3D Object Detection')
plt.legend()
plt.tight_layout()
#plt.gca().yaxis.grid(color='black')
#plt.savefig('ablation_wonpdfps.pdf')
plt.savefig('ablation_woprobs.pdf')


