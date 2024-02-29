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
x = ['Clean', '0.1', '0.05', '0.02', '0.01']
p3 = [58.61, 54.29, 52.46, 38.49, 29.42]
p4 = [58.11, 52.89, 48.47, 35.13, 23.68]
p1 = [58.61, 54.11, 50.03, 36.45, 25.33]
p2 = [58.45, 49.23, 45.63, 26.44, 15.44]
p5 = [58.34, 42.43, 38.77, 16.95, 11.34]

plt.plot(x, p3, marker='D', linewidth=3.0, markersize=7.0, label='PPC')
#plt.plot(x, p1, marker='D', linewidth=3.0, markersize=7.0, label='PPC w/o FPPS')
#plt.plot(x, p2, marker='D', linewidth=3.0, markersize=7.0, label='PPC w/o NPD Filtering')
#plt.plot(x, p5, marker='D', linewidth=3.0, markersize=7.0, label='PPC w/o FPPS & NPD Filtering')
plt.plot(x, p4, marker='D', linewidth=3.0, markersize=7.0, label='PPC w/o Probability')
plt.xlabel('SBR')
plt.ylabel('mAP 3D Object Detection')
plt.legend()
plt.tight_layout()
#plt.gca().yaxis.grid(color='black')
#plt.savefig('ablation_wonpdfps.pdf')
plt.savefig('ablation_woprobs.pdf')


