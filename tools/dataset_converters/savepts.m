clear; close all;
addpath(genpath('.'))
addpath('../OFFICIAL_SUNRGBD/SUNRGBDtoolbox/readData')

load('../OFFICIAL_SUNRGBD/SUNRGBDMeta3DBB_v2.mat');
load('../OFFICIAL_SUNRGBD/SUNRGBDMeta2DBB_v2.mat');

BASEFOLDER = '/nobackup/bhavya/mmdetection3d/data/sunrgbd/sunrgbd_trainval/';
POINTS = 'processed_full/SimSPADDataset_nr-576_nc-704_nt-1024_tres-586ps_dark-0_psf-0/';
SBR = '50_1';

for imageId = 1:5050
	data = SUNRGBDMeta(imageId);
	imageIdst = num2str(imageId, '%06d')
	data.depthpath = fullfile(BASEFOLDER, POINTS, sprintf('spad_%s_%s_argmax.png', imageIdst, SBR));
	data.rgbpath = fullfile(BASEFOLDER, sprintf('image/%s.jpg', imageIdst));
	%data
	
	[rgb,points3d,depthInpaint,imsize]=read3dPoints(data);
	rgb(isnan(points3d(:,1)),:) = [];
	points3d(isnan(points3d(:,1)),:) = [];
	points3d_rgb = [points3d, rgb];
	
	mat_filename = fullfile(BASEFOLDER, POINTS, sprintf('spad_%s_%s_argmax.mat', imageIdst, SBR));
	parsave(mat_filename, points3d_rgb);
end

function parsave(filename, instance)
save(filename, 'instance');
end

