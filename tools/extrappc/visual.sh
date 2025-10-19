filepath=000040.bin

# python tools/extrappc/visual.py ~/Documents/ppcshared/datasets/sunrgbd/points_ppc/clean/${filepath} 8 -1 0.0 &
# # python tools/extrappc/visual.py ~/Documents/ppcshared/datasets/sunrgbd/points_ppc/clean/${filepath} 8 3 0.0 &
# python tools/extrappc/visual.py ~/Documents/ppcshared/datasets/sunrgbd/points_ppc/5_50/${filepath} 8 4 1.0 &
# python tools/extrappc/visual.py ~/Documents/ppcshared/datasets/sunrgbd/points_ppc/5_100/${filepath} 8 4 1.2 &
# # python tools/extrappc/visual.py ~/Documents/ppcshared/datasets/sunrgbd/points_ppc/1_50/${filepath} 8 4 1. &


filepath=000001.bin
python tools/extrappc/visual.py ~/Documents/ppcshared/datasets/kitti/points_ppc/clean/${filepath} 6 5 0.0 &
python tools/extrappc/visual.py ~/Documents/ppcshared/datasets/kitti/points_ppc/5_100/${filepath} 6 4 1.5 &
python tools/extrappc/visual.py ~/Documents/ppcshared/datasets/kitti/points_ppc/5_250/${filepath} 6 4 1.5 &

