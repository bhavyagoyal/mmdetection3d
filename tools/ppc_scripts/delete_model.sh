for (( i = 1; i < 10; i++ ))
do
	find ../../work_dir_py/kitti/pvrcnn/ -iname epoch_$i.pth | xargs rm
done

for (( i = 11; i < 50; i++ ))
do
	find ../../work_dir_py/kitti/pvrcnn -iname epoch_$i.pth | xargs rm
done
