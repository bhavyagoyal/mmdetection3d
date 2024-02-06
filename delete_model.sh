for (( i = 1; i < 12; i++ ))
do
	find work_dir_py -iname epoch_$i.pth | xargs rm
done

for (( i = 13; i < 36; i++ ))
do
	find work_dir_py -iname epoch_$i.pth | xargs rm
done
