for (( i = 2; i < 100; i++ ))
do
	fname=$(printf %06d $i)
	cp flim/000001-camera.json flim/$fname-camera.json
done


