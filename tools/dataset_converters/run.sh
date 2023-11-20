START=$1
TOTAL=$2
EACH=$3
SPLITS=$TOTAL/$EACH
for (( i = 0; i < $SPLITS; i++ ))
do
        BEGIN=$(($i*$EACH+$START))
        echo $BEGIN
	python -u getdepth.py --method=peaks-confidence --sbr=1_50 \
	--start $BEGIN --end $(($BEGIN+$EACH)) &
        sleep 5
done

wait

