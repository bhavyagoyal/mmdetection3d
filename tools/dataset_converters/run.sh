START=$1
TOTAL=$2
EACH=$3
SPLITS=$TOTAL/$EACH
for (( i = 0; i < $SPLITS; i++ ))
do
        BEGIN=$(($i*$EACH+$START))
        echo $BEGIN
	python -u getdepth.py --method=argmax-filtering-conf --sbr=5_1 \
	--start $BEGIN --end $(($BEGIN+$EACH)) &
        sleep 5
done

wait

