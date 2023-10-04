# Download SUNRGBD.zip dataset file and copy images to sunrgbd_trainval/image directory. 

count=0
cat sunrgbd-meta-data/sunrgbd_testing_images.txt | while read -r line; do
    ((count++))
    echo "$line" $(printf "%06d.jpg" $count)
    cp "$line" "image/$(printf "%06d.jpg" $count)"
done

cat sunrgbd-meta-data/sunrgbd_training_images.txt | while read -r line; do
    ((count++))
    echo "$line" $(printf "%06d.jpg" $count)
    cp "$line" "image/$(printf "%06d.jpg" $count)"
done



