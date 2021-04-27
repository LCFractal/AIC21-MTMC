seqs=(c041 c042 c043 c044 c045 c046)
# seqs=(c041)
gpu_id=0
for seq in ${seqs[@]}
do
    CUDA_VISIBLE_DEVICES=${gpu_id} python detect2img.py --name ${seq} --weights yolov5x.pt --conf 0.1 --agnostic --save-txt --save-conf --img-size 1280 --classes 2 5 7 --cfg_file $1&
    gpu_id=$(($gpu_id+1))
done
wait
