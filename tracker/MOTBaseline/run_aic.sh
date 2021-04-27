
cd src
seqs=(c041 c042 c043 c044 c045 c046)
# seqs=(c042)

TrackOneSeq(){
    seq=$1
    config=$2
    echo tracking $seq with ${config}
    python -W ignore fair_app.py \
        --min_confidence=0.1 \
        --display=False \
        --max_frame_idx -1 \
        --nms_max_overlap 0.99 \
        --min-box-area 750 \
        --cfg_file ${config} \
        --seq_name ${seq} \
        --max_cosine_distance 0.5

    cd ./post_processing
    python main.py ${seq} pp ${config}
    cd ../
}

for seq in ${seqs[@]}
do 
    TrackOneSeq ${seq} $1 &
done
wait
