MCMT_CONFIG_FILE="aic_mcmt.yml"
#### Run Detector.####
cd detector/
python gen_images_aic.py ${MCMT_CONFIG_FILE}

cd yolov5/
#sh gen_det.sh ${MCMT_CONFIG_FILE}

#### Extract reid feautres.####
cd ../../reid/
#python extract_image_feat.py "aic_reid1.yml"
#python extract_image_feat.py "aic_reid2.yml"
#python extract_image_feat.py "aic_reid3.yml"
#python merge_reid_feat.py ${MCMT_CONFIG_FILE}

#### MOT. ####
cd ../tracker/MOTBaseline
sh run_aic.sh ${MCMT_CONFIG_FILE}
wait
#### Get results. ####
cd ../../reid/reid-matching/tools
python trajectory_fusion.py ${MCMT_CONFIG_FILE}
python sub_cluster.py ${MCMT_CONFIG_FILE}
python gen_res.py ${MCMT_CONFIG_FILE}

#### Vis. (optional) ####
# python viz_mot.py ${MCMT_CONFIG_FILE}
# python viz_mcmt.py ${MCMT_CONFIG_FILE}
