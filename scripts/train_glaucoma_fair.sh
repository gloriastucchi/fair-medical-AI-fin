#!/bin/bash
TASK=cls
MODEL_TYPE='efficientnet'
LOSS_TYPE='bce' 
LR=5e-5
NUM_EPOCH=10
BATCH_SIZE=6
STRETCH_RATIO=5
MODALITY_TYPE='rnflt'
ATTRIBUTE_TYPE='race'
PERF_FILE=${MODEL_TYPE}_${MODALITY_TYPE}_${ATTRIBUTE_TYPE}.csv
python train_glaucoma_fair.py \
	--data_dir /Users/gloriastucchi/Desktop/dataset \
	--result_dir ./results/crosssectional_${MODALITY_TYPE}_${ATTRIBUTE_TYPE}/fullysup_${MODEL_TYPE}_${MODALITY_TYPE}_Task${TASK}_lr${LR}_bz${BATCH_SIZE} \
	--model_type ${MODEL_TYPE} \
	--image_size 200 \
	--loss_type ${LOSS_TYPE} \
	--lr ${LR} --weight-decay 0. --momentum 0.1 \
	--batch-size ${BATCH_SIZE} \
	--task ${TASK} \
	--epochs ${NUM_EPOCH} \
	--modality_types ${MODALITY_TYPE} \
	--perf_file ${PERF_FILE} \
	--attribute_type ${ATTRIBUTE_TYPE}