export CUDA_VISIBLE_DEVICES=$1
LOG_DIR=$2
DATASET=$3
LABELED_LIST=$4
CKPT=$5
mkdir -p "${LOG_DIR}";
python -u train.py --log_dir="${LOG_DIR}" --dataset="${DATASET}" --detector_checkpoint="${CKPT}" \
--labeled_sample_list="${LABELED_LIST}" --use_iou_for_nms --eval
