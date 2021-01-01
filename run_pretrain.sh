export CUDA_VISIBLE_DEVICES=$1
LOG_DIR=$2
DATASET=$3
LABELED_LIST=$4
mkdir -p "${LOG_DIR}"
python -u pretrain.py --log_dir="${LOG_DIR}" --dataset="${DATASET}" \
--labeled_sample_list="${LABELED_LIST}" 2>&1|tee "${LOG_DIR}"/LOG_ALL.log &