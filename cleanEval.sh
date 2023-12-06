d=$1

# Log file
logs_dir="logs/cleanEval"
log_file="$logs_dir/$d.log"
mkdir -p $logs_dir

echo "Starting..."

# CUDA_VISIBLE_DEVICES=$2 python -u -m $script_full_path > $log_file 2>&1 &
python -u -m scripts.experiments.icleval --dataset $d --task_name $d > $log_file 2>&1 &

echo "Watch the log file with:"
echo "wt $log_file"