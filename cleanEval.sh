modeltype=$1
modelvatriant=$2

# Log file
logs_dir1="logs/cleanEval/cola_one_"
log_name=$modeltype$modelvatriant
log_file1="$logs_dir1$log_name.log"
# mkdir -p $logs_dir

echo "Starting..."
echo "Watch the log file with:"
echo "wt $log_file1"
# CUDA_VISIBLE_DEVICES=$2 python -u -m $script_full_path > $log_file 2>&1 &
python -u -m scripts.experiments.icleval --task_name glue-cola --model_type $modeltype --model_variant $modelvatriant > $log_file1 2>&1 &&



# Log file
logs_dir2="logs/cleanEval/sst2_one_"
log_name=$modeltype$modelvatriant
log_file2="$logs_dir2$log_name.log"
# mkdir -p $logs_dir

echo "Starting..."
echo "Watch the log file with:"
echo "wt $log_file2"
# CUDA_VISIBLE_DEVICES=$2 python -u -m $script_full_path > $log_file 2>&1 &
python -u -m scripts.experiments.icleval --task_name glue-sst2 --model_type $modeltype --model_variant $modelvatriant > $log_file2 2>&1 &



# # Log file
# logs_dir3="logs/cleanEval/emo_"
# log_name=$modeltype$modelvatriant
# log_file3="$logs_dir3$log_name.log"
# # mkdir -p $logs_dir

# echo "Starting..."
# echo "Watch the log file with:"
# echo "wt $log_file3"
# # CUDA_VISIBLE_DEVICES=$2 python -u -m $script_full_path > $log_file 2>&1 &
# python -u -m scripts.experiments.icleval --task_name emo --model_type $modeltype --model_variant $modelvatriant > $log_file3 2>&1 &&



# # Log file
# logs_dir4="logs/cleanEval/poem_"
# log_name=$modeltype$modelvatriant
# log_file4="$logs_dir4$log_name.log"
# # mkdir -p $logs_dir

# echo "Starting..."
# echo "Watch the log file with:"
# echo "wt $log_file4"
# # CUDA_VISIBLE_DEVICES=$2 python -u -m $script_full_path > $log_file 2>&1 &
# python -u -m scripts.experiments.icleval --task_name poem_sentiment --model_type $modeltype --model_variant $modelvatriant > $log_file4 2>&1 &&



# # Log file
# logs_dir5="logs/cleanEval/ag_"
# log_name=$modeltype$modelvatriant
# log_file5="$logs_dir5$log_name.log"
# # mkdir -p $logs_dir

# echo "Starting..."
# echo "Watch the log file with:"
# echo "wt $log_file5"
# # CUDA_VISIBLE_DEVICES=$2 python -u -m $script_full_path > $log_file 2>&1 &
# python -u -m scripts.experiments.icleval --task_name ag_news --model_type $modeltype --model_variant $modelvatriant > $log_file5 2>&1 &