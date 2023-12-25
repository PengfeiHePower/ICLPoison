# clean eval
./cleanEval.sh


### pythia-2.8B
# poison script, all layer
python -u scripts/experiments/iclpoison_all.py --task_name poem_sentiment --dataset poem_sentiment --model_type pythia --model_variant 2.8B > logs/icvpoison/token2_all_poem_pythia2.8B.log 2>&1
python -u scripts/experiments/iclpoison_all.py --task_name glue-cola --dataset glue-cola --model_type pythia --model_variant 2.8B > logs/icvpoison/token2_all_cola_pythia2.8B.log 2>&1
python -u scripts/experiments/iclpoison_all.py --task_name glue-sst2 --dataset glue-sst2 --model_type pythia --model_variant 2.8B > logs/icvpoison/token2_all_sst2_pythia2.8B.log 2>&1
python -u scripts/experiments/iclpoison_all.py --task_name emo --dataset emo --model_type pythia --model_variant 2.8B > logs/icvpoison/token2_all_emo_pythia2.8B.log 2>&1
python -u scripts/experiments/iclpoison_all.py --task_name  --dataset ag_news --model_type pythia --model_variant 2.8B > logs/icvpoison/token2_all_ag_pythia2.8B.log 2>&1

# poison script, best layer
python -u scripts/experiments/iclpoison_best.py --task_name poem_sentiment --dataset poem_sentiment --model_type pythia --model_variant 2.8B > logs/icvpoison/token2_best_poem_pythia2.8B.log 2>&1
python -u scripts/experiments/iclpoison_best.py --task_name glue-cola --dataset glue-cola --model_type pythia --model_variant 2.8B > logs/icvpoison/token2_best_cola_pythia2.8B.log 2>&1
python -u scripts/experiments/iclpoison_best.py --task_name glue-sst2 --dataset glue-sst2 --model_type pythia --model_variant 2.8B > logs/icvpoison/token2_best_sst2_pythia2.8B.log 2>&1
python -u scripts/experiments/iclpoison_best.py --task_name emo --dataset emo --model_type pythia --model_variant 2.8B > logs/icvpoison/token2_best_emo_pythia2.8B.log 2>&1
python -u scripts/experiments/iclpoison_best.py --task_name  --dataset ag_news --model_type pythia --model_variant 2.8B > logs/icvpoison/token2_best_ag_pythia2.8B.log 2>&1

### pythia-6.9B
# poison script, all layer
python -u scripts/experiments/iclpoison_all.py --task_name poem_sentiment --dataset poem_sentiment --model_type pythia --model_variant 6.9B > logs/icvpoison/token2_all_poem_pythia6.9B.log 2>&1
python -u scripts/experiments/iclpoison_all.py --task_name glue-cola --dataset glue-cola --model_type pythia --model_variant 6.9B > logs/icvpoison/token2_all_cola_pythia6.9B.log 2>&1
python -u scripts/experiments/iclpoison_all.py --task_name glue-sst2 --dataset glue-sst2 --model_type pythia --model_variant 6.9B > logs/icvpoison/token2_all_sst2_pythia6.9B.log 2>&1
python -u scripts/experiments/iclpoison_all.py --task_name emo --dataset emo --model_type pythia --model_variant 6.9B > logs/icvpoison/token2_all_emo_pythia6.9B.log 2>&1
python -u scripts/experiments/iclpoison_all.py --task_name  --dataset ag_news --model_type pythia --model_variant 6.9B > logs/icvpoison/token2_all_ag_pythia6.9B.log 2>&1

# poison script, best layer
python -u scripts/experiments/iclpoison_best.py --task_name poem_sentiment --dataset poem_sentiment --model_type pythia --model_variant 6.9B > logs/icvpoison/token2_best_poem_pythia6.9B.log 2>&1
python -u scripts/experiments/iclpoison_best.py --task_name glue-cola --dataset glue-cola --model_type pythia --model_variant 6.9B > logs/icvpoison/token2_best_cola_pythia6.9B.log 2>&1
python -u scripts/experiments/iclpoison_best.py --task_name glue-sst2 --dataset glue-sst2 --model_type pythia --model_variant 6.9B > logs/icvpoison/token2_best_sst2_pythia6.9B.log 2>&1
python -u scripts/experiments/iclpoison_best.py --task_name emo --dataset emo --model_type pythia --model_variant 6.9B > logs/icvpoison/token2_best_emo_pythia6.9B.log 2>&1
python -u scripts/experiments/iclpoison_best.py --task_name  --dataset ag_news --model_type pythia --model_variant 6.9B > logs/icvpoison/token2_best_ag_pythia6.9B.log 2>&1


# poison eval
python -u scripts/experiments/icleval.py --task_name poem_sentiment --dataset poem_sentiment --model_type pythia --model_variant 2.8B --clean False --adv_trainpath /home/pengfei/Documents/icl_task_vectors/adv_train/poem_sentiment/pythia2.8B_all  > logs/poisonEval/all_poem_pythia2.8B.log 2>&1
python -u scripts/experiments/icleval.py --task_name poem_sentiment --dataset poem_sentiment --model_type pythia --model_variant 2.8B --clean False --adv_trainpath /home/pengfei/Documents/icl_task_vectors/adv_train/poem_sentiment/pythia2.8B_best  > logs/poisonEval/best_poem_pythia2.8B.log 2>&1
python -u scripts/experiments/icleval.py --task_name glue-cola --dataset glue-cola --model_type pythia --model_variant 2.8B --clean False --adv_trainpath /home/pengfei/Documents/icl_task_vectors/adv_train/glue-cola/pythia2.8B_all  > logs/poisonEval/all_cola_pythia2.8B.log 2>&1
python -u scripts/experiments/icleval.py --task_name glue-cola --dataset glue-cola --model_type pythia --model_variant 2.8B --clean False --adv_trainpath /home/pengfei/Documents/icl_task_vectors/adv_train/glue-cola/pythia2.8B_best  > logs/poisonEval/best_cola_pythia2.8B.log 2>&1
python -u scripts/experiments/icleval.py --task_name glue-sst2 --dataset glue-sst2 --model_type pythia --model_variant 2.8B --clean False --adv_trainpath /home/pengfei/Documents/icl_task_vectors/adv_train/glue-sst2/pythia2.8B_all  > logs/poisonEval/all_sst2_pythia2.8B.log 2>&1
python -u scripts/experiments/icleval.py --task_name glue-sst2 --dataset glue-sst2 --model_type pythia --model_variant 2.8B --clean False --adv_trainpath /home/pengfei/Documents/icl_task_vectors/adv_train/glue-sst2/pythia2.8B_best  > logs/poisonEval/best_sst2_pythia2.8B.log 2>&1