# clean eval
./cleanEval.sh
### pythia-2.8B
python -u scripts/experiments/icleval.py --task_name glue-cola --dataset glue-cola --model_type pythia --model_variant 2.8B > logs/cleanEval/cola_pythia2.8B.log 2>&1
python -u scripts/experiments/icleval.py --task_name emo --dataset emo --model_type pythia --model_variant 2.8B > logs/cleanEval/emo_pythia2.8B.log 2>&1
python -u scripts/experiments/icleval.py --task_name glue-sst2 --dataset glue-sst2 --model_type pythia --model_variant 2.8B > logs/cleanEval/sst2_pythia2.8B.log 2>&1
python -u scripts/experiments/icleval.py --task_name poem_sentiment --dataset poem_sentiment --model_type pythia --model_variant 2.8B > logs/cleanEval/poem_pythia2.8B.log 2>&1
python -u scripts/experiments/icleval.py --task_name ag_news --dataset ag_news --model_type pythia --model_variant 2.8B > logs/cleanEval/ag_pythia2.8B.log 2>&1
### pythia-6.9B
python -u scripts/experiments/icleval.py --task_name glue-cola --dataset glue-cola --model_type pythia --model_variant 6.9B > logs/cleanEval/cola_pythia6.9B.log 2>&1
python -u scripts/experiments/icleval.py --task_name emo --dataset emo --model_type pythia --model_variant 6.9B > logs/cleanEval/emo_pythia6.9B.log 2>&1
python -u scripts/experiments/icleval.py --task_name glue-sst2 --dataset glue-sst2 --model_type pythia --model_variant 6.9B > logs/cleanEval/sst2_pythia6.9B.log 2>&1
python -u scripts/experiments/icleval.py --task_name poem_sentiment --dataset poem_sentiment --model_type pythia --model_variant 6.9B > logs/cleanEval/poem_pythia6.9B.log 2>&1
python -u scripts/experiments/icleval.py --task_name ag_news --dataset ag_news --model_type pythia --model_variant 6.9B > logs/cleanEval/ag_pythia6.9B.log 2>&1
### llama2-7B
python -u scripts/experiments/icleval.py --task_name glue-cola --dataset glue-cola --model_type llama --model_variant 7B > logs/cleanEval/cola_llama7B.log 2>&1
python -u scripts/experiments/icleval.py --task_name emo --dataset emo --model_type llama --model_variant 7B > logs/cleanEval/emo_llama7B.log 2>&1
python -u scripts/experiments/icleval.py --task_name glue-sst2 --dataset glue-sst2 --model_type llama --model_variant 7B > logs/cleanEval/sst2_llama7B.log 2>&1
python -u scripts/experiments/icleval.py --task_name poem_sentiment --dataset poem_sentiment --model_type llama --model_variant 7B > logs/cleanEval/poem_llama7B.log 2>&1
python -u scripts/experiments/icleval.py --task_name ag_news --dataset ag_news --model_type llama --model_variant 7B > logs/cleanEval/ag_llama7B.log 2>&1
### llama2-13B
python -u scripts/experiments/icleval.py --task_name glue-cola --dataset glue-cola --model_type llama --model_variant 13B > logs/cleanEval/cola_llama13B.log 2>&1
python -u scripts/experiments/icleval.py --task_name emo --dataset emo --model_type llama --model_variant 13B > logs/cleanEval/emo_llama13B.log 2>&1
python -u scripts/experiments/icleval.py --task_name glue-sst2 --dataset glue-sst2 --model_type llama --model_variant 13B > logs/cleanEval/sst2_llama13B.log 2>&1
python -u scripts/experiments/icleval.py --task_name poem_sentiment --dataset poem_sentiment --model_type llama --model_variant 13B > logs/cleanEval/poem_llama13B.log 2>&1
python -u scripts/experiments/icleval.py --task_name ag_news --dataset ag_news --model_type llama --model_variant 13B > logs/cleanEval/ag_llama13B.log 2>&1
### falcon-7B
python -u scripts/experiments/icleval.py --task_name glue-cola --dataset glue-cola --model_type falcon --model_variant 7B > logs/cleanEval/cola_falcon7B.log 2>&1
python -u scripts/experiments/icleval.py --task_name emo --dataset emo --model_type falcon --model_variant 7B > logs/cleanEval/emo_falcon7B.log 2>&1
python -u scripts/experiments/icleval.py --task_name glue-sst2 --dataset glue-sst2 --model_type falcon --model_variant 7B > logs/cleanEval/sst2_falcon7B.log 2>&1
python -u scripts/experiments/icleval.py --task_name poem_sentiment --dataset poem_sentiment --model_type falcon --model_variant 7B > logs/cleanEval/poem_falcon7B.log 2>&1
python -u scripts/experiments/icleval.py --task_name ag_news --dataset ag_news --model_type falcon --model_variant 7B > logs/cleanEval/ag_falcon7B.log 2>&1
### mpt-7B
python -u scripts/experiments/icleval.py --task_name glue-cola --dataset glue-cola --model_type mpt --model_variant 7B > logs/cleanEval/cola_mpt7B.log 2>&1
python -u scripts/experiments/icleval.py --task_name emo --dataset emo --model_type mpt --model_variant 7B > logs/cleanEval/emo_mpt7B.log 2>&1
python -u scripts/experiments/icleval.py --task_name glue-sst2 --dataset glue-sst2 --model_type mpt --model_variant 7B > logs/cleanEval/sst2_mpt7B.log 2>&1
python -u scripts/experiments/icleval.py --task_name poem_sentiment --dataset poem_sentiment --model_type mpt --model_variant 7B > logs/cleanEval/poem_mpt7B.log 2>&1
python -u scripts/experiments/icleval.py --task_name ag_news --dataset ag_news --model_type mpt --model_variant 7B > logs/cleanEval/ag_mpt7B.log 2>&1
### gpt-j-6B
python -u scripts/experiments/icleval.py --task_name glue-cola --dataset glue-cola --model_type gpt-j --model_variant 6B > logs/cleanEval/cola_gpt-j6B.log 2>&1
python -u scripts/experiments/icleval.py --task_name emo --dataset emo --model_type gpt-j --model_variant 6B > logs/cleanEval/emo_gpt-j6B.log 2>&1
python -u scripts/experiments/icleval.py --task_name glue-sst2 --dataset glue-sst2 --model_type gpt-j --model_variant 6B > logs/cleanEval/sst2_gpt-j6B.log 2>&1
python -u scripts/experiments/icleval.py --task_name poem_sentiment --dataset poem_sentiment --model_type gpt-j --model_variant 6B > logs/cleanEval/poem_gpt-j6B.log 2>&1
python -u scripts/experiments/icleval.py --task_name ag_news --dataset ag_news --model_type gpt-j --model_variant 6B > logs/cleanEval/ag_gpt-j6B.log 2>&1
### vicuna-7B
python -u scripts/experiments/icleval.py --task_name glue-cola --dataset glue-cola --model_type vicuna --model_variant 7B > logs/cleanEval/cola_vicuna7B.log 2>&1
python -u scripts/experiments/icleval.py --task_name emo --dataset emo --model_type vicuna --model_variant 7B > logs/cleanEval/emo_vicuna7B.log 2>&1
python -u scripts/experiments/icleval.py --task_name glue-sst2 --dataset glue-sst2 --model_type vicuna --model_variant 7B > logs/cleanEval/sst2_vicuna7B.log 2>&1
python -u scripts/experiments/icleval.py --task_name poem_sentiment --dataset poem_sentiment --model_type vicuna --model_variant 7B > logs/cleanEval/poem_vicuna7B.log 2>&1
python -u scripts/experiments/icleval.py --task_name ag_news --dataset ag_news --model_type vicuna --model_variant 7B > logs/cleanEval/ag_vicuna7B.log 2>&1


# random flip
### pythia-2.8B
python -u scripts/experiments/iclpoison_flip.py --task_name glue-cola --dataset glue-cola --model_type pythia --model_variant 2.8B > logs/iclflip/cola_pythia2.8B.log 2>&1
python -u scripts/experiments/iclpoison_flip.py --task_name emo --dataset emo --model_type pythia --model_variant 2.8B > logs/iclflip/emo_pythia2.8B.log 2>&1
python -u scripts/experiments/iclpoison_flip.py --task_name glue-sst2 --dataset glue-sst2 --model_type pythia --model_variant 2.8B > logs/iclflip/sst2_pythia2.8B.log 2>&1
python -u scripts/experiments/iclpoison_flip.py --task_name poem_sentiment --dataset poem_sentiment --model_type pythia --model_variant 2.8B > logs/iclflip/poem_pythia2.8B.log 2>&1
python -u scripts/experiments/iclpoison_flip.py --task_name ag_news --dataset ag_news --model_type pythia --model_variant 2.8B > logs/iclflip/ag_pythia2.8B.log 2>&1
### pythia-6.9B
python -u scripts/experiments/iclpoison_flip.py --task_name glue-cola --dataset glue-cola --model_type pythia --model_variant 6.9B > logs/iclflip/cola_pythia6.9B.log 2>&1
python -u scripts/experiments/iclpoison_flip.py --task_name emo --dataset emo --model_type pythia --model_variant 6.9B > logs/iclflip/emo_pythia6.9B.log 2>&1
python -u scripts/experiments/iclpoison_flip.py --task_name glue-sst2 --dataset glue-sst2 --model_type pythia --model_variant 6.9B > logs/iclflip/sst2_pythia6.9B.log 2>&1
python -u scripts/experiments/iclpoison_flip.py --task_name poem_sentiment --dataset poem_sentiment --model_type pythia --model_variant 6.9B > logs/iclflip/poem_pythia6.9B.log 2>&1
python -u scripts/experiments/iclpoison_flip.py --task_name ag_news --dataset ag_news --model_type pythia --model_variant 6.9B > logs/iclflip/ag_pythia6.9B.log 2>&1
### llama2-7B
python -u scripts/experiments/iclpoison_flip.py --task_name glue-cola --dataset glue-cola --model_type llama --model_variant 7B > logs/iclflip/cola_llama7B.log 2>&1
python -u scripts/experiments/iclpoison_flip.py --task_name emo --dataset emo --model_type llama --model_variant 7B > logs/iclflip/emo_llama7B.log 2>&1
python -u scripts/experiments/iclpoison_flip.py --task_name glue-sst2 --dataset glue-sst2 --model_type llama --model_variant 7B > logs/iclflip/sst2_llama7B.log 2>&1
python -u scripts/experiments/iclpoison_flip.py --task_name poem_sentiment --dataset poem_sentiment --model_type llama --model_variant 7B > logs/iclflip/poem_llama7B.log 2>&1
python -u scripts/experiments/iclpoison_flip.py --task_name ag_news --dataset ag_news --model_type llama --model_variant 7B > logs/iclflip/ag_llama7B.log 2>&1
### falcon-7B
python -u scripts/experiments/iclpoison_flip.py --task_name glue-cola --dataset glue-cola --model_type falcon --model_variant 7B > logs/iclflip/cola_falcon7B.log 2>&1
python -u scripts/experiments/iclpoison_flip.py --task_name emo --dataset emo --model_type falcon --model_variant 7B > logs/iclflip/emo_falcon7B.log 2>&1
python -u scripts/experiments/iclpoison_flip.py --task_name glue-sst2 --dataset glue-sst2 --model_type falcon --model_variant 7B > logs/iclflip/sst2_falcon7B.log 2>&1
python -u scripts/experiments/iclpoison_flip.py --task_name poem_sentiment --dataset poem_sentiment --model_type falcon --model_variant 7B > logs/iclflip/poem_falcon7B.log 2>&1
python -u scripts/experiments/iclpoison_flip.py --task_name ag_news --dataset ag_news --model_type falcon --model_variant 7B > logs/iclflip/ag_falcon7B.log 2>&1
### mpt-7B
python -u scripts/experiments/iclpoison_flip.py --task_name glue-cola --dataset glue-cola --model_type mpt --model_variant 7B > logs/iclflip/cola_mpt7B.log 2>&1
python -u scripts/experiments/iclpoison_flip.py --task_name emo --dataset emo --model_type mpt --model_variant 7B > logs/iclflip/emo_mpt7B.log 2>&1
python -u scripts/experiments/iclpoison_flip.py --task_name glue-sst2 --dataset glue-sst2 --model_type mpt --model_variant 7B > logs/iclflip/sst2_mpt7B.log 2>&1
python -u scripts/experiments/iclpoison_flip.py --task_name poem_sentiment --dataset poem_sentiment --model_type mpt --model_variant 7B > logs/iclflip/poem_mpt7B.log 2>&1
python -u scripts/experiments/iclpoison_flip.py --task_name ag_news --dataset ag_news --model_type mpt --model_variant 7B > logs/iclflip/ag_mpt7B.log 2>&1
### gpt-j6B
python -u scripts/experiments/iclpoison_flip.py --task_name glue-cola --dataset glue-cola --model_type gpt-j --model_variant 6B > logs/iclflip/cola_gpt-j6B.log 2>&1
python -u scripts/experiments/iclpoison_flip.py --task_name emo --dataset emo --model_type gpt-j --model_variant 6B > logs/iclflip/emo_gpt-j6B.log 2>&1
python -u scripts/experiments/iclpoison_flip.py --task_name glue-sst2 --dataset glue-sst2 --model_type gpt-j --model_variant 6B > logs/iclflip/sst2_gpt-j6B.log 2>&1
python -u scripts/experiments/iclpoison_flip.py --task_name poem_sentiment --dataset poem_sentiment --model_type gpt-j --model_variant 6B > logs/iclflip/poem_gpt-j6B.log 2>&1
python -u scripts/experiments/iclpoison_flip.py --task_name ag_news --dataset ag_news --model_type gpt-j --model_variant 6B > logs/iclflip/ag_gpt-j6B.log 2>&1
### vicuna-7B
python -u scripts/experiments/iclpoison_flip.py --task_name glue-cola --dataset glue-cola --model_type vicuna --model_variant 7B > logs/iclflip/cola_vicuna7B.log 2>&1
python -u scripts/experiments/iclpoison_flip.py --task_name emo --dataset emo --model_type vicuna --model_variant 7B > logs/iclflip/emo_vicuna7B.log 2>&1
python -u scripts/experiments/iclpoison_flip.py --task_name glue-sst2 --dataset glue-sst2 --model_type vicuna --model_variant 7B > logs/iclflip/sst2_vicuna7B.log 2>&1
python -u scripts/experiments/iclpoison_flip.py --task_name poem_sentiment --dataset poem_sentiment --model_type vicuna --model_variant 7B > logs/iclflip/poem_vicuna7B.log 2>&1
python -u scripts/experiments/iclpoison_flip.py --task_name ag_news --dataset ag_news --model_type vicuna --model_variant 7B > logs/iclflip/ag_vicuna7B.log 2>&1


# poison with token exchange
### pythia-2.8B
python -u scripts/experiments/iclpoison_token.py --task_name poem_sentiment --dataset poem_sentiment --model_type pythia --model_variant 2.8B --budget 3 --num_cand 100 > logs/iclpoison_token/poem_b3_pythia2.8B.log 2>&1
python -u scripts/experiments/iclpoison_token.py --task_name glue-cola --dataset glue-cola --model_type pythia --model_variant 2.8B --budget 3 --num_cand 100 > logs/iclpoison_token/cola_b3_pythia2.8B.log 2>&1
python -u scripts/experiments/iclpoison_token.py --task_name glue-sst2 --dataset glue-sst2 --model_type pythia --model_variant 2.8B --budget 3 --num_cand 100 > logs/iclpoison_token/sst2_b3_pythia2.8B.log 2>&1
python -u scripts/experiments/iclpoison_token.py --task_name emo --dataset emo --model_type pythia --model_variant 2.8B --budget 3 --num_cand 100 > logs/iclpoison_token/emo_b3_pythia2.8B.log 2>&1
python -u scripts/experiments/iclpoison_token.py --task_name  ag_news --dataset ag_news --model_type pythia --model_variant 2.8B --budget 3 --num_cand 100 > logs/iclpoison_token/ag_b3_pythia2.8B.log 2>&1
### pythia-6.9B
python -u scripts/experiments/iclpoison_token.py --task_name poem_sentiment --dataset poem_sentiment --model_type pythia --model_variant 6.9B --budget 3 --num_cand 100 > logs/iclpoison_token/poem_b3_pythia6.9B.log 2>&1
python -u scripts/experiments/iclpoison_token.py --task_name glue-cola --dataset glue-cola --model_type pythia --model_variant 6.9B --budget 3 --num_cand 100 > logs/iclpoison_token/cola_b3_pythia6.9B.log 2>&1
python -u scripts/experiments/iclpoison_token.py --task_name glue-sst2 --dataset glue-sst2 --model_type pythia --model_variant 6.9B --budget 3 --num_cand 100 > logs/iclpoison_token/sst2_b3_pythia6.9B.log 2>&1
python -u scripts/experiments/iclpoison_token.py --task_name emo --dataset emo --model_type pythia --model_variant 6.9B --budget 3 --num_cand 100 > logs/iclpoison_token/emo_b3_pythia6.9B.log 2>&1
python -u scripts/experiments/iclpoison_token.py --task_name  ag_news --dataset ag_news --model_type pythia --model_variant 6.9B --budget 3 --num_cand 100 > logs/iclpoison_token/ag_b3_pythia6.9B.log 2>&1
### llama-7B
python -u scripts/experiments/iclpoison_token.py --task_name poem_sentiment --dataset poem_sentiment --model_type llama --model_variant 7B --budget 3 --num_cand 100 > logs/iclpoison_token/poem_b3_llama7B.log 2>&1
python -u scripts/experiments/iclpoison_token.py --task_name glue-cola --dataset glue-cola --model_type llama --model_variant 7B --budget 3 --num_cand 100 > logs/iclpoison_token/cola_b3_llama7B.log 2>&1
python -u scripts/experiments/iclpoison_token.py --task_name glue-sst2 --dataset glue-sst2 --model_type llama --model_variant 7B --budget 3 --num_cand 100 > logs/iclpoison_token/sst2_b3_llama7B.log 2>&1
python -u scripts/experiments/iclpoison_token.py --task_name emo --dataset emo --model_type llama --model_variant 7B --budget 3 --num_cand 100 > logs/iclpoison_token/emo_b3_llama7B.log 2>&1
python -u scripts/experiments/iclpoison_token.py --task_name  ag_news --dataset ag_news --model_type llama --model_variant 7B --budget 3 --num_cand 100 > logs/iclpoison_token/ag_b3_llama7B.log 2>&1
### mpt-7B
python -u scripts/experiments/iclpoison_token.py --task_name poem_sentiment --dataset poem_sentiment --model_type mpt --model_variant 7B --budget 3 --num_cand 100 > logs/iclpoison_token/poem_b3_mpt7B.log 2>&1
python -u scripts/experiments/iclpoison_token.py --task_name glue-cola --dataset glue-cola --model_type mpt --model_variant 7B --budget 3 --num_cand 100 > logs/iclpoison_token/cola_b3_mpt7B.log 2>&1
python -u scripts/experiments/iclpoison_token.py --task_name glue-sst2 --dataset glue-sst2 --model_type mpt --model_variant 7B --budget 3 --num_cand 100 > logs/iclpoison_token/sst2_b3_mpt7B.log 2>&1
python -u scripts/experiments/iclpoison_token.py --task_name emo --dataset emo --model_type mpt --model_variant 7B --budget 3 --num_cand 100 > logs/iclpoison_token/emo_b3_mpt7B.log 2>&1
python -u scripts/experiments/iclpoison_token.py --task_name  ag_news --dataset ag_news --model_type mpt --model_variant 7B --budget 3 --num_cand 100 > logs/iclpoison_token/ag_b3_mpt7B.log 2>&1
### falcon-7B
python -u scripts/experiments/iclpoison_token.py --task_name poem_sentiment --dataset poem_sentiment --model_type falcon --model_variant 7B --budget 3 --num_cand 100 > logs/iclpoison_token/poem_b3_falcon7B.log 2>&1
python -u scripts/experiments/iclpoison_token.py --task_name glue-cola --dataset glue-cola --model_type falcon --model_variant 7B --budget 3 --num_cand 100 > logs/iclpoison_token/cola_b3_falcon7B.log 2>&1
python -u scripts/experiments/iclpoison_token.py --task_name glue-sst2 --dataset glue-sst2 --model_type falcon --model_variant 7B --budget 3 --num_cand 100 > logs/iclpoison_token/sst2_b3_falcon7B.log 2>&1
python -u scripts/experiments/iclpoison_token.py --task_name emo --dataset emo --model_type falcon --model_variant 7B --budget 3 --num_cand 100 > logs/iclpoison_token/emo_b3_falcon7B.log 2>&1
python -u scripts/experiments/iclpoison_token.py --task_name  ag_news --dataset ag_news --model_type falcon --model_variant 7B --budget 3 --num_cand 100 > logs/iclpoison_token/ag_b3_falcon7B.log 2>&1
### gpt-j-6B
python -u scripts/experiments/iclpoison_token.py --task_name poem_sentiment --dataset poem_sentiment --model_type gpt-j --model_variant 6B --budget 3 --num_cand 100 > logs/iclpoison_token/poem_b3_gpt-j6B.log 2>&1
python -u scripts/experiments/iclpoison_token.py --task_name glue-cola --dataset glue-cola --model_type gpt-j --model_variant 6B --budget 3 --num_cand 100 > logs/iclpoison_token/cola_b3_gpt-j6B.log 2>&1
python -u scripts/experiments/iclpoison_token.py --task_name glue-sst2 --dataset glue-sst2 --model_type gpt-j --model_variant 6B --budget 3 --num_cand 100 > logs/iclpoison_token/sst2_b3_gpt-j6B.log 2>&1
python -u scripts/experiments/iclpoison_token.py --task_name emo --dataset emo --model_type gpt-j --model_variant 6B --budget 3 --num_cand 100 > logs/iclpoison_token/emo_b3_lgpt-j6B.log 2>&1
python -u scripts/experiments/iclpoison_token.py --task_name  ag_news --dataset ag_news --model_type gpt-j --model_variant 6B --budget 3 --num_cand 100 > logs/iclpoison_token/ag_b3_gpt-j6B.log 2>&1
### vicuna-7B
python -u scripts/experiments/iclpoison_token.py --task_name poem_sentiment --dataset poem_sentiment --model_type vicuna --model_variant 7B --budget 3 --num_cand 100 > logs/iclpoison_token/poem_b3_vicuna7B.log 2>&1
python -u scripts/experiments/iclpoison_token.py --task_name glue-cola --dataset glue-cola --model_type vicuna --model_variant 7B --budget 3 --num_cand 100 > logs/iclpoison_token/cola_b3_vicuna7B.log 2>&1
python -u scripts/experiments/iclpoison_token.py --task_name glue-sst2 --dataset glue-sst2 --model_type vicuna --model_variant 7B --budget 3 --num_cand 100 > logs/iclpoison_token/sst2_b3_vicuna7B.log 2>&1
python -u scripts/experiments/iclpoison_token.py --task_name emo --dataset emo --model_type vicuna --model_variant 7B --budget 3 --num_cand 100 > logs/iclpoison_token/emo_b3_vicuna7B.log 2>&1
python -u scripts/experiments/iclpoison_token.py --task_name  ag_news --dataset ag_news --model_type vicuna --model_variant 7B --budget 3 --num_cand 100 > logs/iclpoison_token/ag_b3_vicuna7B.log 2>&1

#poison with word change
### llama-7B
python -u scripts/experiments/iclpoison_word.py --task_name poem_sentiment --dataset poem_sentiment --model_type llama --model_variant 7B --budget 3 --num_cand 50 > logs/iclpoison_word/poem_b3_llama7B.log 2>&1
python -u scripts/experiments/iclpoison_word.py --task_name glue-cola --dataset glue-cola --model_type llama --model_variant 7B --budget 3 --num_cand 50 > logs/iclpoison_word/cola_b3_llama7B.log 2>&1
python -u scripts/experiments/iclpoison_word.py --task_name glue-sst2 --dataset glue-sst2 --model_type llama --model_variant 7B --budget 3 --num_cand 50 > logs/iclpoison_word/sst2_b3_llama7B.log 2>&1
python -u scripts/experiments/iclpoison_word.py --task_name emo --dataset emo --model_type llama --model_variant 7B --budget 3 --num_cand 50 > logs/iclpoison_word/emo_b3_llama7B.log 2>&1
python -u scripts/experiments/iclpoison_word.py --task_name  ag_news --dataset ag_news --model_type llama --model_variant 7B --budget 3 --num_cand 50 > logs/iclpoison_word/ag_b3_llama7B.log 2>&1


#poison with word change, min loss 
### llama-7B
python -u scripts/experiments/iclpoison_word_min.py --task_name poem_sentiment --dataset poem_sentiment --model_type llama --model_variant 7B --budget 3 --num_cand 100 > logs/iclpoison_word_min/poem_b3_llama7B.log 2>&1
python -u scripts/experiments/iclpoison_word_min.py --task_name glue-cola --dataset glue-cola --model_type llama --model_variant 7B --budget 3 --num_cand 100 > logs/iclpoison_word_min/cola_b3_llama7B.log 2>&1
python -u scripts/experiments/iclpoison_word_min.py --task_name glue-sst2 --dataset glue-sst2 --model_type llama --model_variant 7B --budget 3 --num_cand 100 > logs/iclpoison_word_min/sst2_b3_llama7B.log 2>&1
python -u scripts/experiments/iclpoison_word_min.py --task_name emo --dataset emo --model_type llama --model_variant 7B --budget 3 --num_cand 100 > logs/iclpoison_word_min/emo_b3_llama7B.log 2>&1
python -u scripts/experiments/iclpoison_word_min.py --task_name  ag_news --dataset ag_news --model_type llama --model_variant 7B --budget 3 --num_cand 100 > logs/iclpoison_word_min/ag_b3_llama7B.log 2>&1


#poison with char change
### llama-7B
python -u scripts/experiments/iclpoison_char.py --task_name poem_sentiment --dataset poem_sentiment --model_type llama --model_variant 7B --budget 5 > logs/iclpoison_char/poem_b5_llama7B.log 2>&1
python -u scripts/experiments/iclpoison_char.py --task_name glue-cola --dataset glue-cola --model_type llama --model_variant 7B --budget 5 > logs/iclpoison_char/cola_b5_llama7B.log 2>&1
python -u scripts/experiments/iclpoison_char.py --task_name glue-sst2 --dataset glue-sst2 --model_type llama --model_variant 7B --budget 5 > logs/iclpoison_char/sst2_b5_llama7B.log 2>&1
python -u scripts/experiments/iclpoison_char.py --task_name emo --dataset emo --model_type llama --model_variant 7B --budget 5 > logs/iclpoison_char/emo_b5_llama7B.log 2>&1
python -u scripts/experiments/iclpoison_char.py --task_name  ag_news --dataset ag_news --model_type llama --model_variant 7B --budget 5 > logs/iclpoison_char/ag_b5_llama7B.log 2>&1

#poison with char change, min loss
### llama-7B
python -u scripts/experiments/iclpoison_char_min.py --task_name poem_sentiment --dataset poem_sentiment --model_type llama --model_variant 7B --budget 5 > logs/iclpoison_char_min/poem_b5_llama7B.log 2>&1
python -u scripts/experiments/iclpoison_char_min.py --task_name glue-cola --dataset glue-cola --model_type llama --model_variant 7B --budget 5 > logs/iclpoison_char_min/cola_b5_llama7B.log 2>&1
python -u scripts/experiments/iclpoison_char_min.py --task_name glue-sst2 --dataset glue-sst2 --model_type llama --model_variant 7B --budget 5 > logs/iclpoison_char_min/sst2_b5_llama7B.log 2>&1
python -u scripts/experiments/iclpoison_char_min.py --task_name emo --dataset emo --model_type llama --model_variant 7B --budget 5 > logs/iclpoison_char_min/emo_b5_llama7B.log 2>&1
python -u scripts/experiments/iclpoison_char_min.py --task_name  ag_news --dataset ag_news --model_type llama --model_variant 7B --budget 5 > logs/iclpoison_char_min/ag_b5_llama7B.log 2>&1



# poison with adv tokens, all layer
### pythia-2.8B
python -u scripts/experiments/iclpoison_all.py --task_name poem_sentiment --dataset poem_sentiment --model_type pythia --model_variant 2.8B > logs/icvpoison/token2_all_poem_pythia2.8B.log 2>&1
python -u scripts/experiments/iclpoison_all.py --task_name glue-cola --dataset glue-cola --model_type pythia --model_variant 2.8B > logs/icvpoison/token2_all_cola_pythia2.8B.log 2>&1
python -u scripts/experiments/iclpoison_all.py --task_name glue-sst2 --dataset glue-sst2 --model_type pythia --model_variant 2.8B > logs/icvpoison/token2_all_sst2_pythia2.8B.log 2>&1
python -u scripts/experiments/iclpoison_all.py --task_name emo --dataset emo --model_type pythia --model_variant 2.8B > logs/icvpoison/token2_all_emo_pythia2.8B.log 2>&1
python -u scripts/experiments/iclpoison_all.py --task_name  ag_news --dataset ag_news --model_type pythia --model_variant 2.8B > logs/icvpoison/token2_all_ag_pythia2.8B.log 2>&1
### pythia-6.9B
python -u scripts/experiments/iclpoison_all.py --task_name poem_sentiment --dataset poem_sentiment --model_type pythia --model_variant 6.9B > logs/icvpoison/token2_all_poem_pythia6.9B.log 2>&1
python -u scripts/experiments/iclpoison_all.py --task_name glue-cola --dataset glue-cola --model_type pythia --model_variant 6.9B > logs/icvpoison/token2_all_cola_pythia6.9B.log 2>&1
python -u scripts/experiments/iclpoison_all.py --task_name glue-sst2 --dataset glue-sst2 --model_type pythia --model_variant 6.9B > logs/icvpoison/token2_all_sst2_pythia6.9B.log 2>&1
python -u scripts/experiments/iclpoison_all.py --task_name emo --dataset emo --model_type pythia --model_variant 6.9B > logs/icvpoison/token2_all_emo_pythia6.9B.log 2>&1
python -u scripts/experiments/iclpoison_all.py --task_name ag_news --dataset ag_news --model_type pythia --model_variant 6.9B > logs/icvpoison/token2_all_ag_pythia6.9B.log 2>&1
###

# poison with adv tokens, min loss
### pythia-2.8B
python -u scripts/experiments/iclpoison_all_min.py --task_name poem_sentiment --dataset poem_sentiment --model_type pythia --model_variant 2.8B > logs/icvpoison_min/poem_pythia2.8B.log 2>&1
python -u scripts/experiments/iclpoison_all_min.py --task_name glue-cola --dataset glue-cola --model_type pythia --model_variant 2.8B > logs/icvpoison_min/cola_pythia2.8B.log 2>&1
python -u scripts/experiments/iclpoison_all_min.py --task_name glue-sst2 --dataset glue-sst2 --model_type pythia --model_variant 2.8B > logs/icvpoison_min/sst2_pythia2.8B.log 2>&1
python -u scripts/experiments/iclpoison_all_min.py --task_name emo --dataset emo --model_type pythia --model_variant 2.8B > logs/icvpoison_min/emo_pythia2.8B.log 2>&1
python -u scripts/experiments/iclpoison_all_min.py --task_name  ag_news --dataset ag_news --model_type pythia --model_variant 2.8B > logs/icvpoison_min/ag_pythia2.8B.log 2>&1
### pythia-6.9B
python -u scripts/experiments/iclpoison_all_min.py --task_name poem_sentiment --dataset poem_sentiment --model_type pythia --model_variant 6.9B > logs/icvpoison_min/poem_pythia6.9B.log 2>&1
python -u scripts/experiments/iclpoison_all_min.py --task_name glue-cola --dataset glue-cola --model_type pythia --model_variant 6.9B > logs/icvpoison_min/cola_pythia6.9B.log 2>&1
python -u scripts/experiments/iclpoison_all_min.py --task_name glue-sst2 --dataset glue-sst2 --model_type pythia --model_variant 6.9B > logs/icvpoison_min/sst2_pythia6.9B.log 2>&1
python -u scripts/experiments/iclpoison_all_min.py --task_name emo --dataset emo --model_type pythia --model_variant 6.9B > logs/icvpoison_min/emo_pythia6.9B.log 2>&1
python -u scripts/experiments/iclpoison_all_min.py --task_name  ag_news --dataset ag_news --model_type pythia --model_variant 6.9B > logs/icvpoison_min/ag_pythia6.9B.log 2>&1
### llama-7B
python -u scripts/experiments/iclpoison_all_min.py --task_name poem_sentiment --dataset poem_sentiment --model_type llama --model_variant 7B > logs/icvpoison_min/poem_llama7B.log 2>&1
python -u scripts/experiments/iclpoison_all_min.py --task_name glue-cola --dataset glue-cola --model_type llama --model_variant 7B > logs/icvpoison_min/cola_llama7B.log 2>&1
python -u scripts/experiments/iclpoison_all_min.py --task_name glue-sst2 --dataset glue-sst2 --model_type llama --model_variant 7B > logs/icvpoison_min/sst2_llama7B.log 2>&1
python -u scripts/experiments/iclpoison_all_min.py --task_name emo --dataset emo --model_type llama --model_variant 7B > logs/icvpoison_min/emo_llama7B.log 2>&1
python -u scripts/experiments/iclpoison_all_min.py --task_name  ag_news --dataset ag_news --model_type llama --model_variant 7B > logs/icvpoison_min/ag_llama7B.log 2>&1



# clean eval on GPT
### gpt-3.5
python -u scripts/experiments/icleval_gpt.py --task_name glue-cola --dataset glue-cola --model gpt3 > logs/cleanEval/cola_gpt3.log 2>&1
python -u scripts/experiments/icleval_gpt.py --task_name glue-sst2 --dataset glue-sst2 --model gpt3 > logs/cleanEval/sst2_gpt3.log 2>&1
python -u scripts/experiments/icleval_gpt.py --task_name emo --dataset emo --model gpt3 > logs/cleanEval/emo_gpt3.log 2>&1
python -u scripts/experiments/icleval_gpt.py --task_name poem_sentiment --dataset poem_sentiment --model gpt3 > logs/cleanEval/poem_gpt3.log 2>&1
python -u scripts/experiments/icleval_gpt.py --task_name ag_news --dataset ag_news --model gpt3 > logs/cleanEval/ag_gpt3.log 2>&1  # beyond rate limit
### gpt-4
python -u scripts/experiments/icleval_gpt.py --task_name glue-cola --dataset glue-cola --model gpt4 > logs/cleanEval/cola_gpt4.log 2>&1
python -u scripts/experiments/icleval_gpt.py --task_name glue-sst2 --dataset glue-sst2 --model gpt4 > logs/cleanEval/sst2_gpt4.log 2>&1
python -u scripts/experiments/icleval_gpt.py --task_name emo --dataset emo --model gpt4 > logs/cleanEval/emo_gpt4.log 2>&1
python -u scripts/experiments/icleval_gpt.py --task_name poem_sentiment --dataset poem_sentiment --model gpt4 > logs/cleanEval/poem_gpt4.log 2>&1
python -u scripts/experiments/icleval_gpt.py --task_name ag_news--dataset ag_news --model gpt4 > logs/cleanEval/ag_gpt4.log 2>&1

# label flip on GPT
### gpt-3.5
python -u scripts/experiments/iclpoison_flip_gpt.py --task_name glue-cola --dataset glue-cola --model gpt3 > logs/iclflip/cola_gpt3.log 2>&1
python -u scripts/experiments/iclpoison_flip_gpt.py --task_name glue-sst2 --dataset glue-sst2 --model gpt3 > logs/iclflip/sst2_gpt3.log 2>&1
python -u scripts/experiments/iclpoison_flip_gpt.py --task_name emo --dataset emo --model gpt3 > logs/iclflip/emo_gpt3.log 2>&1
python -u scripts/experiments/iclpoison_flip_gpt.py --task_name poem_sentiment --dataset poem_sentiment --model gpt3 > logs/iclflip/poem_gpt3.log 2>&1
python -u scripts/experiments/iclpoison_flip_gpt.py --task_name ag_news --dataset ag_news --model gpt3 > logs/iclflip/ag_gpt3.log 2>&1  # beyond rate limit
### gpt-4
python -u scripts/experiments/iclpoison_flip_gpt.py --task_name glue-cola --dataset glue-cola --model gpt4 > logs/iclflip/cola_gpt4.log 2>&1
python -u scripts/experiments/iclpoison_flip_gpt.py --task_name glue-sst2 --dataset glue-sst2 --model gpt4 > logs/iclflip/sst2_gpt4.log 2>&1
python -u scripts/experiments/iclpoison_flip_gpt.py --task_name emo --dataset emo --model gpt4 > logs/iclflip/emo_gpt4.log 2>&1
python -u scripts/experiments/iclpoison_flip_gpt.py --task_name poem_sentiment --dataset poem_sentiment --model gpt4 > logs/iclflip/poem_gpt4.log 2>&1
python -u scripts/experiments/iclpoison_flip_gpt.py --task_name ag_news--dataset ag_news --model gpt4 > logs/iclflip/ag_gpt4.log 2>&1



# poison script, best layer
### pythia-2.8B
python -u scripts/experiments/iclpoison_best.py --task_name poem_sentiment --dataset poem_sentiment --model_type pythia --model_variant 2.8B > logs/icvpoison/token2_best_poem_pythia2.8B.log 2>&1
python -u scripts/experiments/iclpoison_best.py --task_name glue-cola --dataset glue-cola --model_type pythia --model_variant 2.8B > logs/icvpoison/token2_best_cola_pythia2.8B.log 2>&1
python -u scripts/experiments/iclpoison_best.py --task_name glue-sst2 --dataset glue-sst2 --model_type pythia --model_variant 2.8B > logs/icvpoison/token2_best_sst2_pythia2.8B.log 2>&1
python -u scripts/experiments/iclpoison_best.py --task_name emo --dataset emo --model_type pythia --model_variant 2.8B > logs/icvpoison/token2_best_emo_pythia2.8B.log 2>&1
python -u scripts/experiments/iclpoison_best.py --task_name  --dataset ag_news --model_type pythia --model_variant 2.8B > logs/icvpoison/token2_best_ag_pythia2.8B.log 2>&1
### pythia-6.9B
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