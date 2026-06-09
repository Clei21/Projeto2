set -e
python scripts/prepare_spider.py
python scripts/build_dataset.py --max_train_samples 1000
python scripts/evaluate_spider.py --name baseline --limit 200
python scripts/evaluate_mmlu.py --name baseline
python scripts/train_lora.py --run_name lora_lr2e-4_ep1 --learning_rate 0.0002 --num_train_epochs 1
python scripts/train_lora.py --run_name lora_lr1e-4_ep1 --learning_rate 0.0001 --num_train_epochs 1
python scripts/evaluate_spider.py --name lora_lr2e-4_ep1 --adapter outputs/lora_lr2e-4_ep1 --limit 200
python scripts/evaluate_spider.py --name lora_lr1e-4_ep1 --adapter outputs/lora_lr1e-4_ep1 --limit 200
python scripts/evaluate_mmlu.py --name lora_lr2e-4_ep1 --adapter outputs/lora_lr2e-4_ep1
python scripts/evaluate_mmlu.py --name lora_lr1e-4_ep1 --adapter outputs/lora_lr1e-4_ep1
python scripts/summarize_results.py
