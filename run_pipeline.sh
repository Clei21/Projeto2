set -e

export PYTHONPATH=.
export BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}
export SEED=${SEED:-42}

echo "[1/8] Preprocessing Spider training split"
python scripts/preprocess_spider.py

echo "[2/8] Baseline Text-to-SQL evaluation"
python scripts/evaluate_text2sql.py --run_name baseline

echo "[3/8] Baseline MMLU evaluation"
python scripts/evaluate_mmlu.py --run_name baseline

echo "[4/8] Fine-tuning config A (1 epoch, lr=2e-4)"
python scripts/train.py --config configs/config_a.yaml --output_dir adapters/config_a

echo "[4/8] Fine-tuning config B (3 epochs, lr=5e-5)"
python scripts/train.py --config configs/config_b.yaml --output_dir adapters/config_b

echo "[5/8] Fine-tuned Text-to-SQL evaluation (A and B)"
python scripts/evaluate_text2sql.py --adapter_path adapters/config_a --run_name finetuned_a
python scripts/evaluate_text2sql.py --adapter_path adapters/config_b --run_name finetuned_b

echo "[6/8] Fine-tuned MMLU evaluation (A and B)"
python scripts/evaluate_mmlu.py --adapter_path adapters/config_a --run_name finetuned_a
python scripts/evaluate_mmlu.py --adapter_path adapters/config_b --run_name finetuned_b

echo "[7/8] Pytest evaluation gate (Fase 4.1)"
EVAL_RUN_NAME=finetuned_a python -m pytest scripts/test_evaluation.py -v

echo "[8/8] Regression analysis and error analysis"
python scripts/analyze_regression.py --baseline_run baseline --finetuned_runs finetuned_a finetuned_b
python scripts/error_analysis.py --run_name finetuned_a

echo "Pipeline complete. See results/summary.json"
