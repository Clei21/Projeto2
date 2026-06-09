import argparse
import json
import os

from scripts.config import RESULTS_DIR


def load_result(prefix: str, run_name: str):
    path = os.path.join(RESULTS_DIR, f"{prefix}_{run_name}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def pct_change(base: float, new: float) -> float:
    if base == 0:
        return 0.0
    return (new - base) / base * 100.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_run", default="baseline")
    parser.add_argument("--finetuned_runs", nargs="+", required=True)
    args = parser.parse_args()

    base_sql = load_result("text2sql", args.baseline_run)
    base_mmlu = load_result("mmlu", args.baseline_run)

    if base_sql is None:
        raise FileNotFoundError(
            f"text2sql_{args.baseline_run}.json not found"
            " — run evaluate_text2sql.py first"
        )
    if base_mmlu is None:
        raise FileNotFoundError(
            f"mmlu_{args.baseline_run}.json not found"
            " — run evaluate_mmlu.py first"
        )

    summary = {"baseline": {}, "finetuned": {}}
    bl = summary["baseline"]
    bl["execution_accuracy"] = base_sql["execution_accuracy"]
    bl["mmlu_overall"] = base_mmlu["overall_accuracy"]
    bl["mmlu_per_category"] = base_mmlu["per_category_accuracy"]

    for run in args.finetuned_runs:
        ft_sql = load_result("text2sql", run)
        ft_mmlu = load_result("mmlu", run)
        if ft_sql is None or ft_mmlu is None:
            print(f"[skip] results missing for run '{run}', skipping")
            continue

        entry = {
            "execution_accuracy": ft_sql["execution_accuracy"],
            "execution_accuracy_delta_pct": pct_change(
                base_sql["execution_accuracy"], ft_sql["execution_accuracy"]
            ),
            "mmlu_overall": ft_mmlu["overall_accuracy"],
            "mmlu_overall_delta_pct": pct_change(
                base_mmlu["overall_accuracy"], ft_mmlu["overall_accuracy"]
            ),
            "mmlu_per_category": {},
        }
        for category, base_acc in base_mmlu["per_category_accuracy"].items():
            ft_acc = ft_mmlu["per_category_accuracy"][category]
            entry["mmlu_per_category"][category] = {
                "baseline": base_acc,
                "finetuned": ft_acc,
                "delta_pct": pct_change(base_acc, ft_acc),
            }
        summary["finetuned"][run] = entry

    out_path = os.path.join(RESULTS_DIR, "summary.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    base = summary["baseline"]
    print(f"Baseline Execution Accuracy: {base['execution_accuracy']:.4f}")
    print(f"Baseline MMLU Overall:       {base['mmlu_overall']:.4f}")
    print()
    for run, entry in summary["finetuned"].items():
        print(f"== {run} ==")
        print(
            f"  Execution Accuracy: {entry['execution_accuracy']:.4f} "
            f"({entry['execution_accuracy_delta_pct']:+.2f}%)"
        )
        print(
            f"  MMLU Overall: {entry['mmlu_overall']:.4f} "
            f"({entry['mmlu_overall_delta_pct']:+.2f}%)"
        )
        for category, values in entry["mmlu_per_category"].items():
            print(
                f"    {category}: {values['baseline']:.4f} -> "
                f"{values['finetuned']:.4f} ({values['delta_pct']:+.2f}%)"
            )
    print(f"\nSaved summary to {out_path}")


if __name__ == "__main__":
    main()
