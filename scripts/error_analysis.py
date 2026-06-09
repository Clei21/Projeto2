import argparse
import json
import os

from scripts.config import RESULTS_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--n", type=int, default=3)
    args = parser.parse_args()

    path = os.path.join(RESULTS_DIR, f"text2sql_{args.run_name}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — run evaluate_text2sql.py first"
        )

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    failures = [r for r in data["records"] if r["score"] == 0.0]
    selected = failures[: args.n]

    print(f"run={data['run_name']}  exec_acc={data['execution_accuracy']:.4f}"
          f"  failures={len(failures)}/{len(data['records'])}")
    print("-" * 70)

    for i, rec in enumerate(selected, 1):
        print(f"\n[{i}] db={rec['db_id']}")
        print(f"  question : {rec['question']}")
        print(f"  gold     : {rec['gold_sql']}")
        print(f"  predicted: {rec['predicted_sql']}")
        print(f"  reason   : {rec['reason']}")

    out_path = os.path.join(
        RESULTS_DIR, f"error_analysis_{args.run_name}.json"
    )
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "run_name": data["run_name"],
                "execution_accuracy": data["execution_accuracy"],
                "total_failures": len(failures),
                "total_examples": len(data["records"]),
                "selected_failures": selected,
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
