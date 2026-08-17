"""
查看/导出当前生产模型的参数和评估信息。

Usage
-----
    python model_info.py              # 查看最新模型信息
    python model_info.py --export     # 导出 model_info.json
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import mlflow

CUR_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CUR_DIR.parent))

import qlib
from qlib.constant import REG_CN
from qlib.workflow import R

# MLflow tracking URI — same dir where mlruns/ lives
mlflow.set_tracking_uri(f"file:///{CUR_DIR.as_posix()}/mlruns")


def get_model_info(experiment_name="GRU_mid_cap_60d", model_id=None):
    """Get model info from MLflow."""
    qlib.init(provider_uri=r"C:\Users\pp\.qlib\qlib_data\cn_data_bwd", region=REG_CN)

    recs_dict = R.list_recorders(experiment_name=experiment_name)
    recs = [r for rid, r in recs_dict.items() if "trained_model" in r.list_artifacts()]
    if not recs:
        raise ValueError(f"No models found in '{experiment_name}'")

    recs.sort(key=lambda r: r.info.get("end_time") or "", reverse=True)
    rec = recs[0] if model_id is None else recs_dict.get(model_id)
    if rec is None:
        raise ValueError(f"Model {model_id} not found")

    info = rec.info
    run_id = rec.id

    # Get params and metrics from MLflow run
    client = mlflow.tracking.MlflowClient()
    mlflow_run = client.get_run(run_id)
    params = mlflow_run.data.params
    metrics = mlflow_run.data.metrics

    # Get tags
    try:
        tags = {t.key: t.value for t in mlflow_run.data.tags if t.key != "mlflow.user"}
    except Exception:
        tags = {}

    result = {
        "recorder_id": rec.id,
        "experiment": experiment_name,
        "status": info.get("status", "?"),
        "start_time": info.get("start_time", "?"),
        "end_time": info.get("end_time", "?"),
        "tags": tags,
        "metrics": metrics,
        "params": params,
        "artifacts": rec.list_artifacts(),
        "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # If metrics are empty (MLflow run might not have them), try to compute
    # IC from prediction + label data as a fallback evaluation
    return result


def main():
    parser = argparse.ArgumentParser(description="Model info viewer")
    parser.add_argument("--model_id", default=None)
    parser.add_argument("--experiment", default="GRU_mid_cap_60d")
    parser.add_argument("--export", action="store_true", help="Export to model_info.json")
    args = parser.parse_args()

    info = get_model_info(args.experiment, args.model_id)

    print(f"{'='*60}")
    print(f"Model: {info['recorder_id']}")
    print(f"Experiment: {info['experiment']}")
    print(f"Status: {info['status']}")
    print(f"Trained: {info['start_time']} -> {info['end_time']}")
    if info["tags"]:
        print(f"Tags: {json.dumps(info['tags'], ensure_ascii=False)}")
    print()

    if info["params"]:
        print("=== Model Parameters ===")
        for k, v in sorted(info["params"].items()):
            print(f"  {k}: {v}")
    else:
        print("(no params stored)")
    print()

    if info["metrics"]:
        print("=== Training Metrics ===")
        for k, v in sorted(info["metrics"].items()):
            print(f"  {k}: {v:.6f}")
    else:
        print("(no metrics stored)")
    print()

    if args.export:
        out_path = CUR_DIR / "model_info.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2, default=str)
        print(f"已导出: {out_path}")


if __name__ == "__main__":
    main()
