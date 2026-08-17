"""
Generate a formal backtest report (return curves + metrics) from a saved
backtest recorder.

Loads the portfolio analysis report from a backtest_kfold run, produces:
  - cumulative return curve (strategy vs benchmark)
  - excess return curve
  - drawdown curve
  - full metrics table (annualized return, vol, Sharpe, IR, max drawdown, ...)

Usage
-----
    python daily_quant/xgboost/report_kfold.py --recorder_id <id> --freq day
    python daily_quant/xgboost/report_kfold.py --freq day   # latest backtest_kfold run
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.workflow import R

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Chinese font (Windows)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ANN = {"day": 238, "week": 50}


def load_report(recorder_id, experiment_name="backtest_kfold", freq="day"):
    rec = R.get_recorder(recorder_id=recorder_id, experiment_name=experiment_name)
    return rec.load_object(f"portfolio_analysis/report_normal_1{freq}.pkl")


def compute_metrics(ret, bench, cost, turnover, freq):
    ann = ANN[freq]
    net = ret - cost
    excess = ret - bench
    net_excess = net - bench

    def _stats(s, is_return=True):
        cum = (1 + s).cumprod()
        dd = cum / cum.cummax() - 1
        out = {
            "累计收益": cum.iloc[-1] - 1,
            "年化收益": s.mean() * ann,
            "年化波动": s.std() * np.sqrt(ann),
            "最大回撤": dd.min(),
            "胜率": (s > 0).mean(),
        }
        if is_return:
            out["Sharpe"] = (s.mean() * ann) / (s.std() * np.sqrt(ann))
        return out

    strat_gross = _stats(ret)
    strat_net = _stats(net)
    bm = _stats(bench)
    ex_gross = _stats(excess)
    ex_net = _stats(net_excess)
    ex_net["信息比率IR"] = (net_excess.mean() * ann) / (net_excess.std() * np.sqrt(ann))

    return {
        "策略毛": strat_gross,
        "策略净": strat_net,
        "基准": bm,
        "超额毛": ex_gross,
        "超额净": ex_net,
        "平均日换手": turnover.mean(),
        "累计换手": turnover.sum(),
        "日均成本": cost.mean(),
        "交易日数": len(ret),
    }


def plot_report(ret, bench, freq, out_path):
    ann = ANN[freq]
    cum_ret = (1 + ret).cumprod()
    cum_bench = (1 + bench).cumprod()
    excess = ret - bench
    cum_excess = (1 + excess).cumprod()
    dd = cum_ret / cum_ret.cummax() - 1

    fig, axes = plt.subplots(3, 1, figsize=(12, 13), sharex=True)

    axes[0].plot(cum_ret.index, cum_ret.values, label="策略", lw=1.8, color="#c0392b")
    axes[0].plot(cum_bench.index, cum_bench.values, label="基准 (SH510050)", lw=1.5, color="#2980b9")
    axes[0].set_title("累计净值曲线")
    axes[0].set_ylabel("净值")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.3)

    axes[1].plot(cum_excess.index, cum_excess.values, label="超额净值", lw=1.8, color="#16a085")
    axes[1].axhline(1.0, color="gray", ls="--", lw=1)
    axes[1].set_title("累计超额收益")
    axes[1].set_ylabel("超额净值")
    axes[1].legend(loc="upper left")
    axes[1].grid(alpha=0.3)

    axes[2].fill_between(dd.index, dd.values * 100, 0, color="#c0392b", alpha=0.4)
    axes[2].set_title("策略回撤")
    axes[2].set_ylabel("回撤 (%)")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def format_report(metrics):
    lines = []
    lines.append("=" * 60)
    lines.append("ETF 选股策略回测报告")
    lines.append("=" * 60)
    lines.append("")

    def _row(d, title):
        lines.append(f"【{title}】")
        for k, v in d.items():
            if k in ("Sharpe", "信息比率IR", "胜率"):
                lines.append(f"  {k:<10}: {v:.4f}")
            elif "收益" in k or "回撤" in k or "换手" in k or "成本" in k:
                lines.append(f"  {k:<10}: {v:.2%}")
            else:
                lines.append(f"  {k:<10}: {v:.4f}")
        lines.append("")

    _row(metrics["策略毛"], "策略（毛收益，未扣费）")
    _row(metrics["策略净"], "策略（净收益，扣费后）")
    _row(metrics["基准"], "基准")
    _row(metrics["超额毛"], "超额（未扣费）")
    _row(metrics["超额净"], "超额（扣费后）")
    lines.append(f"  平均日换手  : {metrics['平均日换手']:.2%}")
    lines.append(f"  累计换手    : {metrics['累计换手']:.2%}")
    lines.append(f"  日均成本    : {metrics['日均成本']:.4%}")
    lines.append(f"  交易日数    : {metrics['交易日数']}")
    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate formal backtest report")
    parser.add_argument("--recorder_id", default=None, help="backtest recorder id (default: latest)")
    parser.add_argument("--experiment_name", default="backtest_kfold")
    parser.add_argument("--freq", default="day", choices=["day", "week"])
    parser.add_argument("--qlib_data_dir", default=r"C:\Users\pp\.qlib\qlib_data\etf_data")
    args = parser.parse_args()

    qlib.init(provider_uri=args.qlib_data_dir, region=REG_CN)

    if args.recorder_id is None:
        recs = R.list_recorders(experiment_name=args.experiment_name)
        recs = [r for r in recs.values() if f"portfolio_analysis/report_normal_1{args.freq}.pkl" in r.list_artifacts()]
        if not recs:
            raise ValueError(f"No backtest report found in '{args.experiment_name}'")
        recs.sort(key=lambda r: r.info.get("end_time") or "", reverse=True)
        args.recorder_id = recs[0].id
        print(f"Using latest backtest recorder: {args.recorder_id}")

    report = load_report(args.recorder_id, args.experiment_name, args.freq)
    ret = report["return"]
    bench = report["bench"]
    turnover = report["turnover"]

    metrics = compute_metrics(ret, bench, report["cost"], turnover, args.freq)

    out_dir = Path(__file__).resolve().parent / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.recorder_id[:16]

    png_path = out_dir / f"backtest_report_{tag}.png"
    plot_report(ret, bench, args.freq, png_path)

    txt_path = out_dir / f"backtest_report_{tag}.txt"
    report_text = format_report(metrics)
    txt_path.write_text(report_text, encoding="utf-8")

    print(report_text)
    print(f"\n图表已保存: {png_path}")
    print(f"报告已保存: {txt_path}")


if __name__ == "__main__":
    main()
