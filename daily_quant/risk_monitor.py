"""
Strategy & market risk monitor — detect when the system is weakening.

Signals computed from daily backtest report:

    Signal              Threshold       Meaning
    --------            ---------       -------
    roll_win_20d        < 50%           20-day win rate vs benchmark
    roll_sharpe_20d     < 1.0           rolling Sharpe (excess)
    roll_dd_20d         < -8%           max drawdown over 20 days
    lose_streak         >= 3            consecutive losing days
    dd_from_peak        < -10%          drawdown from all-time peak
    bench_trend_5d      < 0             5-day benchmark mean daily return
    bench_trend_20d     < -0.003        20-day benchmark mean daily return
    bench_dd            < -5%           benchmark drawdown

Risk level:
    0-2 warnings  → GREEN   — normal
    3-4 warnings  → YELLOW  — caution, reduce position
    5+ warnings   → RED     — stop trading, wait for recovery

Usage
-----
    from daily_quant.risk_monitor import RiskMonitor

    report = pd.read_pickle("portfolio_analysis/report_normal_1day.pkl")
    monitor = RiskMonitor(report)
    print(monitor.status())
    print(monitor.history())
"""

import pandas as pd
import numpy as np


class RiskMonitor:
    """Monitor strategy and market health from daily return series."""

    def __init__(self, report: pd.DataFrame):
        """
        Parameters
        ----------
        report : DataFrame
            Backtest report with columns ``return``, ``bench``.
        """
        self.daily = report[["return", "bench"]].copy()
        self.daily["excess"] = self.daily["return"] - self.daily["bench"]
        self._lookback = 20
        self._ann = 238
        self._compute()

    def _compute(self):
        d = self.daily
        n = self._lookback

        # cumulative
        d["cum_return"] = (d["return"] + 1).cumprod()
        d["cum_bench"] = (d["bench"] + 1).cumprod()

        # rolling win rate (vs benchmark)
        d["roll_win"] = (d["excess"] > 0).rolling(n).mean()

        # rolling Sharpe (excess)
        ex_mean = d["excess"].rolling(n).mean()
        ex_std = d["excess"].rolling(n).std().replace(0, np.nan)
        d["roll_sharpe"] = ex_mean / ex_std * np.sqrt(self._ann)

        # rolling max drawdown
        d["roll_dd"] = d["return"].rolling(n).apply(self._max_dd, raw=True)

        # consecutive losing streak
        is_lose = (d["excess"] <= 0).astype(int)
        streak_id = (is_lose != is_lose.shift()).cumsum()
        d["lose_streak"] = is_lose.groupby(streak_id).cumsum() * is_lose

        # drawdown from all-time peak
        d["dd_from_peak"] = d["cum_return"] / d["cum_return"].cummax() - 1

        # benchmark signals
        d["bench_trend_5"] = d["bench"].rolling(5).mean()
        d["bench_trend_20"] = d["bench"].rolling(20).mean()
        d["bench_dd"] = d["cum_bench"] / d["cum_bench"].cummax() - 1

    @staticmethod
    def _max_dd(arr):
        cum = np.cumprod(1 + arr)
        return (cum / np.maximum.accumulate(cum) - 1).min()

    # ---- thresholds ----

    THRESHOLDS = {
        "roll_win":        (0.50,   "below", "20d win rate [{:.0%}] < 50%"),
        "roll_sharpe":     (1.0,    "below", "20d Sharpe [{:.2f}] < 1.0"),
        "roll_dd":         (-0.08,  "below", "20d max DD [{:.1%}] < -8%"),
        "lose_streak":     (3,      "above", "lose streak [{:.0f}] >= 3 days"),
        "dd_from_peak":    (-0.10,  "below", "peak DD [{:.1%}] < -10%"),
        "bench_trend_5":   (0.0,    "below", "bench 5d trend [{:.4f}] < 0"),
        "bench_trend_20":  (-0.003, "below", "bench 20d trend [{:.4f}] < -0.3%"),
        "bench_dd":        (-0.05,  "below", "bench DD [{:.1%}] < -5%"),
    }

    def check(self, date=None) -> dict:
        """Return {signal_name: (value, threshold, triggered)} for *date*.

        If *date* is None, uses the last available date.
        """
        if date is None:
            date = self.daily.index[-1]
        row = self.daily.loc[date]
        result = {}
        for name, (thresh, direction, _) in self.THRESHOLDS.items():
            val = row[name]
            if pd.isna(val):
                triggered = False
            elif direction == "below":
                triggered = val <= thresh
            else:
                triggered = val >= thresh
            result[name] = (val, thresh, triggered)
        return result

    def status(self, date=None) -> str:
        """Human-readable risk status."""
        checks = self.check(date)
        warnings = [n for n, (v, t, trig) in checks.items() if trig]
        n_warn = len(warnings)
        if n_warn <= 2:
            level = "GREEN"
        elif n_warn <= 4:
            level = "YELLOW"
        else:
            level = "RED"

        if date is None:
            date = self.daily.index[-1]
        lines = [f"RiskMonitor @ {date.strftime('%Y-%m-%d')}  level={level}  ({n_warn}/{len(checks)} warnings)"]
        lines.append("-" * 64)
        lines.append(f"{'Signal':<20} {'Value':>10} {'Threshold':>10} {'Status':>8}")
        lines.append("-" * 64)
        for name in self.THRESHOLDS:
            val, thresh, trig = checks[name]
            s = "WARN" if trig else "OK"
            lines.append(f"  {name:<18} {val:>10.4f} {thresh:>10.4f} {s:>8}")
        lines.append("-" * 64)
        lines.append({
            "GREEN":  "All clear — system is healthy.",
            "YELLOW": "Caution — consider reducing position size.",
            "RED":    "STOP — do not trade, wait for recovery.",
        }[level])
        return "\n".join(lines)

    def history(self, tail=60) -> pd.DataFrame:
        """Return DataFrame of warning counts per day (last *tail* days)."""
        d = self.daily.tail(tail)
        counts = []
        for idx in d.index:
            checks = self.check(idx)
            n = sum(1 for _, _, trig in checks.values() if trig)
            counts.append((idx, n))
        hist = pd.DataFrame(counts, columns=["date", "warnings"])
        hist["level"] = hist["warnings"].apply(
            lambda n: "RED" if n >= 5 else ("YELLOW" if n >= 3 else "GREEN")
        )
        return hist.set_index("date")

    def summary(self) -> str:
        """Print current status + history tail."""
        lines = [self.status(), "", "=== Recent history ==="]
        hist = self.history(30)
        for idx, row in hist.iterrows():
            w = int(row["warnings"])
            bar = "#" * w + "-" * (8 - w)
            lines.append(f"  {idx.strftime('%m-%d')}  {row['level']:<6s}  {bar}  ({w}/8)")
        return "\n".join(lines)


# ============================================================
class PredictionRiskMonitor:
    """Monitor signal quality from prediction scores alone (no backtest needed).

    Signals computed from ``pred_score_*.csv`` files:

        Signal              Threshold       Meaning
        ------              ---------       -------
        top_score_mean      < 0.5           avg score of top 30 stocks declining
        score_std           < 0.2           model can't differentiate stocks
        neg_ratio           > 0.70          >70% stocks have negative scores
        top_score_trend     < -0.02         5-day slope of top_score_mean
        score_turnover      > 0.60          top 30 list changed >60% vs yesterday
        model_age           > 180           days since last training
        data_age            > 2             trading days since last data update
    """

    THRESHOLDS = {
        "top_score_mean":   (0.5,    "below", "top30 avg score [{:.3f}] < 0.5"),
        "score_std":        (0.2,    "below", "score std [{:.3f}] < 0.2"),
        "neg_ratio":        (0.70,   "above", "negative ratio [{:.0%}] > 70%"),
        "score_turnover":   (0.60,   "above", "top30 turnover [{:.0%}] > 60%"),
        "top_score_trend":  (-0.02,  "below", "5d trend [{:.3f}] < -0.02/day"),
    }

    def __init__(self, score_files: dict, n_top: int = 30):
        """
        Parameters
        ----------
        score_files : {date_str: path_to_csv}
            Dict of prediction files, e.g. {"2026-06-01": Path("pred_score_2026-06-01.csv")}.
        n_top : int
            Number of top stocks to track.
        """
        self.n_top = n_top
        self.dates = sorted(score_files.keys())
        self._load_all(score_files)

    def _load_all(self, score_files: dict):
        """Load all prediction files, compute daily metrics."""
        rows = []
        for d in self.dates:
            df = pd.read_csv(score_files[d])
            scores = df.iloc[:, -1]  # last column = score
            top_n = scores.nlargest(self.n_top)
            rows.append({
                "date": d,
                "top_score_mean": top_n.mean(),
                "top_score_std": top_n.std(),
                "score_mean": scores.mean(),
                "score_std": scores.std(),
                "score_min": scores.min(),
                "score_max": scores.max(),
                "neg_ratio": (scores < 0).mean(),
                "n_stocks": len(scores),
            })
        self.df = pd.DataFrame(rows)
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.df = self.df.set_index("date").sort_index()

        # Derived: top score trend (5-day linear slope)
        self.df["top_score_trend"] = (
            self.df["top_score_mean"]
            .rolling(5, min_periods=2)
            .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False)
        )

        # Derived: day-over-day turnover in top stocks
        self._compute_turnover(score_files)

    def _compute_turnover(self, score_files: dict):
        """Compute fraction of top N stocks that changed from previous day."""
        prev_top = None
        turnovers = []
        for d in self.dates:
            df = pd.read_csv(score_files[d])
            scores = df.iloc[:, -1]
            curr_top = set(scores.nlargest(self.n_top).index)
            if prev_top is not None:
                overlap = len(curr_top & prev_top)
                turnover = 1 - overlap / self.n_top
            else:
                turnover = np.nan
            turnovers.append(turnover)
            prev_top = curr_top
        self.df["score_turnover"] = turnovers

    def check(self, date=None) -> dict:
        """Return {signal: (value, threshold, triggered)} for *date*."""
        if date is None:
            date = self.df.index[-1]
        row = self.df.loc[date]
        result = {}
        for name, (thresh, direction, _) in self.THRESHOLDS.items():
            val = row.get(name, np.nan)
            if pd.isna(val):
                triggered = False
            elif direction == "below":
                triggered = val <= thresh
            else:
                triggered = val >= thresh
            result[name] = (val, thresh, triggered)
        return result

    def status(self, date=None, model_train_end=None, qlib_dir=None) -> str:
        """Human-readable risk status from prediction signals."""
        checks = self.check(date)
        all_warn = {n: (v, t, trig) for n, (v, t, trig) in checks.items() if trig}
        n_warn = len(all_warn)

        # model staleness (warning only)
        if model_train_end:
            age = (pd.Timestamp.now() - pd.Timestamp(model_train_end)).days
            if age > 180:
                n_warn += 1
                all_warn["model_age"] = (age, 180, True)

        # data freshness
        if qlib_dir:
            cal_path = Path(qlib_dir) / "calendars" / "day.txt"
            if cal_path.exists():
                cal = pd.read_csv(cal_path)
                last_cal = pd.Timestamp(cal.iloc[-1, 0])
                bdays = pd.bdate_range(last_cal, pd.Timestamp.now().normalize(), inclusive="right")
                if len(bdays) > 2:
                    n_warn += 1
                    all_warn["data_age"] = (len(bdays), 2, True)

        if n_warn <= 1:
            level = "GREEN"
        elif n_warn <= 3:
            level = "YELLOW"
        else:
            level = "RED"

        if date is None:
            date = self.df.index[-1]
        lines = [f"PredictionRisk @ {str(date)[:10]}  level={level}  ({n_warn} warnings)"]
        lines.append("-" * 64)
        lines.append(f"{'Signal':<20} {'Value':>10} {'Threshold':>10} {'Status':>8}")
        lines.append("-" * 64)
        for name in self.THRESHOLDS:
            if name in checks:
                val, thresh, trig = checks[name]
                lines.append(f"  {name:<18} {val:>10.4f} {thresh:>10.4f} {'WARN' if trig else 'OK':>8}")
        if "model_age" in all_warn:
            v, t, _ = all_warn["model_age"]
            lines.append(f"  {'model_age':<18} {v:>10.0f} {t:>10.0f} {'WARN':>8}")
        if "data_age" in all_warn:
            v, t, _ = all_warn["data_age"]
            lines.append(f"  {'data_age':<18} {v:>10.0f} {t:>10.0f} {'WARN':>8}")
        lines.append("-" * 64)
        lines.append({"GREEN": "Signal quality OK.", "YELLOW": "Signal weakening — caution.",
                       "RED": "Signal degraded — stop trading."}[level])
        return "\n".join(lines)

    def history(self) -> pd.DataFrame:
        """Return DataFrame with warnings per date."""
        counts = []
        for idx in self.df.index:
            checks = self.check(idx)
            n = sum(1 for _, _, trig in checks.values() if trig)
            counts.append((idx, n))
        hist = pd.DataFrame(counts, columns=["date", "warnings"])
        hist["level"] = hist["warnings"].apply(
            lambda n: "RED" if n >= 4 else ("YELLOW" if n >= 2 else "GREEN")
        )
        return hist.set_index("date")

    def risk_map(self) -> dict:
        """Return {date_str: (level, warnings)} for all dates."""
        hist = self.history()
        return {str(idx)[:10]: (row["level"], int(row["warnings"]))
                for idx, row in hist.iterrows()}


# ============================================================
# Standalone usage
# ============================================================
if __name__ == "__main__":
    import sys
    import re
    from pathlib import Path

    cur = Path(__file__).resolve().parent

    # -- Try PredictionRiskMonitor first (from pred_score_*.csv) --
    pattern = re.compile(r"pred_score_(\d{4}-\d{2}-\d{2})\.csv$")
    score_files: dict[str, Path] = {}
    pred_dir = cur / "predictions"
    if pred_dir.is_dir():
        for f in sorted(pred_dir.glob("*/*.csv")):
            m = pattern.match(f.name)
            if m:
                score_files[m.group(1)] = f
    # Fallback: legacy flat files
    for f in sorted(cur.glob("pred_score_*.csv")):
        m = pattern.match(f.name)
        if m and m.group(1) not in score_files:
            score_files[m.group(1)] = f

    if score_files:
        print("=== PredictionRiskMonitor (from pred_score_*.csv) ===\n")
        mon = PredictionRiskMonitor(score_files)
        print(mon.status(model_train_end="2024-12-31",
                         qlib_dir=r"C:\Users\pp\.qlib\qlib_data\cn_data_bwd"))
        print("\n=== History ===")
        for idx, row in mon.history().iterrows():
            w = int(row["warnings"])
            bar = "#" * w + "-" * (5 - w)
            print(f"  {str(idx)[:10]}  {row['level']:<6s}  {bar}  ({w}/5)")

    # -- Fallback to RiskMonitor (from backtest report) --
    exp_dir = cur / "mlruns" / "948584480955148621"
    recs = sorted(
        exp_dir.glob("*/artifacts/portfolio_analysis/report_normal_1day.pkl"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    if recs:
        print("\n=== RiskMonitor (from backtest) ===\n")
        report = pd.read_pickle(recs[0])
        mon2 = RiskMonitor(report)
        print(mon2.summary())
