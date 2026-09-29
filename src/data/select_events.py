"""
강우 이벤트 선별 스크립트.
AWS 시간강수량 CSV → SWMM 학습용 이벤트 목록 출력.

이벤트 정의:
  - 총 강수량 >= min_total_mm
  - 이벤트 앞뒤 dry_gap_h 시간 이상 무강우
  - 최소 지속시간 >= min_duration_h

Regime 분류:
  - A (고강도): 최대 시간강수량 >= 10mm/hr 또는 총량 >= 50mm
  - B (혼합):   총량 20~50mm or 최대 5~10mm/hr
  - C (소규모): 총량 10~20mm
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ── 설정 ────────────────────────────────────────────────
DATA_DIR   = Path(__file__).parents[2] / "data" / "rainfall"
OUT_DIR    = Path(__file__).parents[2] / "data"
OUT_JSON   = OUT_DIR / "event_catalog.json"
OUT_CSV    = OUT_DIR / "event_catalog.csv"

MIN_TOTAL_MM   = 10.0   # 이벤트 최소 총강수량 (mm)
DRY_GAP_H      = 6      # 이벤트 앞뒤 무강우 간격 (시간)
MIN_DURATION_H = 1      # 최소 지속 시간

TARGET_COUNTS = {"A": 45, "B": 30, "C": 12, "holdout": 5}  # 목표 이벤트 수 (확장: 30→92)


# ── CSV 파일 로드 ────────────────────────────────────────
def load_aws_files(data_dir: Path) -> pd.Series:
    """모든 AWS CSV를 읽어 station 401 시간강수량 시계열로 병합."""
    frames = []
    for f in sorted(data_dir.glob("*.csv")):
        try:
            df = pd.read_csv(f, encoding="cp949", header=0,
                             names=["stn_id", "stn_name", "datetime", "rain_mm"])
            df["rain_mm"] = pd.to_numeric(df["rain_mm"], errors="coerce").fillna(0.0)
            df["rain_mm"] = df["rain_mm"].clip(lower=0)
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
            df = df.dropna(subset=["datetime"])
            # station 401 우선, 없으면 400
            for stn in [401, 400]:
                sub = df[df["stn_id"] == stn]
                if len(sub) > 0:
                    frames.append(sub[["datetime", "rain_mm"]])
                    print(f"  {f.name}: stn={stn}, {len(sub)} rows, "
                          f"{sub['datetime'].min()} ~ {sub['datetime'].max()}")
                    break
        except Exception as e:
            print(f"  [WARN] {f.name}: {e}")

    if not frames:
        raise RuntimeError("No data loaded.")

    combined = pd.concat(frames).drop_duplicates("datetime").sort_values("datetime")
    series = combined.set_index("datetime")["rain_mm"]
    # 1시간 간격으로 리샘플 (결측 → 0)
    series = series.resample("1h").sum().fillna(0.0)
    print(f"\n전체 시계열: {series.index[0]} ~ {series.index[-1]}, "
          f"{len(series)} 시간, 총강수량 {series.sum():.1f} mm")
    return series


# ── 이벤트 탐지 ─────────────────────────────────────────
def detect_events(series: pd.Series,
                  min_total: float = MIN_TOTAL_MM,
                  dry_gap: int = DRY_GAP_H,
                  min_dur: int = MIN_DURATION_H) -> list:
    """시계열에서 강우 이벤트 탐지."""
    rain = series.values
    times = series.index
    n = len(rain)

    events = []
    in_event = False
    start_idx = 0
    dry_count = 0

    for i in range(n):
        if rain[i] > 0:
            if not in_event:
                # 앞 dry_gap 확인
                if i < dry_gap or all(rain[max(0, i-dry_gap):i] == 0):
                    in_event = True
                    start_idx = i
            dry_count = 0
        else:
            if in_event:
                dry_count += 1
                if dry_count >= dry_gap:
                    end_idx = i - dry_gap
                    _save_event(events, rain, times, start_idx, end_idx,
                                min_total, min_dur)
                    in_event = False
                    dry_count = 0

    if in_event:
        _save_event(events, rain, times, start_idx, n - 1, min_total, min_dur)

    return events


def _save_event(events, rain, times, start, end, min_total, min_dur):
    if end <= start:
        return
    segment = rain[start:end + 1]
    total = segment.sum()
    duration = end - start + 1
    if total >= min_total and duration >= min_dur:
        events.append({
            "start": str(times[start]),
            "end":   str(times[end]),
            "duration_h": int(duration),
            "total_mm":   round(float(total), 1),
            "peak_mm_hr": round(float(segment.max()), 1),
            "mean_mm_hr": round(float(segment[segment > 0].mean()), 2),
        })


# ── Regime 분류 ──────────────────────────────────────────
def classify_regime(event: dict) -> str:
    if event["peak_mm_hr"] >= 10 or event["total_mm"] >= 50:
        return "A"
    elif event["total_mm"] >= 20 or event["peak_mm_hr"] >= 5:
        return "B"
    else:
        return "C"


# ── 이벤트 선별 ──────────────────────────────────────────
def select_events(events: list, targets: dict = TARGET_COUNTS) -> list:
    """각 Regime에서 목표 수만큼 선별. holdout은 최신 이벤트에서 추출."""
    for ev in events:
        ev["regime"] = classify_regime(ev)

    # Holdout: 2024년 이후 상위 이벤트
    holdout_pool = [e for e in events if e["start"] >= "2024-01-01"]
    holdout_pool.sort(key=lambda x: x["total_mm"], reverse=True)
    holdout = holdout_pool[:targets["holdout"]]
    holdout_ids = {e["start"] for e in holdout}
    for e in holdout:
        e["split"] = "holdout"

    # Train pool (holdout 제외)
    train_pool = [e for e in events if e["start"] not in holdout_ids]

    selected = list(holdout)
    for regime, n_target in [("A", targets["A"]),
                              ("B", targets["B"]),
                              ("C", targets["C"])]:
        pool = [e for e in train_pool if e["regime"] == regime]
        # 총강수량 내림차순 정렬 후 상위 n_target 선택
        pool.sort(key=lambda x: x["total_mm"], reverse=True)
        chosen = pool[:n_target]
        for e in chosen:
            e["split"] = "train"
        selected.extend(chosen)

    # event_id 부여
    selected.sort(key=lambda x: x["start"])
    for i, e in enumerate(selected):
        e["event_id"] = f"E{i+1:03d}"

    return selected


# ── 결과 출력 ────────────────────────────────────────────
def print_summary(selected: list):
    print("\n" + "="*60)
    print(f"{'이벤트':^6} {'시작':^17} {'종료':^17} {'총량':>7} {'최대':>7} {'Regime':^7} {'Split':^8}")
    print("-"*60)
    for e in selected:
        print(f"{e['event_id']:^6} {e['start'][:16]:^17} {e['end'][:16]:^17} "
              f"{e['total_mm']:>6.1f}mm {e['peak_mm_hr']:>6.1f}/hr "
              f"  {e['regime']:^5}   {e['split']:^8}")

    print("="*60)
    from collections import Counter
    regime_cnt = Counter(e["regime"] for e in selected)
    split_cnt  = Counter(e["split"]  for e in selected)
    print(f"Regime 분포: A={regime_cnt.get('A',0)}, "
          f"B={regime_cnt.get('B',0)}, C={regime_cnt.get('C',0)}")
    print(f"Split 분포: train={split_cnt.get('train',0)}, "
          f"holdout={split_cnt.get('holdout',0)}")
    print(f"총 선별 이벤트: {len(selected)}개")


def main():
    raise RuntimeError("Legacy exploratory selector; it does not generate the locked final splits. "
                       "See docs/DATA_PROTOCOL.md.")
    print("강우 이벤트 선별 시작\n")

    series = load_aws_files(DATA_DIR)
    print("\n이벤트 탐지 중...")
    all_events = detect_events(series)
    print(f"탐지된 전체 이벤트: {len(all_events)}개 (총강수량 ≥{MIN_TOTAL_MM}mm)")

    # Regime별 분포 확인
    from collections import Counter
    regimes = [classify_regime(e) for e in all_events]
    rc = Counter(regimes)
    print(f"  A(고강도): {rc.get('A',0)}개, B(혼합): {rc.get('B',0)}개, "
          f"C(소규모): {rc.get('C',0)}개")

    selected = select_events(all_events)
    print_summary(selected)

    # 저장
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)

    df = pd.DataFrame(selected)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\n저장 완료:")
    print(f"  {OUT_JSON}")
    print(f"  {OUT_CSV}")


if __name__ == "__main__":
    main()
