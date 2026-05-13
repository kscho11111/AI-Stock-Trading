from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


# =========================
# 경로 설정
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE_PATH = PROCESSED_DIR / "krw_btc_minute1_features.csv"
OUTPUT_FILE_PATH = PROCESSED_DIR / "krw_btc_minute1_signals.csv"


# =========================
# 보조 함수
# =========================
def normalize_pattern(prices: np.ndarray) -> np.ndarray:
    """
    시작값 기준 정규화
    예:
    [100, 102, 101] -> [0.0, 0.02, 0.01]
    """
    prices = np.asarray(prices, dtype=float)

    if len(prices) == 0:
        return prices

    first_value = prices[0]
    if first_value == 0:
        return np.zeros_like(prices, dtype=float)

    return prices / first_value - 1.0


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    두 패턴 간 유클리드 거리
    """
    return float(np.linalg.norm(a - b))


def build_pattern_library(
    df: pd.DataFrame,
    pattern_len: int,
    future_len: int,
    price_col: str = "close",
) -> List[dict]:
    """
    과거 패턴 라이브러리 생성

    각 원소:
    - pattern_start_idx
    - pattern_end_idx
    - normalized_pattern
    - future_return
    """
    library: List[dict] = []

    prices = df[price_col].values
    n = len(prices)

    # 패턴 구간 + 미래 구간이 모두 존재해야 함
    max_start = n - pattern_len - future_len
    if max_start < 0:
        return library

    for start_idx in range(max_start + 1):
        end_idx = start_idx + pattern_len
        future_end_idx = end_idx + future_len - 1

        pattern = prices[start_idx:end_idx]
        normalized = normalize_pattern(pattern)

        current_price = prices[end_idx - 1]
        future_price = prices[future_end_idx]
        future_return = future_price / current_price - 1.0

        library.append(
            {
                "pattern_start_idx": start_idx,
                "pattern_end_idx": end_idx - 1,
                "normalized_pattern": normalized,
                "future_return": future_return,
            }
        )

    return library


def find_similar_patterns(
    current_pattern: np.ndarray,
    library: List[dict],
    top_k: int,
) -> List[dict]:
    """
    현재 패턴과 가장 유사한 과거 패턴 top_k 반환
    """
    scored_patterns: List[dict] = []

    for item in library:
        dist = euclidean_distance(current_pattern, item["normalized_pattern"])

        scored_patterns.append(
            {
                **item,
                "distance": dist,
            }
        )

    scored_patterns.sort(key=lambda x: x["distance"])
    return scored_patterns[:top_k]


def decide_signal(
    similar_patterns: List[dict],
    buy_threshold: float = 0.002,
    sell_threshold: float = -0.002,
) -> tuple[str, float, float, float]:
    """
    유사 패턴들의 미래 수익률 평균을 기준으로 신호 결정

    Returns:
        signal, avg_future_return, win_rate, avg_distance
    """
    if not similar_patterns:
        return "HOLD", np.nan, np.nan, np.nan

    future_returns = np.array(
        [item["future_return"] for item in similar_patterns],
        dtype=float,
    )
    distances = np.array(
        [item["distance"] for item in similar_patterns],
        dtype=float,
    )

    avg_future_return = float(np.mean(future_returns))
    win_rate = float(np.mean(future_returns > 0))
    avg_distance = float(np.mean(distances))

    if avg_future_return >= buy_threshold:
        signal = "BUY"
    elif avg_future_return <= sell_threshold:
        signal = "SELL"
    else:
        signal = "HOLD"

    return signal, avg_future_return, win_rate, avg_distance


# =========================
# 전략 적용
# =========================
def generate_pattern_signals(
    df: pd.DataFrame,
    pattern_len: int = 30,
    future_len: int = 5,
    top_k: int = 20,
    min_library_size: int = 100,
    recent_exclude: int = 30,
    price_col: str = "close",
) -> pd.DataFrame:
    """
    각 시점마다 현재 패턴을 과거 패턴과 비교해서
    BUY / SELL / HOLD 신호를 생성한다.

    Args:
        pattern_len: 현재 패턴 길이
        future_len: 과거 패턴 이후 성과를 볼 구간 길이
        top_k: 유사 패턴 상위 개수
        min_library_size: 라이브러리가 너무 작을 때는 신호 생성 안 함
        recent_exclude: 현재와 너무 가까운 최근 구간은 비교 대상에서 제외
        price_col: 패턴 비교에 사용할 가격 컬럼
    """
    df = df.copy()

    # 기본 정리
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    if price_col not in df.columns:
        raise KeyError(f"'{price_col}' 컬럼이 없습니다.")

    # 결과 컬럼 초기화
    df["signal"] = "HOLD"
    df["pattern_score"] = np.nan
    df["pattern_win_rate"] = np.nan
    df["pattern_avg_distance"] = np.nan
    df["pattern_match_count"] = 0

    prices = pd.to_numeric(df[price_col], errors="coerce").values
    n = len(df)

    # 현재 패턴을 만들 수 있는 최소 시작점
    start_index = pattern_len
    # 미래 성과를 계산하지 않으므로 마지막까지 신호는 가능
    end_index = n

    for i in range(start_index, end_index):
        # 현재 패턴: [i-pattern_len, i)
        current_raw = prices[i - pattern_len:i]

        if np.isnan(current_raw).any():
            continue

        current_pattern = normalize_pattern(current_raw)

        # 과거 비교용 데이터: 현재 직전까지 중 최근 일부는 제외
        historical_end = i - recent_exclude
        if historical_end <= pattern_len + future_len:
            continue

        historical_df = df.iloc[:historical_end].copy()

        library = build_pattern_library(
            df=historical_df,
            pattern_len=pattern_len,
            future_len=future_len,
            price_col=price_col,
        )

        if len(library) < max(top_k, min_library_size):
            continue

        similar_patterns = find_similar_patterns(
            current_pattern=current_pattern,
            library=library,
            top_k=top_k,
        )

        signal, avg_future_return, win_rate, avg_distance = decide_signal(
            similar_patterns=similar_patterns,
            buy_threshold=0.002,   # +0.2%
            sell_threshold=-0.002, # -0.2%
        )

        df.loc[i, "signal"] = signal
        df.loc[i, "pattern_score"] = avg_future_return
        df.loc[i, "pattern_win_rate"] = win_rate
        df.loc[i, "pattern_avg_distance"] = avg_distance
        df.loc[i, "pattern_match_count"] = len(similar_patterns)

    return df


# =========================
# 데이터 로드 / 저장
# =========================
def load_feature_data(file_path: Path = INPUT_FILE_PATH) -> pd.DataFrame:
    """
    features CSV 로드
    """
    if not file_path.exists():
        raise FileNotFoundError(f"features 파일이 존재하지 않습니다: {file_path}")

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError(f"features 파일이 비어 있습니다: {file_path}")

    return df


def save_signal_data(df: pd.DataFrame, file_path: Path = OUTPUT_FILE_PATH) -> Path:
    """
    signals CSV 저장
    """
    df.to_csv(file_path, index=False, encoding="utf-8-sig")
    return file_path


# =========================
# 검증/요약
# =========================
def validate_signal_data(df: pd.DataFrame) -> None:
    """
    signal 생성 결과 요약 출력
    """
    print("[INFO] strategy 결과 요약")
    print(f" - 총 행 수: {len(df)}")
    print(f" - 시작 시각: {df['datetime'].min()}")
    print(f" - 종료 시각: {df['datetime'].max()}")

    signal_counts = df["signal"].value_counts(dropna=False).to_dict()
    print(f" - signal 분포: {signal_counts}")

    valid_score_count = df["pattern_score"].notna().sum()
    print(f" - pattern_score 존재 행 수: {valid_score_count}")

    if valid_score_count > 0:
        print(f" - pattern_score 평균: {df['pattern_score'].dropna().mean():.6f}")
        print(f" - pattern_win_rate 평균: {df['pattern_win_rate'].dropna().mean():.4f}")
        print(f" - pattern_avg_distance 평균: {df['pattern_avg_distance'].dropna().mean():.6f}")


# =========================
# 실행
# =========================
def main() -> None:
    try:
        print(f"[INFO] features 데이터 로드: {INPUT_FILE_PATH}")
        df = load_feature_data(INPUT_FILE_PATH)

        print("[INFO] 패턴 기반 strategy 시작")
        signal_df = generate_pattern_signals(
            df=df,
            pattern_len=30,
            future_len=5,
            top_k=20,
            min_library_size=100,
            recent_exclude=30,
            price_col="close",
        )

        validate_signal_data(signal_df)

        saved_path = save_signal_data(signal_df, OUTPUT_FILE_PATH)
        print(f"[OK] strategy 완료 및 저장: {saved_path}")

    except Exception as e:
        print(f"[ERROR] strategy 실패: {e}")


if __name__ == "__main__":
    main()


    