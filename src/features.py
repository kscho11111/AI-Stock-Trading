from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# =========================
# 경로 설정
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE_PATH = PROCESSED_DIR / "krw_btc_minute1_processed.csv"
OUTPUT_FILE_PATH = PROCESSED_DIR / "krw_btc_minute1_features.csv"


# =========================
# 보조 함수
# =========================
def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI 계산
    """
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi


# =========================
# 데이터 로드
# =========================
def load_processed_data(file_path: Path = INPUT_FILE_PATH) -> pd.DataFrame:
    """
    processed CSV 로드
    """
    if not file_path.exists():
        raise FileNotFoundError(f"processed 파일이 존재하지 않습니다: {file_path}")

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError(f"processed 파일이 비어 있습니다: {file_path}")

    return df


# =========================
# feature 생성
# =========================
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    BTC 1분봉 데이터에 feature 컬럼 추가
    """
    df = df.copy()

    # datetime 정렬 보장
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    # 숫자형 보정
    numeric_columns = ["open", "high", "low", "close", "volume", "value"]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # -------------------------
    # 1. 수익률 계열
    # -------------------------
    df["return_1"] = df["close"].pct_change(periods=1)
    df["return_3"] = df["close"].pct_change(periods=3)
    df["return_5"] = df["close"].pct_change(periods=5)

    # 로그수익률도 하나 추가해두면 나중에 유용함
    df["log_return_1"] = np.log(df["close"] / df["close"].shift(1))

    # -------------------------
    # 2. 이동평균
    # -------------------------
    df["ma5"] = df["close"].rolling(window=5, min_periods=5).mean()
    df["ma10"] = df["close"].rolling(window=10, min_periods=10).mean()
    df["ma20"] = df["close"].rolling(window=20, min_periods=20).mean()

    # 이동평균 대비 현재가 비율
    df["close_ma5_ratio"] = df["close"] / df["ma5"] - 1
    df["close_ma10_ratio"] = df["close"] / df["ma10"] - 1
    df["close_ma20_ratio"] = df["close"] / df["ma20"] - 1

    # -------------------------
    # 3. 변동성
    # -------------------------
    df["volatility_5"] = df["return_1"].rolling(window=5, min_periods=5).std()
    df["volatility_10"] = df["return_1"].rolling(window=10, min_periods=10).std()

    # -------------------------
    # 4. 거래량 관련
    # -------------------------
    df["volume_ma5"] = df["volume"].rolling(window=5, min_periods=5).mean()
    df["volume_ma20"] = df["volume"].rolling(window=20, min_periods=20).mean()

    df["volume_ratio_ma5"] = df["volume"] / df["volume_ma5"]
    df["volume_ratio_ma20"] = df["volume"] / df["volume_ma20"]

    # -------------------------
    # 5. 캔들 구조
    # -------------------------
    df["body"] = df["close"] - df["open"]
    df["body_abs"] = df["body"].abs()

    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

    df["candle_range"] = df["high"] - df["low"]
    df["body_ratio"] = df["body_abs"] / df["candle_range"].replace(0, np.nan)

    # 양봉 / 음봉 플래그
    df["is_bullish"] = (df["close"] > df["open"]).astype(int)
    df["is_bearish"] = (df["close"] < df["open"]).astype(int)

    # -------------------------
    # 6. RSI
    # -------------------------
    df["rsi14"] = calculate_rsi(df["close"], period=14)

    # -------------------------
    # 7. 미래 수익률 / 라벨
    # -------------------------
    # 현재 close 대비 5분 뒤 close 수익률
    df["future_return_5"] = df["close"].shift(-5) / df["close"] - 1

    # 5분 뒤 0.2% 이상 상승하면 1, 아니면 0
    df["target_up_5m"] = (df["future_return_5"] > 0.002).astype(int)

    return df


# =========================
# 검증/요약
# =========================
def validate_feature_data(df: pd.DataFrame) -> None:
    """
    feature 생성 결과 요약 출력
    """
    print("[INFO] feature 생성 결과 요약")
    print(f" - 총 행 수: {len(df)}")
    print(f" - 시작 시각: {df['datetime'].min()}")
    print(f" - 종료 시각: {df['datetime'].max()}")

    check_columns = [
        "return_1",
        "ma5",
        "ma10",
        "ma20",
        "volatility_5",
        "volume_ma5",
        "body_ratio",
        "rsi14",
        "future_return_5",
        "target_up_5m",
    ]

    for col in check_columns:
        if col in df.columns:
            null_count = df[col].isna().sum()
            print(f" - {col} 결측 수: {null_count}")


# =========================
# 저장
# =========================
def save_feature_data(df: pd.DataFrame, file_path: Path = OUTPUT_FILE_PATH) -> Path:
    """
    feature CSV 저장
    """
    df.to_csv(file_path, index=False, encoding="utf-8-sig")
    return file_path


# =========================
# 실행
# =========================
def main() -> None:
    try:
        print(f"[INFO] processed 데이터 로드: {INPUT_FILE_PATH}")
        df = load_processed_data(INPUT_FILE_PATH)

        print("[INFO] feature 생성 시작")
        feature_df = add_features(df)

        validate_feature_data(feature_df)

        saved_path = save_feature_data(feature_df, OUTPUT_FILE_PATH)
        print(f"[OK] feature 생성 완료 및 저장: {saved_path}")

    except Exception as e:
        print(f"[ERROR] features 생성 실패: {e}")


if __name__ == "__main__":
    main()