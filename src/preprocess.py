from __future__ import annotations

from pathlib import Path

import pandas as pd


# =========================
# 경로 설정
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

RAW_FILE_PATH = RAW_DIR / "krw_btc_minute1.csv"
PROCESSED_FILE_PATH = PROCESSED_DIR / "krw_btc_minute1_processed.csv"


# =========================
# 데이터 로드
# =========================
def load_raw_data(file_path: Path = RAW_FILE_PATH) -> pd.DataFrame:
    """
    raw CSV 파일을 불러온다.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"raw 파일이 존재하지 않습니다: {file_path}")

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError(f"raw 파일이 비어 있습니다: {file_path}")

    return df


# =========================
# 전처리
# =========================
def preprocess_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    BTC 1분봉 OHLCV 데이터 기본 전처리
    """
    df = df.copy()

    # 1. datetime 변환
    if "datetime" not in df.columns:
        raise KeyError("'datetime' 컬럼이 없습니다.")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    # 2. 필수 컬럼 존재 여부 확인
    required_columns = ["open", "high", "low", "close", "volume"]
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"필수 컬럼이 없습니다: {col}")

    # 3. 숫자형 변환
    numeric_columns = ["open", "high", "low", "close", "volume", "value"]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. datetime 결측 제거
    df = df.dropna(subset=["datetime"])

    # 5. 시간순 정렬
    df = df.sort_values("datetime").reset_index(drop=True)

    # 6. 중복 제거
    # 보통 1분봉은 datetime이 같으면 같은 캔들로 간주
    df = df.drop_duplicates(subset=["datetime"], keep="last").reset_index(drop=True)

    # 7. 필수 OHLCV 결측 제거
    df = df.dropna(subset=required_columns).reset_index(drop=True)

    # 8. ticker 컬럼 없으면 기본값 추가
    if "ticker" not in df.columns:
        df["ticker"] = "KRW-BTC"

    # 9. 컬럼 순서 정리
    preferred_order = [
        "datetime",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "value",
    ]
    existing_columns = [col for col in preferred_order if col in df.columns]
    remaining_columns = [col for col in df.columns if col not in existing_columns]
    df = df[existing_columns + remaining_columns]

    return df


# =========================
# 검증/요약
# =========================
def validate_preprocessed_data(df: pd.DataFrame) -> None:
    """
    전처리된 데이터의 기본 상태를 출력한다.
    """
    print("[INFO] 전처리 결과 요약")
    print(f" - 총 행 수: {len(df)}")
    print(f" - 시작 시각: {df['datetime'].min()}")
    print(f" - 종료 시각: {df['datetime'].max()}")

    duplicated_count = df.duplicated(subset=["datetime"]).sum()
    print(f" - datetime 중복 개수: {duplicated_count}")

    missing_count = df[["open", "high", "low", "close", "volume"]].isna().sum().sum()
    print(f" - OHLCV 결측 개수 합계: {missing_count}")

    is_sorted = df["datetime"].is_monotonic_increasing
    print(f" - 시간 오름차순 정렬 여부: {is_sorted}")


# =========================
# 저장
# =========================
def save_processed_data(
    df: pd.DataFrame,
    file_path: Path = PROCESSED_FILE_PATH,
) -> Path:
    """
    processed CSV 저장
    """
    df.to_csv(file_path, index=False, encoding="utf-8-sig")
    return file_path


# =========================
# 실행
# =========================
def main() -> None:
    try:
        print(f"[INFO] raw 데이터 로드: {RAW_FILE_PATH}")
        raw_df = load_raw_data(RAW_FILE_PATH)

        print("[INFO] 전처리 시작")
        processed_df = preprocess_ohlcv(raw_df)

        validate_preprocessed_data(processed_df)

        saved_path = save_processed_data(processed_df, PROCESSED_FILE_PATH)
        print(f"[OK] 전처리 완료 및 저장: {saved_path}")

    except Exception as e:
        print(f"[ERROR] preprocess 실패: {e}")


if __name__ == "__main__":
    main()