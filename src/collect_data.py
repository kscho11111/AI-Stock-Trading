from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pandas as pd
import pyupbit


# =========================
# 기본 설정
# =========================
TICKER = "KRW-BTC"
INTERVAL = "minute1"

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

RAW_FILE_PATH = RAW_DIR / "krw_btc_minute1.csv"


# =========================
# 데이터 수집 함수
# =========================
def fetch_ohlcv(
    ticker: str = TICKER,
    interval: str = INTERVAL,
    count: int = 200,
    to: Optional[str] = None,
) -> pd.DataFrame:
    """
    Upbit에서 OHLCV 데이터를 가져온다.

    Args:
        ticker: 종목명 (기본: KRW-BTC)
        interval: 봉 단위 (기본: minute1)
        count: 가져올 캔들 개수
        to: 종료 시각 (예: '2026-03-25 21:00:00'), 없으면 최신 기준

    Returns:
        OHLCV DataFrame
    """
    df = pyupbit.get_ohlcv(
        ticker=ticker,
        interval=interval,
        count=count,
        to=to,
    )

    if df is None or df.empty:
        raise ValueError(f"OHLCV 데이터를 가져오지 못했습니다. ticker={ticker}")

    df = df.copy()
    df = df.reset_index().rename(columns={"index": "datetime"})
    df["ticker"] = ticker

    return df


def fetch_historical_ohlcv(
    ticker: str = TICKER,
    interval: str = INTERVAL,
    total_count: int = 1000,
    chunk_size: int = 200,
    sleep_sec: float = 0.15,
) -> pd.DataFrame:
    """
    Upbit에서 과거 데이터를 여러 번 나눠 받아 total_count개 정도 모은다.

    주의:
    - pyupbit.get_ohlcv는 한 번에 count 개수만큼만 가져온다.
    - 과거로 거슬러 올라가려면 `to`를 이전 데이터의 시작 시각으로 옮겨가며 반복 호출해야 한다.

    Args:
        ticker: 종목명
        interval: 봉 단위
        total_count: 목표 캔들 수
        chunk_size: 한 번 호출 시 가져올 개수 (보통 200 이하 권장)
        sleep_sec: 호출 간 대기 시간

    Returns:
        병합된 OHLCV DataFrame
    """
    if total_count <= 0:
        raise ValueError("total_count는 1 이상이어야 합니다.")

    dfs: list[pd.DataFrame] = []
    fetched = 0
    to: Optional[str] = None

    while fetched < total_count:
        current_count = min(chunk_size, total_count - fetched)

        df = fetch_ohlcv(
            ticker=ticker,
            interval=interval,
            count=current_count,
            to=to,
        )

        if df.empty:
            break

        dfs.append(df)
        fetched += len(df)

        # 다음 요청은 현재 받아온 데이터의 가장 이른 시점 이전으로 이동
        earliest_dt = pd.to_datetime(df["datetime"].min())
        to = earliest_dt.strftime("%Y-%m-%d %H:%M:%S")

        time.sleep(sleep_sec)

        # 혹시 같은 데이터만 반복 수집되는 비정상 상황 방지
        if len(df) < current_count:
            break

    if not dfs:
        raise ValueError("과거 데이터를 하나도 수집하지 못했습니다.")

    result = pd.concat(dfs, ignore_index=True)
    result["datetime"] = pd.to_datetime(result["datetime"])
    result = result.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last")
    result = result.reset_index(drop=True)

    return result


# =========================
# 파일 입출력 함수
# =========================
def load_existing_raw(file_path: Path = RAW_FILE_PATH) -> pd.DataFrame:
    """
    기존 raw CSV가 있으면 불러오고, 없으면 빈 DataFrame 반환
    """
    if not file_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(file_path)
    if df.empty:
        return df

    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def merge_and_deduplicate(
    old_df: pd.DataFrame,
    new_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    기존 데이터와 신규 데이터를 합쳐서 중복 제거
    """
    if old_df.empty:
        merged = new_df.copy()
    else:
        merged = pd.concat([old_df, new_df], ignore_index=True)

    merged["datetime"] = pd.to_datetime(merged["datetime"])
    merged = merged.sort_values("datetime")
    merged = merged.drop_duplicates(subset=["datetime"], keep="last")
    merged = merged.reset_index(drop=True)

    return merged


def save_raw_data(
    df: pd.DataFrame,
    file_path: Path = RAW_FILE_PATH,
) -> Path:
    """
    raw CSV 저장
    """
    save_df = df.copy()
    save_df.to_csv(file_path, index=False, encoding="utf-8-sig")
    return file_path


# =========================
# 실행용 함수
# =========================
def collect_and_save_latest(count: int = 500) -> Path:
    """
    최신 기준으로 count개 수집 후 저장
    기존 파일이 있으면 병합 후 저장
    """
    print(f"[INFO] 최신 {count}개 1분봉 수집 시작: {TICKER}")

    new_df = fetch_historical_ohlcv(
        ticker=TICKER,
        interval=INTERVAL,
        total_count=count,
        chunk_size=200,
        sleep_sec=0.15,
    )

    old_df = load_existing_raw(RAW_FILE_PATH)
    merged_df = merge_and_deduplicate(old_df, new_df)

    saved_path = save_raw_data(merged_df, RAW_FILE_PATH)

    print(f"[OK] 저장 완료: {saved_path}")
    print(f"[INFO] 총 행 수: {len(merged_df)}")
    print(f"[INFO] 시작 시각: {merged_df['datetime'].min()}")
    print(f"[INFO] 종료 시각: {merged_df['datetime'].max()}")

    return saved_path


def main() -> None:
    """
    기본 실행:
    KRW-BTC 1분봉 최신 1000개 수집 후 raw CSV 저장
    """
    try:
        collect_and_save_latest(count=1000)
    except Exception as e:
        print(f"[ERROR] 데이터 수집 실패: {e}")


if __name__ == "__main__":
    main()