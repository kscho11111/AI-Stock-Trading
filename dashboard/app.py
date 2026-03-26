from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# =========================
# 경로 설정
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIGNAL_FILE_PATH = PROJECT_ROOT / "data" / "processed" / "krw_btc_minute1_signals.csv"


# =========================
# Streamlit 기본 설정
# =========================
st.set_page_config(
    page_title="BTC Signal Dashboard",
    layout="wide",
)


# =========================
# 데이터 로드
# =========================
@st.cache_data(ttl=30)
def load_signal_data(file_path: Path = SIGNAL_FILE_PATH) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"signals 파일이 존재하지 않습니다: {file_path}")

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError(f"signals 파일이 비어 있습니다: {file_path}")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    return df


# =========================
# 차트 생성
# =========================
def make_price_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    # 캔들 차트
    fig.add_trace(
        go.Candlestick(
            x=df["datetime"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="BTC 1min",
        )
    )

    # 이동평균선
    if "ma5" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["datetime"],
                y=df["ma5"],
                mode="lines",
                name="MA5",
            )
        )

    if "ma20" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["datetime"],
                y=df["ma20"],
                mode="lines",
                name="MA20",
            )
        )

    # 매수 / 매도 포인트
    if "signal" in df.columns:
        buy_df = df[df["signal"] == "BUY"]
        sell_df = df[df["signal"] == "SELL"]

        if not buy_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_df["datetime"],
                    y=buy_df["close"],
                    mode="markers",
                    name="BUY",
                    marker=dict(symbol="triangle-up", size=10),
                )
            )

        if not sell_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_df["datetime"],
                    y=sell_df["close"],
                    mode="markers",
                    name="SELL",
                    marker=dict(symbol="triangle-down", size=10),
                )
            )

    fig.update_layout(
        title="KRW-BTC 1분봉 패턴 기반 신호 대시보드",
        xaxis_title="Datetime",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=700,
    )

    return fig


# =========================
# 메인 앱
# =========================
def main() -> None:
    st.title("KRW-BTC Pattern Signal Dashboard")
    st.write("Phase 1 - BTC 1분봉 패턴 기반 방향성 분석")

    st.sidebar.header("설정")
    candle_count = st.sidebar.slider(
        "최근 몇 개 캔들을 볼지",
        min_value=50,
        max_value=500,
        value=200,
        step=10,
    )

    if st.sidebar.button("새로고침"):
        st.cache_data.clear()

    try:
        df = load_signal_data(SIGNAL_FILE_PATH)
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return

    if df.empty:
        st.warning("표시할 데이터가 없습니다.")
        return

    # 최근 일부만 보기
    display_df = df.tail(candle_count).copy()

    latest = display_df.iloc[-1]

    st.subheader("현재 상태")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("현재 종가", f"{latest['close']:,.0f}")
    col2.metric("현재 신호", str(latest.get("signal", "N/A")))
    col3.metric(
        "패턴 점수",
        f"{latest['pattern_score']:.6f}" if pd.notna(latest.get("pattern_score")) else "N/A",
    )
    col4.metric(
        "패턴 승률",
        f"{latest['pattern_win_rate']:.2%}" if pd.notna(latest.get("pattern_win_rate")) else "N/A",
    )
    col5.metric(
        "평균 거리",
        f"{latest['pattern_avg_distance']:.6f}" if pd.notna(latest.get("pattern_avg_distance")) else "N/A",
    )

    st.subheader("가격 차트")
    fig = make_price_chart(display_df)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("최근 신호 이력")
    signal_cols = [
        "datetime",
        "close",
        "signal",
        "pattern_score",
        "pattern_win_rate",
        "pattern_avg_distance",
        "pattern_match_count",
    ]
    existing_signal_cols = [col for col in signal_cols if col in display_df.columns]

    recent_signal_df = display_df[display_df["signal"].isin(["BUY", "SELL", "HOLD"])][existing_signal_cols].tail(30)
    st.dataframe(recent_signal_df, use_container_width=True)

    st.subheader("최근 원본+feature 데이터")
    raw_cols = [
        "datetime",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ma5",
        "ma20",
        "rsi14",
        "signal",
    ]
    existing_raw_cols = [col for col in raw_cols if col in display_df.columns]
    st.dataframe(display_df[existing_raw_cols].tail(30), use_container_width=True)


if __name__ == "__main__":
    main()