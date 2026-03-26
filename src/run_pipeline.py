from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_step(script_path: str) -> None:
    full_path = PROJECT_ROOT / script_path

    print(f"\n[RUN] {script_path}")
    result = subprocess.run(
        [sys.executable, str(full_path)],
        cwd=PROJECT_ROOT,
        check=True,
    )

    if result.returncode == 0:
        print(f"[OK] {script_path} 완료")


def main() -> None:
    steps = [
        "src/collect_data.py",
        "src/preprocess.py",
        "src/features.py",
        "src/strategy.py",
    ]

    try:
        for step in steps:
            run_step(step)

        print("\n==========")
        print("[SUCCESS] 전체 파이프라인 완료")
        print("==========")

    except subprocess.CalledProcessError as e:
        print("\n==========")
        print(f"[ERROR] 실행 실패: {e}")
        print("==========")


if __name__ == "__main__":
    main()
