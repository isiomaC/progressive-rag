#!/usr/bin/env python
"""Run the full experiment matrix (Phases 1-6 + challenges).

Usage:
    python scripts/run_experiments.py                    # offline mode
    python scripts/run_experiments.py --llm deepseek     # DeepSeek API
    python scripts/run_experiments.py --llm openai       # OpenAI API
    python scripts/run_experiments.py --limit 5          # quick sanity run
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragprog.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.argv = ["ragprog", "run", *sys.argv[1:]]
    sys.exit(main())
