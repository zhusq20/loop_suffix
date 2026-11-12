"""Run the root-level baselines package directly from this checkout."""
from pathlib import Path
import sys


def main():
    # Shared LoopFill executors live in this repository's existing src package.
    # The baseline implementations themselves live alongside this launcher.
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
    from baselines.__main__ import main as run
    run()


if __name__ == '__main__':
    main()
