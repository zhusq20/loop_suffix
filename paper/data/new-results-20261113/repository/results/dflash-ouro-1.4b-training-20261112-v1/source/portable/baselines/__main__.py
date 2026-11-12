"""Command dispatch without importing external experiment repositories."""
import argparse
import importlib
import sys

COMMANDS = {'train-dflash': 'train_dflash', 'train-loopahead': 'train_loopahead',
            'extract-features': 'extract_features', 'infer': 'infer'}


def main():
    p = argparse.ArgumentParser(description='Standalone DFlash and LoopAhead training/inference')
    p.add_argument('command', choices=COMMANDS)
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        p.parse_args()
        return
    args = p.parse_args(sys.argv[1:2])
    importlib.import_module(f'.{COMMANDS[args.command]}', __package__).main(sys.argv[2:])


if __name__ == '__main__':
    main()
