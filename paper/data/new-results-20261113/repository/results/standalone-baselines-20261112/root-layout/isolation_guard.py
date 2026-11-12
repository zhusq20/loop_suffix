import os
import sys

def deny_external_source(event, args):
    if event in ('open', 'os.listdir', 'os.scandir') and args and isinstance(args[0], (str, bytes)):
        path = os.fsdecode(args[0])
        if path.startswith(('/ossfs/workspace/sz/', '/ossfs/workspace/loop_suffix/src/', '/ossfs/workspace/loop_suffix/baselines/', '/tmp/loop-baselines-root-layout-2q8j46qt/source/src/', '/tmp/loop-baselines-root-layout-2q8j46qt/source/baselines/')):
            raise RuntimeError('Blocked external source during package audit: ' + path)

sys.addaudithook(deny_external_source)
