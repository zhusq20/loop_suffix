import os
import sys

def deny_external_code(event, args):
    if event in ('open', 'os.listdir', 'os.scandir') and args and isinstance(args[0], (str, bytes)):
        path = os.fsdecode(args[0])
        if path.startswith('/ossfs/workspace/sz/') or path.startswith('/ossfs/workspace/loop_suffix/src/'):
            raise RuntimeError('Portability audit blocked external source access: ' + path)

sys.addaudithook(deny_external_code)
