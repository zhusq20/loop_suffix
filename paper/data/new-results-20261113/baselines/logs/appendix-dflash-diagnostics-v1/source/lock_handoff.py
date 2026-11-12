"""Pass an already acquired GPU reservation to an unchanged legacy evaluator."""
import fcntl
import os
from pathlib import Path
import runpy
import sys


def install_handoff(inherited, expected):
    original = fcntl.flock
    assert os.path.realpath(f'/proc/self/fd/{inherited}') == expected
    identity = os.fstat(inherited)

    def handoff(handle, operation):
        fd = handle if isinstance(handle, int) else handle.fileno()
        if os.path.realpath(f'/proc/self/fd/{fd}') != expected:
            return original(handle, operation)
        assert operation == fcntl.LOCK_EX | fcntl.LOCK_NB
        current = os.fstat(fd)
        assert (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino)
        # dup2 preserves the same open-file description and its flock lease.
        # The reservation never passes through an unlocked window.
        os.dup2(inherited, fd)
        original(fd, operation)
        fcntl.flock = original

    fcntl.flock = handoff


if __name__ == '__main__':
    target = Path(sys.argv[1]).resolve()
    inherited = int(os.environ.pop('LOOPFILL_RETRY_LOCK_FD'))
    expected = f"/tmp/loopfill-{os.environ['CUDA_VISIBLE_DEVICES']}.lock"
    install_handoff(inherited, expected)
    sys.argv = [str(target), *sys.argv[2:]]
    sys.path.insert(0, str(target.parent))
    runpy.run_path(str(target), run_name='__main__')
