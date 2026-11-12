import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import uuid

module_path = Path(__file__).resolve().parents[1] / 'controller.py'
spec = importlib.util.spec_from_file_location('dflash_queue', module_path)
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


class QueueTests(unittest.TestCase):
    def test_waits_for_pending_throughput_and_each_sql_method(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = dict(throughput_queue=str(root / 'throughput'), sql_suite=str(root / 'sql'))
            write(root / 'throughput/status.json', {'stage': 'running'})
            self.assertFalse(queue.dependencies(plan, 5)[0])
            write(root / 'throughput/status.json', {'stage': 'complete'})
            write(root / 'sql/dispatch.json', {'bindings': {'ouro-2.6b/t0': 5}})
            write(root / 'sql/protocol.json', {'main_arms': ['ar', 'dflash']})
            write(root / 'sql/runs/ouro-2.6b-ar-t0/summary.json',
                  {'status': 'complete', 'completed_requests': 100})
            self.assertFalse(queue.dependencies(plan, 5)[0])
            path = root / 'sql/runs/ouro-2.6b-dflash-t0/summary.json'
            write(path, {'status': 'complete', 'completed_requests': 99})
            self.assertFalse(queue.dependencies(plan, 5)[0])
            write(path, {'status': 'complete', 'completed_requests': 100})
            self.assertTrue(queue.dependencies(plan, 5)[0])

    def test_gpu_lock_excludes_other_queues_and_survives_parent_handle_close(self):
        name = 'test-' + uuid.uuid4().hex
        handle = queue.try_lock(name)
        self.assertIsNotNone(handle)
        self.assertIsNone(queue.try_lock(name))
        program = ('import fcntl,os,sys; fd=int(sys.argv[1]); '
                   'fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB); '
                   'print("locked",flush=True); sys.stdin.readline()')
        child = subprocess.Popen([sys.executable, '-c', program, str(handle.fileno())],
                                 pass_fds=(handle.fileno(),), stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(child.stdout.readline().strip(), 'locked')
            handle.close()
            self.assertIsNone(queue.try_lock(name))
            child.communicate('\n', timeout=5)
            self.assertEqual(child.returncode, 0)
            released = queue.try_lock(name)
            self.assertIsNotNone(released)
            released.close()
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
            handle.close()
            Path(f'/tmp/loopfill-{name}.lock').unlink()

    def test_training_keeps_global_batch_and_uses_explicit_config(self):
        plan = dict(train_python='python', target_path='/target', draft_config='/draft.json')
        argv = queue.train_command(plan, Path('/out'), Path('/teachers'))
        self.assertEqual(int(argv[argv.index('--micro-batch') + 1]) *
                         int(argv[argv.index('--accumulation') + 1]), 32)
        self.assertEqual(argv[argv.index('--draft-config') + 1], '/draft.json')
        self.assertEqual(argv[argv.index('--epochs') + 1], '6')
        self.assertNotIn('--max-updates', argv)


if __name__ == '__main__':
    unittest.main()
