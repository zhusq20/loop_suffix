"""Retain the existing GPU guard scope while extra GPUs run queued work."""
import sys
sys.path.insert(0, '/ossfs/workspace/sz/loopfill/experiments/baselines')
import guard_assigned_gpus as guard
original_allowed = guard.allowed_gpus
guard.allowed_gpus = lambda: [gpu for gpu in original_allowed() if gpu in [2, 3, 4, 6]]
guard.main()
