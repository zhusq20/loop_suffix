"""Cooperative model operations shared by single-request and batched decoding."""

from dataclasses import dataclass


@dataclass
class ModelCall:
    name: str
    args: tuple = ()


def run_single(backbone, steps):
    """Drive exactly the same state machine without a batch scheduler."""
    result = None
    while True:
        try:
            call = steps.send(result)
        except StopIteration as done:
            return done.value
        result = getattr(backbone, call.name)(*call.args)
