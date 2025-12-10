"""Helpers for enforcing per-move time limits."""

import time


def deadline_after(seconds):
    return time.time() + seconds


def time_remaining(deadline):
    return deadline - time.time()
