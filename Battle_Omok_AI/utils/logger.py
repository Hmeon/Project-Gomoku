"""Lightweight logging utilities for matches and debugging."""

import datetime


def log_event(message):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")
