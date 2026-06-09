#!/usr/bin/env python
"""Fail fast unless every CUDA device allocated to the job is an RTX 3090."""

import argparse
import sys

import torch


def validate_gpu_names(names, required_substring="RTX 3090"):
    if not names:
        return False, "No CUDA devices are visible to this job."
    mismatched = [name for name in names if required_substring not in name]
    if mismatched:
        return (
            False,
            f"Allocated GPU model is not allowed: {mismatched}; "
            f"required substring: {required_substring!r}.",
        )
    return True, f"GPU guard passed: {names}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--required", default="RTX 3090")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("RTX 3090 guard failed: CUDA is unavailable.", file=sys.stderr)
        return 42
    names = [
        torch.cuda.get_device_name(index)
        for index in range(torch.cuda.device_count())
    ]
    valid, message = validate_gpu_names(names, args.required)
    stream = sys.stdout if valid else sys.stderr
    print(message, file=stream)
    return 0 if valid else 42


if __name__ == "__main__":
    raise SystemExit(main())
