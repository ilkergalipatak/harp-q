#!/usr/bin/env python3
"""Capture the HARP-Q handoff hardware report."""

from __future__ import annotations

import argparse
import json

from harpq.hardware import collect_active_linux_report, collect_hardware_report, write_hardware_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="optional JSON destination")
    parser.add_argument("--active", action="store_true")
    parser.add_argument("--scratch", help="existing filesystem directory for active fio probe")
    args = parser.parse_args()
    if args.active and not args.scratch:
        parser.error("--active requires --scratch")
    if args.output:
        destination = write_hardware_report(
            args.output,
            active_scratch=args.scratch if args.active else None,
        )
        print(destination)
    else:
        report = (
            collect_active_linux_report(args.scratch)
            if args.active
            else collect_hardware_report()
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
