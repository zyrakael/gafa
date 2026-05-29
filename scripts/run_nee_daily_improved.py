#!/usr/bin/env python3
"""Compatibility entrypoint for the improved daily carbon batch run."""

from __future__ import annotations

from run_nee_daily import main


if __name__ == "__main__":
    raise SystemExit(main())
