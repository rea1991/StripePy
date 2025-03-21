#!/usr/bin/env python3

# Copyright (C) 2025 Roberto Rossini <roberroso@uio.no>
#
# SPDX-License-Identifier: MIT


import argparse
import json
import logging
import pathlib
import sys

from stripepy.io import compare_result_files


def existing_file(arg: str) -> pathlib.Path:
    if (path := pathlib.Path(arg)).is_file():
        return path

    raise argparse.ArgumentTypeError(f'Not an existing file: "{arg}"')


def make_cli() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser("Script to check whether two result files generated by stripepy call are identical.")

    cli.add_argument(
        "hdf5",
        nargs=2,
        type=existing_file,
    )
    cli.add_argument(
        "--verbosity",
        type=str,
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Set verbosity of output to the console.",
    )

    cli.add_argument(
        "--print-traceback",
        action="store_true",
        default=False,
        help="Upon encountering an exception, print the traceback and exit immediately (mostly useful for debugging).",
    )

    return cli


def setup_logger(level: str):
    fmt = "[%(asctime)s] %(levelname)s: %(message)s"
    logging.basicConfig(level=level, format=fmt)
    logging.getLogger().setLevel(level)


def main() -> int:
    args = vars(make_cli().parse_args())
    setup_logger(args["verbosity"].upper())
    report = compare_result_files(*args["hdf5"], raise_on_exception=args["print_traceback"])
    if report["success"]:
        logging.debug("%s", json.dumps(report, indent=2))
        logging.info("### SUCCESS!")
        return 0

    logging.critical("%s", json.dumps(report, indent=2))
    logging.critical("### FAILURE")
    return 1


if __name__ == "__main__":
    sys.exit(main())
