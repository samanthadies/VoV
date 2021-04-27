""" Utilities for data validation
"""

import argparse

from datetime import date
from datetime import datetime


def check_date(s):
    """Helper function to validate date strings in argument parser.
    """
    try:
        date.fromisoformat(s)
        return s
    except ValueError:
        raise argparse.ArgumentTypeError("invalid data format, should be YYYY-MM-DD")


def check_datetime(s):
    """Helper function to validate date time strings in argument parser.
    """
    try:
        datetime.fromisoformat(s)
        return s
    except ValueError:
        raise argparse.ArgumentTypeError("invalid data format, should be YYYY-MM-DD or YYYY-MM-DD HH:MM:SS")
