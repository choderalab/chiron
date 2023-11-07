"""
Unit and regression test for the chiron package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import chiron


def test_chiron_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "chiron" in sys.modules
