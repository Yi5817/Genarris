"""
This module provides testing utilities for the parallel module.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

__author__ = ["Yi Yang", "Rithwik Tom"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

import gnrs.parallel as gp


def test_bcast() -> None:
    """
    Simple test of MPI broadcast function.
    """
    if gp.is_master:
        test_var = 999
    else:
        test_var = None

    test_var = gp.comm.bcast(test_var, root=0)
    
    try:
        assert test_var == 999
    except AssertionError:
        print(f"MPI broadcast test failed: test_var = {test_var}", flush=True)
        raise
