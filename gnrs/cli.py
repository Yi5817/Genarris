"""
This module provides the command line interface for Genarris.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

__author__ = ["Yi Yang", "Rithwik Tom"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"
import logging
from mpi4py import MPI

import gnrs.output as gout
from gnrs.workflow import Genarris

import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def main():
    """
    Main CLI for Genarris workflow.
    """
    parser = argparse.ArgumentParser(description="Genarris3.0")
    parser.add_argument("-c", "--config", required=True, type=str, help="Path to the configuration file")
    parser.add_argument("--restart", action="store_true", help="Restart Genarris with previous config file")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    logger = logging.getLogger("genarris")
    try:
        # Initialize and run Genarris
        gnrs_workflow = Genarris(args)
        gnrs_workflow.run()
        gout.emit("All tasks completed successfully.\nHave a nice day! :-)")
    except KeyboardInterrupt:
        logger.warning("Genarris interrupted by user")
        gout.emit("\n***Genarris was interrupted by user***")
        comm.Abort(130)
    except Exception:
        logger.exception("Genarris exiting due to error")
        gout.emit("\n***Genarris has abruptly exited. Please see the log for Traceback***")
        comm.Abort(1)
    finally:
        comm.Barrier()
        MPI.Finalize()

if __name__ == "__main__":
    main()