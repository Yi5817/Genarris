"""
This module provides the command line interface for Genarris.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

__author__ = ["Yi Yang", "Rithwik Tom"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

import argparse
import logging
import sys
import traceback
import warnings

from mpi4py import MPI

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main():
    """
    Main CLI for Genarris workflow.
    """
    parser = argparse.ArgumentParser(description="Genarris3.0")
    parser.add_argument("-c", "--config", required=True, type=str, help="Path to the configuration file")
    parser.add_argument("-d", "--seed", type=int, help="Random seed", default=42)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--restart", action="store_true",
        help="Resume the previous run in the current directory, "
        "skipping completed tasks and structures",
    )
    mode.add_argument(
        "--overwrite", action="store_true",
        help="Start over in a directory that contains a previous run, "
        "discarding its progress",
    )
    args = parser.parse_args()

    from gnrs.core.restart import RestartError

    comm = MPI.COMM_WORLD
    logger = logging.getLogger("genarris")
    aborted = False
    exit_code = 0
    try:
        # Initialize and run Genarris
        from gnrs.workflow import Genarris
        import gnrs.output as gout
        gnrs_workflow = Genarris(args)
        gnrs_workflow.run()
        gout.emit("All tasks completed successfully.\nHave a nice day! :-)")
    except KeyboardInterrupt:
        logger.warning("Genarris interrupted by user")
        aborted = True
        comm.Abort(130)
    except RestartError as exc:
        logger.error(f"Restart failed: {exc}")
        gout.emit("")
        for line in f"ERROR: {exc}".splitlines():
            gout.emit(line)
        exit_code = 1
    except Exception as exc:
        logger.exception("Genarris exiting due to error")
        print(
            f"\nERROR: Genarris rank {comm.Get_rank()} failed with "
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            file=sys.stderr,
            flush=True,
        )
        aborted = True
        comm.Abort(1)
    finally:
        if not aborted:
            comm.Barrier()
            MPI.Finalize()
    if exit_code:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()