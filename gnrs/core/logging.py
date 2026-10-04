"""
This module provides functions for logging.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

__author__ = ["Yi Yang", "Rithwik Tom"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

import logging
from collections import defaultdict
from logging.handlers import BufferingHandler

from mpi4py import MPI

logger = logging.getLogger(__name__)


class GenarrisLogger:
    """
    Sets up the logging.

    Rank 0 writes ``Genarris.log``. The other ranks hold their warnings and
    errors in memory until ``sync`` writes each distinct message one time.
    """

    def __init__(self, comm: MPI.Comm, level: str = "INFO") -> None:
        """
        Initialize the logger.

        Args:
            comm: MPI communicator
            level: Log level
        """
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        if self.rank == 0:
            self.handler = logging.FileHandler("Genarris.log")
            self.handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
            )
        else:
            self.handler = BufferingHandler(10_000)
            self.handler.setLevel(logging.WARNING)
        logging.getLogger().addHandler(self.handler)
        logging.captureWarnings(True)
        logging.getLogger("gnrs").setLevel(level)
        logger.info(10 * "xx" + "  STARTING GENARRIS  " + 10 * "xx")
        logger.info(f"Launching Genarris on {self.size} process(es)")

    def reset_loglevel(self, level: str) -> None:
        """
        Reset the log level.

        Args:
            level: Log level
        """
        logging.getLogger("gnrs").setLevel(level.upper())
        logger.info(f"Log level: {level}")

    def sync(self) -> None:
        """
        Write the records that ranks above 0 hold to the log file, one line
        per distinct message. Must be called by all ranks.
        """
        held = []
        if self.rank != 0:
            held = list(
                dict.fromkeys(
                    (r.levelno, r.name, self.handler.format(r))
                    for r in self.handler.buffer
                )
            )
            self.handler.flush()
        ranks = defaultdict(list)
        for rank, keys in enumerate(self.comm.gather(held, root=0) or []):
            for key in keys:
                ranks[key].append(rank)
        for (levelno, name, message), who in ranks.items():
            tag = f"rank {who[0]}" if len(who) == 1 else f"{len(who)} ranks"
            logging.getLogger(name).log(levelno, f"[{tag}] {message}")
