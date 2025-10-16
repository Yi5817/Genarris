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

from mpi4py import MPI


class GenarrisLogger:
    """
    Sets up the logging.
    """

    def __init__(self, comm: MPI.Comm, level: str = "DEBUG", parallel_log: str = "redirect_errors") -> None:
        """
        Initialize the logger.

        Args:
            comm: MPI communicator
            level: Log level
            parallel_log: Parallel log mode
        """
        self.log_level = getattr(logging, level.upper())
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.parallel_log = parallel_log
        self._configure()
        self.logger = logging.getLogger("genarris")
        self._welcome()
        return

    def _configure(self) -> None:
        """
        Configure the logger.
        """
        if self.parallel_log == "slave_log":
            slave_logfile = "Genarris_slave.log"
            slave_loglevel = logging.DEBUG
        elif self.parallel_log == "supress":
            slave_logfile = "/dev/null"
            slave_loglevel = logging.ERROR
        elif self.parallel_log == "redirect_errors":
            slave_logfile = "Genarris.log"
            slave_loglevel = logging.ERROR
        else:
            print("logging method not implemented!")
            raise RuntimeError

        if self.rank == 0:
            logging.basicConfig(
                filename="Genarris.log",
                level=self.log_level,
                format="%(asctime)s: %(levelname)5s: " "%(name)15s- %(message)s",
                datefmt="%b %d %I:%M:%S %p",
            )
        else:
            logging.basicConfig(
                filename=slave_logfile,
                level=slave_loglevel,
                format=f"Genarris slave process {self.rank} -"
                "%(asctime)s: %(levelname)5s: "
                "%(name)15s- %(message)s",
                datefmt="%b %d %I:%M:%S %p",
            )

        return

    def _welcome(self) -> None:
        """
        Welcome message.
        """
        self.logger.info(10 * "xx" + "  STARTING GENARRIS  " + 10 * "xx")
        self.logger.info("Initializing Genarris Logger")
        self.logger.info(f"Launching Genarris on {self.size} process(es)")
        return

    def reset_loglevel(self, level: str) -> None:
        """
        Reset the log level.

        Args:
            level: Log level
        """
        self.logger.info(f"Setting new log level: {level}")
        self.log_level = getattr(logging, level.upper())
        logging.getLogger().setLevel(self.log_level)
