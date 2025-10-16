"""
This module provides functions for printing messages and configuration settings.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

__author__ = ["Yi Yang", "Rithwik Tom"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

import os
import socket
import logging
import subprocess
import textwrap
from datetime import datetime

import numpy as np
from mpi4py import MPI

from gnrs.output.templates import ascii_art, citation_v2, citation_v3, pymove

width = 100
logger = logging.getLogger("genarris")
wrapper = textwrap.TextWrapper(
    width=width, subsequent_indent=4 * " ", break_long_words=True
)
size = 0
rank = 0
is_master = False


def init_output(comm: MPI.Comm) -> None:
    """
    Initialize output module.
    
    Args:
        comm: MPI communicator object
    """
    logger.info("Initializing Genarris output")
    global size, rank, is_master
    size = comm.Get_size()
    rank = comm.Get_rank()
    is_master = rank == 0


def welcome_message() -> None:
    """
    Display welcome message.
    """
    if not is_master:
        return
        
    logger.debug("Printing Genarris startup message")
    double_separator()
    print(ascii_art)
    emit("If using Genarris 3.0+, please cite the following references:")
    print("\nGenarris 3.0:")
    print("-" * 50)
    print(citation_v3.strip())
    print("\nGenarris 2.0:")
    print("-" * 50)
    print(citation_v2.strip())
    print("\nPYMOVE:")
    print("-" * 50)
    print(pymove.strip())
    print("")
    double_separator()
    
    # System information
    current_time = datetime.now().strftime("%m/%d/%Y %I:%M:%S %p")
    hostname = socket.gethostname()
    
    emit(f"Welcome to Genarris 3.0")
    emit(f"")
    emit(f"Date and Time: {current_time}")
    emit(f"Host Machine: {hostname}")
    emit(f"Using {size} parallel tasks.")
    
    # Version information
    install_location = os.path.dirname(os.path.dirname(__file__))
    git_hash = get_git_revision_hash(install_location)
    
    emit(f"")
    emit(f"Version Information:")
    emit(f"Git Revision: {git_hash}")
    emit(f"Installation Location: {install_location}")
    emit(f"")
    
    logger.info(f"Genarris started on {hostname} with {size} processes")
    logger.info(f"Git Rev Hash: {git_hash}")


def get_git_revision_hash(install_location: str) -> str:
    """
    Get the git revision hash for the installation.
    """
    try:
        gh = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], 
            cwd=install_location,
            stderr=subprocess.DEVNULL
        )
        return gh.decode("ascii").strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "Unable to retrieve git hash: git not installed?"


def print_configs(settings_dict: dict) -> None:
    """
    Print config settings in a formatted way.
    """
    if not is_master:
        return
        
    logger.info("Printing configs")
    emit("")
    emit("Printing settings to be used for this Genarris run...")
    half_separator()
    
    for section, options in settings_dict.items():
        emit("")
        emit(f"[{section}]")
        if options:
            for option, option_val in options.items():
                emit(f"{option:<30} =  {option_val}")

    half_separator()


def print_dict_table(
    dct: dict | None, 
    header: str | None = None, 
    skip: tuple = ()
) -> None:
    """
    Print dictionary as a formatted table.
    """
    if not is_master or dct is None:
        return
        
    emit("")
    if header is not None:
        emit(f"{header[0]:^30}  |  {header[1]}")

    three_quarter_separator()
    
    for key, value in dct.items():
        if key in skip:
            continue

        key_str = str(key)
        if isinstance(value, (float, np.floating)):
            value_str = f"{value:.2f}"
        else:
            value_str = str(value)
            
        emit(f"{key_str:<30}  | {value_str:<60}")

    three_quarter_separator()
    emit("")


def section_complete() -> None:
    if is_master:
        emit("")
        double_separator()


def print_title(title: str) -> None:
    if not is_master:
        return
        
    title = title.upper()
    print("")
    current_time = datetime.now().strftime("%m/%d/%Y %I:%M:%S %p")
    single_separator()
    print(f"[{current_time:<10}] {title:^45}")
    single_separator()
    print("")


def print_sub_section(section_name: str) -> None:
    emit("")
    emit(section_name.upper())
    emit("-" * len(section_name))
    emit("")


def skip_task(task_name: str) -> None:
    print_title(f"skipping {task_name}")


def emit(message: str) -> None:
    if is_master:
        print(wrapper.fill(message), flush=True)


def single_separator() -> None:
    emit(width * "-")


def double_separator() -> None:
    emit(width * "=" + "\n")


def half_separator() -> None:
    emit(int(width / 2) * "-")


def three_quarter_separator() -> None:
    emit(int(0.75 * width) * "-")
