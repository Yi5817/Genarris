"""
Genarris: A random molecular crystal structure generator.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("gnrs")
except PackageNotFoundError:
    __version__ = "3.1.1"