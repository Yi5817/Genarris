"""
This module provides functionality for parsing user settings from config files.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

__author__ = ["Yi Yang", "Rithwik Tom"]
__email__ = "yiy5@andrew.cmu.edu"
__group__ = "https://www.noamarom.com/"

import os
import logging
import json
import yaml
from ast import literal_eval
from configparser import ConfigParser
import importlib.resources as pkg_resources
from gnrs.output import emit

logger = logging.getLogger("parser")


class UserSettingsParser:
    """
    Parser for loading and validating user settings from config files.
    """

    def __init__(self, config_path: str) -> None:
        """
        Initialize parser with path to config file.

        Args:
            config_path: Path to configuration file
        """
        logger.info("Parsing Genarris input file")
        self.config_path = config_path
        self.config = {}
        self.defaults_dict = {}

        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"Input configuration file was not found: {self.config_path}"
            )

    def load_config(self) -> dict:
        """
        Load and validate user settings from config file.
        
        Returns:
            Dictionary containing user settings.
        """
        # Load from conf/ini file
        if self.config_path.endswith(("conf", "ini")):
            logger.info("Reading input from conf/ini file") 
            self.config = self.load_settings_config_parser()
            
        # Load from json file
        elif self.config_path.endswith("json"):
            logger.info("Reading input from json file")
            self.config = self.load_settings_from_json()
            
        # Load from yaml file
        elif self.config_path.endswith(("yaml", "yml")):
            logger.info("Reading input from yaml file")
            self.config = self.load_settings_from_yaml()
            
        else:
            raise ValueError(
                "Genarris supports only json/yaml/config file. "
                "File extension should be json/yaml/yml/conf/ini"
            )

        self.update_settings_with_defaults()
        return self.config

    def load_settings_config_parser(self) -> dict:
        """
        Load settings from conf/ini file using ConfigParser.
        
        Returns:
            Dictionary containing parsed settings
        """
        settings_dict: dict = {}
        config = ConfigParser()
        # Read config
        with open(self.config_path, "r") as config_file:
            config.read_file(config_file)
            
        # Convert to dict, handling type conversion
        for section in config.sections():
            section_dict = {}
            for option in config.options(section):
                option_val = config.get(section, option)
                try:
                    section_dict[option] = literal_eval(option_val)
                except (SyntaxError, ValueError):
                    section_dict[option] = option_val
            settings_dict[section] = section_dict

        return settings_dict

    def load_settings_from_json(self) -> dict:
        """
        Load settings from JSON file.
        
        Returns:
            Dictionary containing parsed settings
        """
        with open(self.config_path, "r") as config_file:
            return json.load(config_file)
            
    def load_settings_from_yaml(self) -> dict:
        """
        Load settings from YAML file.
        
        Returns:
            Dictionary containing parsed settings
        """
        with open(self.config_path, "r") as config_file:
            return yaml.safe_load(config_file)

    def _load_defaults(self) -> dict:
        """
        Load default settings from package defaults.json.
        
        Returns:
            Dictionary containing default settings
        """
        logger.info("Loading defaults.json")
        with pkg_resources.path("gnrs.parser", "defaults.json") as de_file:
            defaults = de_file.read_text()
        return json.loads(defaults)

    def update_settings_with_defaults(self) -> None:
        """
        Update user settings with default values where not specified.
        """
        logger.info("Adding defaults to user settings")
        emit("Setting defaults...")
        self.defaults_dict = self._load_defaults()

        for section, default_options in self.defaults_dict.items():
            user_options = self.config.get(section, {})
            
            if not user_options:
                continue

            for option, default_value in default_options.items():
                if option not in user_options and default_value != "required":
                    self.config[section][option] = default_value
                    emit(
                        f"Defaulting option '{option}' in section '{section}' "
                        f"to '{default_value}'."
                    )

    def _check_for_unavailable_options(self) -> None:
        """
        Check for any options in user settings that aren't in defaults.
        """
        for section, user_options in self.config.items():
            if section not in self.defaults_dict:
                continue
            
            default_options = self.defaults_dict[section]
            for option in user_options:
                if option not in default_options:
                    logger.error(f"{option} is not an available option")
                    raise ValueError("Unavailable option")
