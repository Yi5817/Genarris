import logging

from gnrs.output import emit

logger = logging.getLogger("sanity_check")


class UserSettingsSanityChecker:
    def __init__(self, user_settings):
        logger.info("Performing input sanity checks")
        emit("")
        emit("Performing checks on user input...")
        self.user_settings = user_settings

        self._check_required_sections()
        self._check_required_options()
        self._check_optionvalue_types()
        self._check_unavailable_options()

        emit("All checks passed. Input OK.")

        return

    def _check_required_sections(self):
        """
        See if all required sections are present in settings dict
        """
        logger.debug("Performing required sections check")
        return

    def _check_required_options(self):
        """
        See if all required options are present
        """
        logger.debug("Performing required options check")
        return

    def _check_optionvalue_types(self):
        """
        See if values of all options are required types
        """
        logger.debug("Checking data types of options")
        return

    def _check_unavailable_options(self):
        """
        See if extra option not supported by Genarris are present
        """
        logger.debug("Checking if unsupported options are present")
        return
