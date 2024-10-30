from __future__ import (
    annotations,
)
import logging
from abc import (
    ABC,
    abstractmethod,
)

import dpdata
import numpy as np
from typing import Dict
from pfd.exploration.inference import TestReports


class ConfFilterConv(ABC):
    __FilterTypes = {}

    @staticmethod
    def register(key: str):
        """Register a model interface. Used as decorators

        Args:
            key (str): key of the model
        """

        def decorator(object):
            ConfFilterConv.__FilterTypes[key] = object
            return object

        return decorator

    @staticmethod
    def get_filter(key: str):
        """Get a driver for ModelEval

        Args:
            key (str): _description_

        Raises:
            RuntimeError: _description_

        Returns:
            _type_: _description_
        """
        try:
            return ConfFilterConv.__FilterTypes[key]
        except KeyError as e:
            raise RuntimeError("unknown driver: " + key) from e

    @staticmethod
    def get_filters() -> dict:
        """Get all filters

        Returns:
            dict: all filters
        """
        return ConfFilterConv.__FilterTypes

    @abstractmethod
    def check(self, frame: dpdata.System, res: Dict) -> TestReports:
        """Check if the configuration is valid based on model test.

        Parameters
        ----------
        frame : dpdata.System
            A dpdata.System containing a single frame

        res: Dict
            A dict which stores the predition error for the given configuration

        Returns
        -------
        valid : bool
            `True` if the configuration is a valid configuration, else `False`.

        """
        pass


class ConfFiltersConv:
    """A list of ConfFilters"""

    def __init__(
        self,
    ):
        self._filters = []

    def add(
        self,
        conf_filter: ConfFilterConv,
    ) -> ConfFiltersConv:
        self._filters.append(conf_filter)
        return self

    def check(self, reports: TestReports) -> TestReports:
        selected_idx = np.arange(len(reports))
        for ff in self._filters:
            fsel = np.where([ff.check(reports[ii]) for ii in range(len(reports))])[0]
            selected_idx = np.intersect1d(selected_idx, fsel)
        sub_reports = reports.sub_reports(selected_idx)
        logging.info("#### %d systems are added to the training set" % len(sub_reports))
        return sub_reports
