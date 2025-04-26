# -*- coding: utf-8 -*-
"""
Module Name: pipeline_step.py
Description: Run a stage of the pipeline as parallel tasks

Copyright (C) 2025 J.Cincotta

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

"""


import concurrent.futures
import time
from queue import Queue
from welfareobs.handlers.abstract_handler import AbstractHandler
from welfareobs.utils.performance_monitor import PerformanceMonitor


class PipelineStep(object):
    """
    Pipeline step is a threadpool for a single step. We don't go as far as building a dependency graph of all the
    steps since 1. they should finish close in time to each other and 2. as soon as one step depends on aggregating
    inputs of the previous steps, we end up with exactly the same blocking/performance.
    """
    def __init__(self,
                 label: str,
                 performance_history_size: int = 1,
                 thread_pool_size: int = 5,
                 ):
        self.__threadpool_size:int = thread_pool_size
        self.__jobs: [AbstractHandler] = []
        self.__label:str = label
        self.__performance_monitor: PerformanceMonitor = PerformanceMonitor(
            label=label,
            history_size=performance_history_size
        )

    @property
    def label(self) -> str:
        return self.__label

    @property
    def performance(self) -> PerformanceMonitor:
        return self.__performance_monitor

    def add_job(self, job: AbstractHandler):
        self.__jobs.append(job)

    @property
    def jobs(self) -> [AbstractHandler]:
        return self.__jobs

    def run(self):
        self.__performance_monitor.track_start()
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.__threadpool_size) as executor:
            futures = []
            for job in self.__jobs:
                futures.append(executor.submit(job.run))
            while any(not f.done() for f in futures):
                pass
        self.__performance_monitor.track_end()
        print(str(self.__performance_monitor))
