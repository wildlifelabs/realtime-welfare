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


class PipelineStep(object):
    """
    Pipeline step is a threadpool for a single step. We don't go as far as building a dependency graph of all the
    steps since 1. they should finish close in time to each other and 2. as soon as one step depends on aggregating
    inputs of the previous steps, we end up with exactly the same blocking/performance.
    """
    THREAD_POOL_SIZE = 5

    def __init__(self):
        self.__jobs: [AbstractHandler] = []
        self.__last_execution_time = 0
        self.__overall_execution_time = 0
        self.__number_of_execution_runs = 0

    @property
    def last_execution_time(self) -> float:
        return self.__last_execution_time

    @property
    def overall_execution_time(self) -> float:
        return self.__overall_execution_time

    def add_job(self, job: AbstractHandler):
        self.__jobs.append(job)

    @property
    def jobs(self) -> [AbstractHandler]:
        return self.__jobs

    def run(self):
        start_time = time.time()
        job_queue: Queue = Queue()
        list(map(job_queue.put, self.__jobs))
        with concurrent.futures.ThreadPoolExecutor(max_workers=PipelineStep.THREAD_POOL_SIZE) as executor:
            futures = []
            finished_jobs = []
            while not job_queue.empty() or any(f.running() for f in futures):
                while not job_queue.empty() and len(futures) < PipelineStep.THREAD_POOL_SIZE:
                    job = job_queue.get()
                    finished_jobs.append(job)
                    future = executor.submit(job.run)
                    futures.append(future)
                futures = [f for f in futures if not f.done()]
            end_time = time.time()
            self.__last_execution_time = end_time - start_time
            self.__overall_execution_time += self.__last_execution_time
            self.__number_of_execution_runs += 1
