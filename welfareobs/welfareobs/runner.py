# -*- coding: utf-8 -*-
"""
Module Name: runner.py
Description: Run the pipeline (orchestration)

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
import threading

from welfareobs.utils.config import Config
from welfareobs.handlers.abstract_handler import AbstractHandler
from welfareobs.pipeline_step import PipelineStep
from welfareobs.utils.performance_monitor import PerformanceMonitor


class Runner(object):

    def __init__(self, config: Config):
        self.__config: Config = config
        self.__job_map: {str: AbstractHandler} = {}
        self.__pipeline_steps: [PipelineStep] = []
        self.__performance_history_size: int = 1
        self.__thread_pool_size: int = 5
        self.__pipeline_label: str = ""
        self.__last_execution_time = 0
        self.__number_of_execution_runs = 0
        self.__overall_execution_time = 0
        self.__validate()  # will raise SyntaxError on invalid config
        self.__parse()
        self.__performance_monitor: PerformanceMonitor = PerformanceMonitor(
            label=self.__pipeline_label,
            history_size=100
        )
        self.__loop = threading.Event()
        self.__loop.set()
        self.__hnd = None
        self.__has_setup = False
        self.__has_torndown = False

    @property
    def performance(self) -> PerformanceMonitor:
        return self.__performance_monitor

    def __setup(self):
        for job in self.__job_map.values():
            job.setup()
        self.__has_setup = True

    def __teardown(self):
        for job in self.__job_map.values():
            job.teardown()
        self.__has_torndown = True

    def run(self, run_count:None|int=None):
        if not self.__has_setup:
            self.__setup()
        if self.__has_torndown:
            raise RuntimeError("Already torn down")
        trigger: bool = True
        while trigger:
            self.__performance_monitor.track_start()
            for ps in self.__pipeline_steps:
                for job in ps.jobs:
                    src_jobs = [self.__job_map[o] for o in job.required_jobs_for_inputs()]
                    job.set_inputs([o.get_output() for o in src_jobs])
                ps.run()
            trigger = self.__loop.is_set()
            if run_count is not None:
                if run_count == 0:
                    trigger = False
                run_count -= 1
            self.__performance_monitor.track_end()
        self.__teardown()

    def run_once(self):
        self.__loop.clear()  # stop the event loop from running
        self.run()

    def run_async(self):
        if self.__hnd is not None:
            return
        self.__hnd = threading.Thread(target=self.run)
        self.__hnd.start()

    def stop_async(self):
        self.__loop.clear()  # stop the event loop from running gracefully
        self.__hnd.join()

    def __getitem__(self, item) -> AbstractHandler:
        return self.__job_map[item]

    def get_step(self, index) -> PipelineStep:
        return self.__pipeline_steps[index]

    def len_steps(self) -> int:
        return len(self.__pipeline_steps)

    def __parse(self):
        #
        # Parse the JSON into a pipeline of jobs
        #
        steps = self.__config.as_list("pipeline")
        self.__pipeline_label = self.__config.as_string("settings.configuration-name")
        self.__thread_pool_size = self.__config.as_int("settings.threadpool-size")
        self.__performance_history_size = self.__config.as_int("settings.performance-history-size")
        for step in steps:
            ps: PipelineStep = PipelineStep(
                label=step,
                performance_history_size=self.__performance_history_size,
                thread_pool_size=self.__thread_pool_size
            )
            tasks = self.__config[step]
            for task in tasks:
                job_hnd = self.__config.instance(f"{task}.handler")
                job = job_hnd(
                    task,
                    inputs=self.__config.as_list(f"{task}.input"),
                    param=self.__config[f"{task}.config"])
                self.__job_map[task] = job
                ps.add_job(job)
            self.__pipeline_steps.append(ps)

    def __validate(self):
        #
        # Does not validate input or config parameters, if these don't exist, it's up to the handler to die gracefully.
        #
        if not self.__config.as_bool("settings.configuration-name"):
            raise SyntaxError("Missing settings.configuration-name")
        if not self.__config.as_bool("settings.threadpool-size"):
            raise SyntaxError("Missing settings.threadpool-size")
        if not self.__config.as_bool("settings.performance-history-size"):
            raise SyntaxError("Missing settings.performance-history-size")
        steps = self.__config.as_list("pipeline")
        if len(steps) < 1:
            raise SyntaxError("`pipeline` element must exist and contain an array of at least one element")
        for step in steps:
            if not self.__config.as_bool(step):
                raise SyntaxError(f"`{step}` element must exist")
            tasks = self.__config.as_list(step)
            if len(tasks) < 1:
                raise SyntaxError(f"`{step}` element must exist and contain an array of at least one task")
            for task in tasks:
                if not self.__config.as_bool(task):
                    raise SyntaxError(f"`{task}` element must exist")
                if not self.__config.as_bool(f"{task}.handler"):
                    raise SyntaxError(f"`{task}` element must contain a valid handler")
                if not self.__config.validate_instance(f"{task}.handler"):
                    raise SyntaxError(f"`{task}` element must contain a valid handler in <package>.<class> format")
