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


class Runner(object):

    def __init__(self, config: Config):
        self.__config: Config = config
        self.__job_map: {str: AbstractHandler} = {}
        self.__pipeline_steps: [PipelineStep] = []
        self.__last_execution_time = 0
        self.__number_of_execution_runs = 0
        self.__overall_execution_time = 0
        self.__validate()  # will raise SyntaxError on invalid config
        self.__parse()
        self.__loop = threading.Event()
        self.__loop.set()
        self.__hnd = None
        self.__has_setup = False
        self.__has_torndown = False

    def __setup(self):
        for job in self.__job_map.values():
            job.setup()
        self.__has_setup = True

    def __teardown(self):
        for job in self.__job_map.values():
            job.teardown()
        self.__has_torndown = True

    def run(self):
        if not self.__has_setup:
            self.__setup()
        if self.__has_torndown:
            raise RuntimeError("Already torn down")
        trigger: bool = True
        while trigger:
            self.__last_execution_time = 0
            for ps in self.__pipeline_steps:
                for job in ps.jobs:
                    src_jobs = [self.__job_map[o] for o in job.required_jobs_for_inputs()]
                    job.set_inputs([o.get_output() for o in src_jobs])
                ps.run()
                self.__last_execution_time += ps.last_execution_time
            self.__number_of_execution_runs += 1
            self.__overall_execution_time += self.__last_execution_time
            trigger = self.__loop.is_set()
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

    def __parse(self):
        #
        # Parse the JSON into a pipeline of jobs
        #
        steps = self.__config.as_list("pipeline")
        for step in steps:
            ps: PipelineStep = PipelineStep()
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
