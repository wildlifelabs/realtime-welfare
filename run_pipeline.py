# -*- coding: utf-8 -*-
"""
Module Name: run_pipeline.py
Description: Run the pipeline

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
from welfareobs.utils.performance_monitor import PerformanceMonitor
from welfareobs.runner import Runner
from welfareobs.utils.config import Config
import matplotlib.pyplot as plt


cfg: Config = Config(
    "/project/config/non-rtsp-example.json",
    "/project/welfareobs/welfareobs"
)

runner: Runner = Runner(cfg)
runner.run(107894) # this matches the performance-history-size configuration
runner.performance.save("/project/performance.csv")