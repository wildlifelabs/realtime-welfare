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
import argparse



def main():
    parser = argparse.ArgumentParser(description='Process pipeline')
    parser.add_argument('-c', '--config',
                        help='JSON Configuration to use for running the pipeline (filename only)',
                        required=True)
    args = parser.parse_args()                    
    cfg: Config = Config(
	    f"/project/config/{args.config}",
	    "/project/welfareobs/welfareobs"
    )

    runner: Runner = Runner(cfg)
    run_count = None
    if cfg.exists("settings.run-count"):
        run_count=cfg.as_int("settings.run-count")

    run_seconds = None
    if cfg.exists("settings.run-seconds"):
        run_seconds=cfg.as_int("settings.run-seconds")

    performance_csv_filename = None
    if cfg.exists("settings.performance-csv-filename"):
        performance_csv_filename=cfg.as_string("settings.performance-csv-filename")
    
    runner.run(run_count, run_seconds) # 107894 matches the performance-history-size configuration of the evaluation dataset
    runner.performance.save(performance_csv_filename)


if __name__ == "__main__":
    main()

