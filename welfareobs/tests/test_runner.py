import unittest

from welfareobs.utils.config import Config
from welfareobs.runner import Runner


class TestRunner(unittest.TestCase):
    def test_invalid_runner(self):
        # TODO: permutations for each syntax error
        config: Config = Config("tests/invalid_runner_config.json")
        result = False
        try:
            runner: Runner = Runner(config)
            print(str(runner))  # should never get to this place
        except SyntaxError:
            result = True
        self.assertTrue(result)

    def test_runner(self):
        config: Config = Config("tests/runner_config.json")
        runner: Runner = Runner(config)
        runner.loop = False  # required to return control to the test harness
        runner.run_once()
        self.assertEqual(runner["task-3"].get_output(), 2)
        self.assertEqual(runner["task-1"].get_output(), 1)
        self.assertEqual(runner["task-2"].get_output(), 1)
        self.assertEqual(runner["task-1"].param, "first config")
        self.assertEqual(runner["task-2"].param, "second config")
        self.assertEqual(runner["task-3"].param, "third config")
        self.assertEqual(runner.get_step(0).jobs[0].name, "task-1" )
        self.assertEqual(runner.get_step(0).jobs[1].name, "task-2" )
        self.assertEqual(runner.get_step(1).jobs[0].name, "task-3" )
        self.assertTrue(runner.get_step(0).jobs[0].has_run)
        self.assertTrue(runner.get_step(0).jobs[1].has_run)
        self.assertTrue(runner.get_step(1).jobs[0].has_run)
        self.assertTrue(runner.get_step(0).jobs[0].has_setup)
        self.assertTrue(runner.get_step(0).jobs[1].has_setup)
        self.assertTrue(runner.get_step(1).jobs[0].has_setup)
        self.assertTrue(runner.get_step(0).jobs[0].has_torndown)
        self.assertTrue(runner.get_step(0).jobs[1].has_torndown)
        self.assertTrue(runner.get_step(1).jobs[0].has_torndown)


