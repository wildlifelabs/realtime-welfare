import unittest
from welfareobs.utils.config import Config


class TestConfig(unittest.TestCase):
    def test_get_item_exist(self):
        config: Config = Config("tests/test_config.json")
        self.assertTrue(type(config["pipeline"]) is list)
        self.assertTrue(type(config["camera-1.handler"]) is str)

    def test_get_item_not_exist(self):
        config: Config = Config("tests/test_config.json")

        try:
            tmp = config["nothing"]
            print(tmp) # if this works, something went horribly wrong!
        except KeyError:
            self.assertTrue(True)
            return
        self.assertTrue(False)

    def test_as_list(self):
        config: Config = Config("tests/test_config.json")
        self.assertEqual(len(config.as_list("pipeline")), 7)
        self.assertEqual(len(config.as_list("camera-1.handler")), 1)
        self.assertEqual(len(config.as_list("nothing")), 0)

    def test_as_str(self):
        config: Config = Config("tests/test_config.json")
        self.assertEqual(config.as_string("camera-1.handler"), "welfareobs.handlers.camera.Camera")
        self.assertEqual(config.as_string("camera-1.nothing"), "")
        self.assertEqual(config.as_string("nothing"), "")
        self.assertEqual(config.as_string("nothing.nothing"), "")

    def test_as_int(self):
        config: Config = Config("tests/test_config.json")
        self.assertEqual(config.as_int("camera-1.handler"), 0)
        self.assertEqual(config.as_int("int"), 1)
        self.assertEqual(config.as_int("float"), 0)

    def test_as_float(self):
        config: Config = Config("tests/test_config.json")
        self.assertEqual(config.as_float("camera-1.handler"), 0)
        self.assertEqual(config.as_float("int"), 1.0)
        self.assertEqual(config.as_float("float"), 1.1)

    def test_as_bool(self):
        config: Config = Config("tests/test_config.json")
        self.assertEqual(config.as_bool("camera-1.handler"), True)
        self.assertEqual(config.as_bool("int"), True)
        self.assertEqual(config.as_bool("bool-1"), True)
        self.assertEqual(config.as_bool("bool-2"), False)
        self.assertEqual(config.as_bool("bool-3"), True)
        self.assertEqual(config.as_bool("bool-4"), False)

    def test_validate_instance(self):
        config: Config = Config("tests/test_config.json")
        self.assertEqual(config.validate_instance("camera-1.handler"), True)
        self.assertEqual(config.validate_instance("camera-1.config"), False)
