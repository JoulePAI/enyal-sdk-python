"""
Smoke test: verify the SDK is importable from an installed wheel.

This catches packaging bugs (missing __init__.py, wrong package structure,
filename mismatches) that break every user's first import.
"""

import unittest


class TestImports(unittest.TestCase):
    def test_import_package(self):
        """enyal_sdk package imports without error."""
        import enyal_sdk
        self.assertTrue(hasattr(enyal_sdk, "archive"))
        self.assertTrue(hasattr(enyal_sdk, "__version__"))

    def test_import_agent(self):
        """EnyalAgent class is accessible from package root."""
        from enyal_sdk import EnyalAgent
        self.assertTrue(callable(EnyalAgent))

    def test_import_functions(self):
        """Client functions accessible from package root."""
        from enyal_sdk import archive, prove, disclose, send_message
        self.assertTrue(callable(archive))
        self.assertTrue(callable(prove))

    def test_create_agent(self):
        """EnyalAgent can be instantiated with an API key."""
        from enyal_sdk import EnyalAgent
        agent = EnyalAgent(api_key="test-key")
        self.assertEqual(agent.api_key, "test-key")
        self.assertEqual(agent.base_url, "https://api.enyal.ai")

    def test_version(self):
        """__version__ is set and matches pyproject.toml."""
        import enyal_sdk
        self.assertEqual(enyal_sdk.__version__, "2.1.0")


if __name__ == "__main__":
    unittest.main()
