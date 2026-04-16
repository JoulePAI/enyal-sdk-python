"""
Smoke test: verify the SDK is importable.

This catches filename mismatches (e.g., enyal-client.py vs enyal_client.py)
that break every user's first import.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestImports(unittest.TestCase):
    def test_import_enyal_client(self):
        """enyal_client module imports without error."""
        import enyal_client
        self.assertTrue(hasattr(enyal_client, "archive"))
        self.assertTrue(hasattr(enyal_client, "_api_call"))

    def test_import_enyal_agent(self):
        """enyal_agent module imports and EnyalAgent class is accessible."""
        from enyal_agent import EnyalAgent
        self.assertTrue(callable(EnyalAgent))

    def test_create_agent(self):
        """EnyalAgent can be instantiated with an API key."""
        from enyal_agent import EnyalAgent
        agent = EnyalAgent(api_key="test-key")
        self.assertEqual(agent.api_key, "test-key")
        self.assertEqual(agent.base_url, "https://api.enyal.ai")


if __name__ == "__main__":
    unittest.main()
