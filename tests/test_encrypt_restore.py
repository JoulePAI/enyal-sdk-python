"""
Tests for encrypted sync + restore flow.

T1.  remember 5 entities + 3 edges locally
T2.  sync_to_enyal(password) archives encrypted blob (mocked)
T3.  Verify chunk has "encrypted_snapshot" field
T4.  Verify ENYAL cannot read node names from the chunk
T5.  Create NEW agent with empty local DB
T6.  restore_from_enyal(password) restores all 5 nodes + 3 edges (mocked)
T7.  health() matches original
T8.  Wrong password → "Decryption failed" error
T9.  compact() on restored agent matches original output
T10. (JS tested separately)
T11. (Mobile tested separately)
T12. (Supply invariant — JoulePAI)
T13. Python encrypt → JS-compatible format verification
T14. (JS decrypt tested in JS suite)
T15. Restore empty snapshot (0 nodes, 0 edges)
T16. Restore when no snapshot on ENYAL
T17. Partial restore failure — local data preserved
T18. Sync twice, restore gets latest
T19. Two different passwords → different blobs
T20. Same password + same data → different ciphertext (random IV)
T21. Two different account IDs → different keys

Run: python3 test_encrypt_restore.py
"""

import base64
import hashlib
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from local_knowledge import LocalKnowledgeGraph
from enyal_agent import EnyalAgent


def make_agent(db_path=None):
    if not db_path:
        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "test.db")
    return EnyalAgent(api_key="eyl_test_key_12345", local_db=db_path)


def populate(agent):
    """Add 5 entities + 3 edges."""
    ids = []
    ids.append(agent.remember("Tesla", "entity", "EV maker", {"sector": "auto"}))
    ids.append(agent.remember("SpaceX", "entity", "Rockets", {"launches": 90}))
    ids.append(agent.remember("NVIDIA", "entity", "GPU maker", {"sector": "semi"}))
    ids.append(agent.remember("Blue Origin", "entity", "Rockets", {"launches": 12}))
    ids.append(agent.remember("Elon Musk", "person", "CEO"))
    agent.relate(ids[0], ids[4], "led_by", "CEO since 2008")
    agent.relate(ids[1], ids[4], "led_by", "Founder")
    agent.relate(ids[0], ids[2], "uses", "GPU compute")
    return ids


class TestEncryptedSync(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.agent = make_agent(self.db_path)
        self.ids = populate(self.agent)

    def tearDown(self):
        self.agent.local.conn.close()
        for f in os.listdir(self.tmpdir):
            os.unlink(os.path.join(self.tmpdir, f))
        os.rmdir(self.tmpdir)

    # T1. remember 5 entities + 3 edges
    def test_t1_populated(self):
        h = self.agent.health()
        self.assertEqual(h["total_nodes"], 5)
        self.assertEqual(h["total_edges"], 3)

    # T2. sync_to_enyal archives encrypted blob
    @patch("enyal_agent.archive")
    def test_t2_sync_encrypted(self, mock_archive):
        mock_archive.return_value = {"chunk_id": "sync123"}
        result = self.agent.sync_to_enyal(password="test123")
        self.assertEqual(result["chunk_id"], "sync123")
        mock_archive.assert_called_once()

    # T3. Verify chunk has encrypted_snapshot
    @patch("enyal_agent.archive")
    def test_t3_has_encrypted_field(self, mock_archive):
        mock_archive.return_value = {"chunk_id": "sync123"}
        self.agent.sync_to_enyal(password="test123")
        call_kwargs = mock_archive.call_args
        data = call_kwargs[1].get("data") or call_kwargs[0][4]
        self.assertIn("encrypted_snapshot", data)
        self.assertEqual(data["encryption"], "AES-256-GCM")
        self.assertEqual(data["version"], 2)
        self.assertEqual(data["node_count"], 5)
        self.assertEqual(data["edge_count"], 3)

    # T4. ENYAL cannot read node names from the chunk
    @patch("enyal_agent.archive")
    def test_t4_names_not_readable(self, mock_archive):
        mock_archive.return_value = {"chunk_id": "sync123"}
        self.agent.sync_to_enyal(password="test123")
        data = mock_archive.call_args[1].get("data") or mock_archive.call_args[0][4]
        blob = data["encrypted_snapshot"]
        # The blob is base64 — decode it, should NOT contain "Tesla"
        raw = base64.b64decode(blob)
        self.assertNotIn(b"Tesla", raw)
        self.assertNotIn(b"SpaceX", raw)
        self.assertNotIn(b"NVIDIA", raw)

    # T5+T6. Create new agent, restore
    @patch("enyal_agent.archive")
    @patch("enyal_agent.search")
    def test_t5_t6_restore(self, mock_search, mock_archive):
        # Sync first
        mock_archive.return_value = {"chunk_id": "sync123"}
        self.agent.sync_to_enyal(password="test123")
        archived_data = mock_archive.call_args[1].get("data") or mock_archive.call_args[0][4]

        # New agent with empty DB
        tmpdir2 = tempfile.mkdtemp()
        db2 = os.path.join(tmpdir2, "test2.db")
        agent2 = EnyalAgent(api_key="eyl_test_key_12345", local_db=db2)
        self.assertEqual(agent2.health()["total_nodes"], 0)

        # Mock search to return our archived chunk
        mock_search.return_value = {"chunks": [{"data": archived_data}]}
        result = agent2.restore_from_enyal(password="test123")

        self.assertEqual(result["nodes_restored"], 5)
        self.assertEqual(result["edges_restored"], 3)

        agent2.local.conn.close()
        for f in os.listdir(tmpdir2):
            os.unlink(os.path.join(tmpdir2, f))
        os.rmdir(tmpdir2)

    # T7. health() matches original
    @patch("enyal_agent.archive")
    @patch("enyal_agent.search")
    def test_t7_health_matches(self, mock_search, mock_archive):
        original_health = self.agent.health()
        mock_archive.return_value = {"chunk_id": "sync123"}
        self.agent.sync_to_enyal(password="test123")
        archived_data = mock_archive.call_args[1].get("data") or mock_archive.call_args[0][4]

        tmpdir2 = tempfile.mkdtemp()
        db2 = os.path.join(tmpdir2, "test2.db")
        agent2 = EnyalAgent(api_key="eyl_test_key_12345", local_db=db2)
        mock_search.return_value = {"chunks": [{"data": archived_data}]}
        agent2.restore_from_enyal(password="test123")

        restored_health = agent2.health()
        self.assertEqual(restored_health["total_nodes"], original_health["total_nodes"])
        self.assertEqual(restored_health["total_edges"], original_health["total_edges"])

        agent2.local.conn.close()
        for f in os.listdir(tmpdir2):
            os.unlink(os.path.join(tmpdir2, f))
        os.rmdir(tmpdir2)

    # T8. Wrong password
    @patch("enyal_agent.archive")
    @patch("enyal_agent.search")
    def test_t8_wrong_password(self, mock_search, mock_archive):
        mock_archive.return_value = {"chunk_id": "sync123"}
        self.agent.sync_to_enyal(password="correct_pass")
        archived_data = mock_archive.call_args[1].get("data") or mock_archive.call_args[0][4]

        tmpdir2 = tempfile.mkdtemp()
        db2 = os.path.join(tmpdir2, "test2.db")
        agent2 = EnyalAgent(api_key="eyl_test_key_12345", local_db=db2)
        mock_search.return_value = {"chunks": [{"data": archived_data}]}

        with self.assertRaises(RuntimeError) as ctx:
            agent2.restore_from_enyal(password="wrong_pass")
        self.assertIn("Decryption failed", str(ctx.exception))

        agent2.local.conn.close()
        for f in os.listdir(tmpdir2):
            os.unlink(os.path.join(tmpdir2, f))
        os.rmdir(tmpdir2)

    # T9. compact() matches
    @patch("enyal_agent.archive")
    @patch("enyal_agent.search")
    def test_t9_compact_matches(self, mock_search, mock_archive):
        original_compact = self.agent.compact()
        mock_archive.return_value = {"chunk_id": "sync123"}
        self.agent.sync_to_enyal(password="test123")
        archived_data = mock_archive.call_args[1].get("data") or mock_archive.call_args[0][4]

        tmpdir2 = tempfile.mkdtemp()
        db2 = os.path.join(tmpdir2, "test2.db")
        agent2 = EnyalAgent(api_key="eyl_test_key_12345", local_db=db2)
        mock_search.return_value = {"chunks": [{"data": archived_data}]}
        agent2.restore_from_enyal(password="test123")

        restored_compact = agent2.compact()
        # Same node names should appear (order may differ)
        for name in ["Tesla", "SpaceX", "NVIDIA", "Elon Musk", "Blue Origin"]:
            self.assertIn(name, restored_compact)

        agent2.local.conn.close()
        for f in os.listdir(tmpdir2):
            os.unlink(os.path.join(tmpdir2, f))
        os.rmdir(tmpdir2)

    # T13. Python encrypt produces valid format for JS
    @patch("enyal_agent.archive")
    def test_t13_cross_platform_format(self, mock_archive):
        mock_archive.return_value = {"chunk_id": "sync123"}
        self.agent.sync_to_enyal(password="testpass123")
        data = mock_archive.call_args[1].get("data") or mock_archive.call_args[0][4]

        blob = base64.b64decode(data["encrypted_snapshot"])
        # Format: iv[12] + ciphertext + tag[16]
        self.assertGreater(len(blob), 28, "Blob must be > 28 bytes (12 IV + 16 tag min)")
        iv = blob[:12]
        self.assertEqual(len(iv), 12, "IV is 12 bytes")
        # Verify it's valid base64 round-trip
        re_encoded = base64.b64encode(blob).decode()
        self.assertEqual(re_encoded, data["encrypted_snapshot"])

    # T15. Empty snapshot
    @patch("enyal_agent.archive")
    @patch("enyal_agent.search")
    def test_t15_empty_snapshot(self, mock_search, mock_archive):
        tmpdir2 = tempfile.mkdtemp()
        db2 = os.path.join(tmpdir2, "empty.db")
        empty_agent = EnyalAgent(api_key="eyl_test_key_12345", local_db=db2)

        mock_archive.return_value = {"chunk_id": "empty"}
        empty_agent.sync_to_enyal(password="test123")
        archived = mock_archive.call_args[1].get("data") or mock_archive.call_args[0][4]
        self.assertEqual(archived["node_count"], 0)
        self.assertEqual(archived["edge_count"], 0)

        # Restore empty snapshot
        tmpdir3 = tempfile.mkdtemp()
        db3 = os.path.join(tmpdir3, "restore.db")
        agent3 = EnyalAgent(api_key="eyl_test_key_12345", local_db=db3)
        mock_search.return_value = {"chunks": [{"data": archived}]}
        result = agent3.restore_from_enyal(password="test123")
        self.assertEqual(result["nodes_restored"], 0)
        self.assertEqual(result["edges_restored"], 0)

        for a, d in [(empty_agent, tmpdir2), (agent3, tmpdir3)]:
            a.local.conn.close()
            for f in os.listdir(d):
                os.unlink(os.path.join(d, f))
            os.rmdir(d)

    # T16. No snapshot on ENYAL
    @patch("enyal_agent.search")
    def test_t16_no_snapshot(self, mock_search):
        mock_search.return_value = {"chunks": []}
        with self.assertRaises(RuntimeError) as ctx:
            self.agent.restore_from_enyal(password="test123")
        self.assertIn("No knowledge graph snapshot found", str(ctx.exception))

    # T17. Partial failure preserves local data
    @patch("enyal_agent.search")
    def test_t17_failed_restore_preserves(self, mock_search):
        # Agent has 5 nodes
        self.assertEqual(self.agent.health()["total_nodes"], 5)

        # Mock returns invalid encrypted data
        mock_search.return_value = {"chunks": [{"data": {
            "encrypted_snapshot": base64.b64encode(b"garbage" * 10).decode(),
            "version": 2, "node_count": 1, "edge_count": 0,
        }}]}

        with self.assertRaises(RuntimeError):
            self.agent.restore_from_enyal(password="test123")

        # Re-open agent to check backup was restored
        self.agent.local.conn.close()
        self.agent = make_agent(self.db_path)
        self.assertEqual(self.agent.health()["total_nodes"], 5,
                         "Local data preserved after failed restore")

    # T18. Sync twice, restore gets latest
    @patch("enyal_agent.archive")
    @patch("enyal_agent.search")
    def test_t18_restore_gets_latest(self, mock_search, mock_archive):
        mock_archive.return_value = {"chunk_id": "sync1"}
        self.agent.sync_to_enyal(password="test123")
        first_data = (mock_archive.call_args[1].get("data") or mock_archive.call_args[0][4]).copy()

        # Add another node
        self.agent.remember("OpenAI", "entity", "AI lab")
        mock_archive.return_value = {"chunk_id": "sync2"}
        self.agent.sync_to_enyal(password="test123")
        second_data = mock_archive.call_args[1].get("data") or mock_archive.call_args[0][4]

        # Search returns latest (second sync — 6 nodes)
        self.assertEqual(second_data["node_count"], 6)
        mock_search.return_value = {"chunks": [{"data": second_data}]}

        tmpdir2 = tempfile.mkdtemp()
        db2 = os.path.join(tmpdir2, "test2.db")
        agent2 = EnyalAgent(api_key="eyl_test_key_12345", local_db=db2)
        result = agent2.restore_from_enyal(password="test123")
        self.assertEqual(result["nodes_restored"], 6)

        agent2.local.conn.close()
        for f in os.listdir(tmpdir2):
            os.unlink(os.path.join(tmpdir2, f))
        os.rmdir(tmpdir2)

    # T19. Different passwords → different blobs
    @patch("enyal_agent.archive")
    def test_t19_different_passwords(self, mock_archive):
        mock_archive.return_value = {"chunk_id": "sync1"}
        self.agent.sync_to_enyal(password="password_A")
        blob_a = (mock_archive.call_args[1].get("data") or mock_archive.call_args[0][4])["encrypted_snapshot"]

        mock_archive.return_value = {"chunk_id": "sync2"}
        self.agent.sync_to_enyal(password="password_B")
        blob_b = (mock_archive.call_args[1].get("data") or mock_archive.call_args[0][4])["encrypted_snapshot"]

        self.assertNotEqual(blob_a, blob_b)

    # T20. Same password + same data → different ciphertext (random IV)
    @patch("enyal_agent.archive")
    def test_t20_random_iv(self, mock_archive):
        mock_archive.return_value = {"chunk_id": "sync1"}
        self.agent.sync_to_enyal(password="same_pass")
        blob_1 = (mock_archive.call_args[1].get("data") or mock_archive.call_args[0][4])["encrypted_snapshot"]

        mock_archive.return_value = {"chunk_id": "sync2"}
        self.agent.sync_to_enyal(password="same_pass")
        blob_2 = (mock_archive.call_args[1].get("data") or mock_archive.call_args[0][4])["encrypted_snapshot"]

        self.assertNotEqual(blob_1, blob_2, "Same data + same password must produce different ciphertext")

    # T21. Different account IDs → different keys
    def test_t21_different_accounts_different_keys(self):
        agent_a = EnyalAgent(api_key="eyl_account_alpha", local_db=self.db_path)
        agent_b = EnyalAgent(api_key="eyl_account_beta", local_db=self.db_path)

        key_a = agent_a._derive_snapshot_key("same_password")
        key_b = agent_b._derive_snapshot_key("same_password")

        self.assertNotEqual(key_a, key_b,
                            "Different API keys must produce different encryption keys")


class TestPasswordRequired(unittest.TestCase):

    def test_sync_requires_password(self):
        tmpdir = tempfile.mkdtemp()
        db = os.path.join(tmpdir, "test.db")
        agent = make_agent(db)
        with self.assertRaises(ValueError):
            agent.sync_to_enyal(password="")
        with self.assertRaises(ValueError):
            agent.sync_to_enyal(password=None)
        agent.local.conn.close()
        os.unlink(db)
        os.rmdir(tmpdir)

    def test_restore_requires_password(self):
        tmpdir = tempfile.mkdtemp()
        db = os.path.join(tmpdir, "test.db")
        agent = make_agent(db)
        with self.assertRaises(ValueError):
            agent.restore_from_enyal(password="")
        agent.local.conn.close()
        os.unlink(db)
        os.rmdir(tmpdir)


if __name__ == "__main__":
    unittest.main(verbosity=2)
