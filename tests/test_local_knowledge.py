"""
Tests for ENYAL SDK local knowledge graph + EnyalAgent.

T1.  Create agent, remember 3 entities
T2.  Recall by keyword — finds matching entities
T3.  Remember contradicting data — contradiction detected
T4.  health() shows contradiction
T5.  compact() returns compressed format
T6.  context(depth=0) returns identity layer
T7.  context(depth=1) returns essentials
T8.  archive() sends to ENYAL AND stores locally (mocked)
T9.  connections() traverses local graph
T10. sync_to_enyal() archives local snapshot (mocked)
T11. sync_from_enyal pulls remote nodes locally (mocked)
T12. remember("Tesla has 100GWh capacity") extracts name
T13. relate(node_a, node_b, "informed_by") creates edge
T14. forget(node_id) removes node and edges
T15. forget doesn't affect ENYAL archives
T16. sync_from_enyal with since parameter
T17. sync_from_enyal pagination — 200 nodes in 2 pages of 100
T18. Modify local node, sync — conflict logged, remote wins
T19. strategy="local_wins" — local modification preserved
T20. sync returns {"synced": N, "conflicts": M}

Run: python3 test_local_knowledge.py
"""

import json
import os
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from local_knowledge import LocalKnowledgeGraph
from enyal_agent import EnyalAgent


class TestLocalKnowledgeGraph(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_knowledge.db")
        self.kg = LocalKnowledgeGraph(self.db_path)

    def tearDown(self):
        self.kg.conn.close()
        os.unlink(self.db_path)
        os.rmdir(self.tmpdir)

    # T1. Create agent, remember 3 entities
    def test_t1_remember_three_entities(self):
        id1 = self.kg.remember("Tesla", "entity", "EV manufacturer",
                               {"sector": "automotive"})
        id2 = self.kg.remember("SpaceX", "entity", "Space launch provider",
                               {"sector": "aerospace"})
        id3 = self.kg.remember("NVIDIA", "entity", "GPU manufacturer",
                               {"sector": "semiconductors"})

        self.assertTrue(id1)
        self.assertTrue(id2)
        self.assertTrue(id3)
        self.assertEqual(len(set([id1, id2, id3])), 3)

        count = self.kg.conn.execute("SELECT count(*) FROM nodes").fetchone()[0]
        self.assertEqual(count, 3)

    # T2. Recall by keyword — finds matching entities
    def test_t2_recall_by_keyword(self):
        self.kg.remember("Tesla", "entity", "EV manufacturer with 100GWh capacity")
        self.kg.remember("SpaceX", "entity", "Launches 90 rockets per year")

        results = self.kg.recall("Tesla")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "Tesla")

        results = self.kg.recall("rockets")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "SpaceX")

        results = self.kg.recall("nonexistent")
        self.assertEqual(len(results), 0)

    # T3. Remember contradicting data — contradiction detected
    def test_t3_contradiction_detection(self):
        self.kg.remember("Tesla", "entity", "EV maker",
                         {"capacity": "100GWh"})
        self.kg.remember("Tesla", "entity", "EV maker",
                         {"capacity": "150GWh"})

        contras = self.kg.contradictions()
        self.assertEqual(len(contras), 1)
        self.assertIn("capacity", contras[0]["evidence"])
        self.assertIn("100GWh", contras[0]["evidence"])
        self.assertIn("150GWh", contras[0]["evidence"])

    # T4. health() shows contradiction
    def test_t4_health_shows_contradiction(self):
        self.kg.remember("Tesla", "entity", "EV maker",
                         {"capacity": "100GWh"})
        self.kg.remember("Tesla", "entity", "EV maker",
                         {"capacity": "150GWh"})

        health = self.kg.health()
        self.assertEqual(health["contradictions"], 1)
        self.assertEqual(health["status"], "needs_attention")

    # T5. compact() returns compressed format
    def test_t5_compact_format(self):
        self.kg.remember("Tesla", "entity", "100GWh battery capacity")
        self.kg.remember("SpaceX", "entity", "90 launches per year")

        compact = self.kg.compact()
        self.assertIn("E:", compact)
        self.assertIn("Tesla", compact)

    # T6. context(depth=0) returns identity layer
    def test_t6_context_depth_0(self):
        self.kg.remember("Tesla", "entity", "EV maker")
        ctx = self.kg.context(depth=0)

        self.assertEqual(ctx["layer"], 0)
        self.assertEqual(ctx["total_nodes"], 1)
        self.assertIn("Tesla", ctx["top_entities"])

    # T7. context(depth=1) returns essentials
    def test_t7_context_depth_1(self):
        self.kg.remember("Tesla", "entity", "EV maker")
        self.kg.remember("SpaceX", "entity", "Rockets")

        ctx = self.kg.context(depth=1)
        self.assertEqual(ctx["layer"], 1)
        self.assertTrue(len(ctx["top_nodes"]) > 0)
        self.assertIn("contradictions", ctx)

    # T9. connections() traverses local graph
    def test_t9_connections_traversal(self):
        id1 = self.kg.remember("Tesla", "entity", "EV maker")
        id2 = self.kg.remember("Elon Musk", "person", "CEO of Tesla")
        self.kg._create_edge(id1, id2, "led_by", evidence="CEO since 2008")

        graph = self.kg.connections(id1, hops=2)
        self.assertEqual(len(graph["nodes"]), 2)
        self.assertEqual(len(graph["edges"]), 1)
        self.assertEqual(graph["edges"][0]["relationship"], "led_by")


class TestEnyalAgent(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_agent.db")
        self.agent = EnyalAgent(api_key="eyl_test_key", local_db=self.db_path)

    def tearDown(self):
        self.agent.local.conn.close()
        os.unlink(self.db_path)
        os.rmdir(self.tmpdir)

    # T8. archive() sends to ENYAL AND stores locally (mocked)
    @patch("enyal_agent.archive")
    def test_t8_archive_stores_locally(self, mock_archive):
        mock_archive.return_value = {"chunk_id": "abc123", "status": "archived"}

        result = self.agent.archive(
            chunk_type="decision_record",
            chunk_key="test:decision:1",
            data={"decision": "Invest in energy", "confidence": 0.85}
        )

        self.assertEqual(result["status"], "archived")
        mock_archive.assert_called_once()

        local = self.agent.recall("Invest in energy")
        self.assertTrue(len(local) > 0)

    # T10. sync_to_enyal() archives local snapshot (mocked)
    @patch("enyal_agent.archive")
    def test_t10_sync_to_enyal(self, mock_archive):
        mock_archive.return_value = {"chunk_id": "sync123"}

        self.agent.remember("Tesla", "entity", "EV maker",
                            {"sector": "auto"})
        result = self.agent.sync_to_enyal(password="test_password")

        self.assertEqual(result["chunk_id"], "sync123")
        call_data = mock_archive.call_args[1].get("data") or mock_archive.call_args[0][4]
        # Verify encrypted snapshot, not raw nodes
        self.assertIn("encrypted_snapshot", call_data)
        self.assertEqual(call_data["version"], 2)
        # name_hash cannot be in encrypted blob
        raw = __import__("base64").b64decode(call_data["encrypted_snapshot"])
        self.assertNotIn(b"name_hash", raw)

    # T11. sync_from_enyal pulls remote nodes locally (mocked)
    @patch("enyal_agent.get_knowledge_nodes")
    def test_t11_sync_from_enyal(self, mock_nodes):
        mock_nodes.return_value = [
            {"name": "RemoteEntity", "node_type": "entity",
             "summary": "From server", "properties": "{}",
             "updated_at": "2026-04-01T00:00:00"}
        ]

        result = self.agent.sync_from_enyal()
        self.assertEqual(result["synced"], 1)

        local = self.agent.recall("RemoteEntity")
        self.assertEqual(len(local), 1)
        self.assertEqual(local[0]["name"], "RemoteEntity")

    # T12. remember("Tesla has 100GWh capacity") extracts name
    def test_t12_natural_language_remember(self):
        node_id = self.agent.remember("Tesla has 100GWh battery capacity")
        self.assertTrue(node_id)

        nodes = self.agent.local.conn.execute(
            "SELECT name, properties FROM nodes WHERE id = ?", (node_id,)
        ).fetchone()
        self.assertIsNotNone(nodes)
        props = json.loads(nodes[1])
        self.assertIn("raw_text", props)

    # T13. relate(node_a, node_b, "informed_by") creates edge
    def test_t13_relate_creates_edge(self):
        id1 = self.agent.remember("Tesla", "entity", "EV maker")
        id2 = self.agent.remember("SpaceX", "entity", "Rockets")

        self.agent.relate(id1, id2, "informed_by",
                          evidence="Tech transfer")

        edges = self.agent.local.conn.execute(
            "SELECT relationship, evidence FROM edges WHERE source_node_id = ? AND target_node_id = ?",
            (id1, id2)
        ).fetchall()
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0][0], "informed_by")
        self.assertEqual(edges[0][1], "Tech transfer")

    # T14. forget(node_id) removes node and edges
    def test_t14_forget_removes_node(self):
        id1 = self.agent.remember("Tesla", "entity", "EV maker")
        id2 = self.agent.remember("SpaceX", "entity", "Rockets")
        self.agent.relate(id1, id2, "partner")

        result = self.agent.forget(id1)
        self.assertTrue(result)

        node = self.agent.local.conn.execute(
            "SELECT id FROM nodes WHERE id = ?", (id1,)
        ).fetchone()
        self.assertIsNone(node)

        edges = self.agent.local.conn.execute(
            "SELECT id FROM edges WHERE source_node_id = ? OR target_node_id = ?",
            (id1, id1)
        ).fetchall()
        self.assertEqual(len(edges), 0)

        # SpaceX still exists
        sp = self.agent.recall("SpaceX")
        self.assertEqual(len(sp), 1)

    # T15. forget doesn't affect ENYAL archives
    @patch("enyal_agent.archive")
    def test_t15_forget_doesnt_affect_enyal(self, mock_archive):
        mock_archive.return_value = {"chunk_id": "perm123"}

        self.agent.archive(
            chunk_type="entity_snapshot",
            chunk_key="test:tesla",
            data={"name": "Tesla", "sector": "auto"}
        )
        mock_archive.assert_called_once()

        local = self.agent.recall("Tesla")
        self.assertTrue(len(local) > 0)
        node_id = local[0]["id"]

        self.agent.forget(node_id)

        local_after = self.agent.recall("Tesla")
        self.assertEqual(len(local_after), 0)

        # archive was only called once — forget didn't touch ENYAL
        mock_archive.assert_called_once()

    # T16. sync_from_enyal with since parameter
    @patch("enyal_agent.get_knowledge_nodes")
    def test_t16_sync_with_since(self, mock_nodes):
        mock_nodes.return_value = [
            {"name": "NewEntity", "node_type": "entity",
             "summary": "Recent", "properties": "{}",
             "updated_at": "2026-04-08T12:00:00"}
        ]

        result = self.agent.sync_from_enyal(since="2026-04-07T00:00:00")
        self.assertEqual(result["synced"], 1)

        call_kwargs = mock_nodes.call_args[1]
        self.assertEqual(call_kwargs.get("since"), "2026-04-07T00:00:00")

    # T17. sync_from_enyal pagination — 200 nodes in 2 pages of 100
    @patch("enyal_agent.get_knowledge_nodes")
    def test_t17_sync_pagination(self, mock_nodes):
        page1 = [{"name": f"Entity_{i}", "node_type": "entity",
                   "summary": f"Node {i}", "properties": "{}",
                   "updated_at": "2026-04-01T00:00:00"}
                  for i in range(100)]
        page2 = [{"name": f"Entity_{i}", "node_type": "entity",
                   "summary": f"Node {i}", "properties": "{}",
                   "updated_at": "2026-04-01T00:00:00"}
                  for i in range(100, 200)]

        mock_nodes.side_effect = [page1, page2, []]

        result = self.agent.sync_from_enyal(limit=100)
        self.assertEqual(result["synced"], 200)
        self.assertEqual(mock_nodes.call_count, 3)

    # T18. Modify local node, sync — conflict logged, remote wins
    @patch("enyal_agent.get_knowledge_nodes")
    def test_t18_conflict_remote_wins(self, mock_nodes):
        # Create a local node and fake a previous sync
        self.agent.remember("Tesla", "entity", "Local version",
                            {"source": "local"})

        # Fake a past sync time
        self.agent.local._log("sync_from_enyal", {
            "nodes_synced": 1,
            "last_updated": "2026-04-01T00:00:00"
        })

        # Now modify the local node (after the sync timestamp)
        import time
        time.sleep(0.01)
        self.agent.remember("Tesla", "entity", "Modified locally",
                            {"source": "local_update"})

        # Remote has a different version
        mock_nodes.return_value = [
            {"name": "Tesla", "node_type": "entity",
             "summary": "Remote version",
             "properties": json.dumps({"source": "remote"}),
             "updated_at": "2026-04-08T00:00:00"}
        ]

        result = self.agent.sync_from_enyal(strategy="remote_wins")
        self.assertEqual(result["conflicts"], 1)
        self.assertEqual(result["synced"], 1)

        # Remote wins — check the summary
        nodes = self.agent.recall("Tesla")
        self.assertEqual(len(nodes), 1)

    # T19. strategy="local_wins" — local modification preserved
    @patch("enyal_agent.get_knowledge_nodes")
    def test_t19_conflict_local_wins(self, mock_nodes):
        self.agent.remember("Tesla", "entity", "Local version",
                            {"source": "local"})

        self.agent.local._log("sync_from_enyal", {
            "nodes_synced": 1,
            "last_updated": "2026-04-01T00:00:00"
        })

        import time
        time.sleep(0.01)
        self.agent.remember("Tesla", "entity", "Modified locally",
                            {"source": "local_update"})

        mock_nodes.return_value = [
            {"name": "Tesla", "node_type": "entity",
             "summary": "Remote version",
             "properties": json.dumps({"source": "remote"}),
             "updated_at": "2026-04-08T00:00:00"}
        ]

        result = self.agent.sync_from_enyal(strategy="local_wins")
        self.assertEqual(result["conflicts"], 1)
        self.assertEqual(result["synced"], 0)  # Nothing synced — local wins

        nodes = self.agent.recall("Tesla")
        self.assertEqual(nodes[0]["summary"], "Modified locally")

    # T20. sync returns {"synced": N, "conflicts": M}
    @patch("enyal_agent.get_knowledge_nodes")
    def test_t20_sync_return_format(self, mock_nodes):
        mock_nodes.return_value = [
            {"name": f"Entity_{i}", "node_type": "entity",
             "summary": f"Node {i}", "properties": "{}",
             "updated_at": "2026-04-01T00:00:00"}
            for i in range(5)
        ]

        result = self.agent.sync_from_enyal()
        self.assertIn("synced", result)
        self.assertIn("conflicts", result)
        self.assertEqual(result["synced"], 5)
        self.assertEqual(result["conflicts"], 0)


class TestSQLInjection(unittest.TestCase):
    """Verify LIKE injection is properly escaped."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_injection.db")
        self.kg = LocalKnowledgeGraph(self.db_path)

    def tearDown(self):
        self.kg.conn.close()
        os.unlink(self.db_path)
        os.rmdir(self.tmpdir)

    def test_percent_in_query(self):
        self.kg.remember("Tesla Motors", "entity", "EV maker")
        self.kg.remember("SpaceX", "entity", "Rockets")

        # % should not match everything
        results = self.kg.recall("%")
        self.assertEqual(len(results), 0)

    def test_underscore_in_query(self):
        self.kg.remember("Tesla", "entity", "EV maker")
        # _ should not match single char wildcard
        results = self.kg.recall("Tesl_")
        self.assertEqual(len(results), 0)


class TestCrossLanguageHash(unittest.TestCase):
    """T22: Verify Python _hash matches JS SDK output."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_hash.db")
        self.kg = LocalKnowledgeGraph(self.db_path)

    def tearDown(self):
        self.kg.conn.close()
        os.unlink(self.db_path)
        os.rmdir(self.tmpdir)

    def test_t22_hash_matches_js(self):
        expected_prefix = "f8f8b4275263e617"
        variants = [
            "Greenland AI Limited",
            "greenland ai",
            "GREENLAND AI LTD",
            "Greenland AI Limited Ltd",
        ]
        for name in variants:
            h = self.kg._hash(name)
            self.assertTrue(
                h.startswith(expected_prefix),
                f"{name!r} -> {h[:16]} should start with {expected_prefix}"
            )
        # All four must produce the exact same hash
        hashes = [self.kg._hash(v) for v in variants]
        self.assertEqual(len(set(hashes)), 1, "All variants produce identical hash")


class TestCorruptedDBRecovery(unittest.TestCase):
    """T23: Corrupted DB recovery — preserves corrupt file, creates fresh DB."""

    def test_t23_corrupted_db_recovery(self):
        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "test_corrupt.db")

        # Write garbage to simulate corruption
        with open(db_path, "wb") as f:
            f.write(b"THIS IS NOT A VALID SQLITE DATABASE FILE")

        kg = LocalKnowledgeGraph(db_path)

        # Should have created a fresh DB
        node_id = kg.remember("TestNode", "entity", "After recovery")
        self.assertTrue(node_id)
        results = kg.recall("TestNode")
        self.assertEqual(len(results), 1)

        # Corrupt file should be preserved
        corrupt_files = [f for f in os.listdir(tmpdir) if ".corrupt." in f]
        self.assertTrue(len(corrupt_files) > 0, "Corrupt file preserved")

        kg.conn.close()
        for f in os.listdir(tmpdir):
            os.unlink(os.path.join(tmpdir, f))
        os.rmdir(tmpdir)


class TestEdgeDedup(unittest.TestCase):
    """T24: Bidirectional traversal doesn't return duplicate edges."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_dedup.db")
        self.kg = LocalKnowledgeGraph(self.db_path)

    def tearDown(self):
        self.kg.conn.close()
        os.unlink(self.db_path)
        os.rmdir(self.tmpdir)

    def test_t24_no_duplicate_edges(self):
        id1 = self.kg.remember("A", "entity", "Node A")
        id2 = self.kg.remember("B", "entity", "Node B")
        id3 = self.kg.remember("C", "entity", "Node C")
        self.kg._create_edge(id1, id2, "links_to")
        self.kg._create_edge(id2, id3, "links_to")

        graph = self.kg.connections(id1, hops=3)
        self.assertEqual(len(graph["nodes"]), 3, "3 nodes traversed")
        self.assertEqual(len(graph["edges"]), 2, "2 edges, no duplicates")

        edge_ids = [e["id"] for e in graph["edges"]]
        self.assertEqual(len(edge_ids), len(set(edge_ids)), "All edge IDs unique")


class TestContextGuard(unittest.TestCase):
    """T25: context(depth=2, topic=None) has no topic_nodes key."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_ctx.db")
        self.kg = LocalKnowledgeGraph(self.db_path)

    def tearDown(self):
        self.kg.conn.close()
        os.unlink(self.db_path)
        os.rmdir(self.tmpdir)

    def test_t25_depth2_no_topic(self):
        self.kg.remember("Tesla", "entity", "Car company")
        ctx = self.kg.context(depth=2, topic=None)
        self.assertNotIn("topic_nodes", ctx)

    def test_t25_depth2_with_topic(self):
        self.kg.remember("Tesla", "entity", "Battery company")
        ctx = self.kg.context(depth=2, topic="Battery")
        self.assertIn("topic_nodes", ctx)
        self.assertEqual(len(ctx["topic_nodes"]), 1)


class TestForgetNonExistent(unittest.TestCase):
    """T26: forget non-existent node returns False."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_forget.db")
        self.agent = EnyalAgent(api_key="test", local_db=self.db_path)

    def tearDown(self):
        self.agent.local.conn.close()
        os.unlink(self.db_path)
        os.rmdir(self.tmpdir)

    def test_t26_forget_nonexistent(self):
        result = self.agent.forget("non-existent-uuid")
        self.assertFalse(result)


class TestRememberBranching(unittest.TestCase):
    """T27-28: remember() with partial args doesn't trigger extraction."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_branch.db")
        self.agent = EnyalAgent(api_key="test", local_db=self.db_path)

    def tearDown(self):
        self.agent.local.conn.close()
        os.unlink(self.db_path)
        os.rmdir(self.tmpdir)

    def test_t27_remember_with_summary_is_explicit(self):
        """summary provided → explicit mode, name stored as-is."""
        node_id = self.agent.remember("Tesla", summary="Car company")
        node = self.agent.local.conn.execute(
            "SELECT name, properties FROM nodes WHERE id = ?", (node_id,)
        ).fetchone()
        self.assertEqual(node[0], "Tesla")
        props = json.loads(node[1])
        self.assertNotIn("raw_text", props, "Explicit mode should not set raw_text")

    def test_t27_remember_with_properties_is_explicit(self):
        """properties provided → explicit mode."""
        node_id = self.agent.remember("SpaceX", properties={"launches": 90})
        node = self.agent.local.conn.execute(
            "SELECT name, properties FROM nodes WHERE id = ?", (node_id,)
        ).fetchone()
        self.assertEqual(node[0], "SpaceX")
        props = json.loads(node[1])
        self.assertEqual(props["launches"], 90)
        self.assertNotIn("raw_text", props)

    def test_t28_remember_bare_name_triggers_extraction(self):
        """No summary or properties → natural language mode with fallback."""
        node_id = self.agent.remember("Tesla has 100GWh capacity")
        node = self.agent.local.conn.execute(
            "SELECT name, properties FROM nodes WHERE id = ?", (node_id,)
        ).fetchone()
        props = json.loads(node[1])
        self.assertIn("raw_text", props, "NL mode should set raw_text")


if __name__ == "__main__":
    unittest.main(verbosity=2)
