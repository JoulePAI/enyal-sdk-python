"""
ENYAL Local Knowledge Graph — on-device SQLite knowledge store.

Free, private, instant. No API calls, no network, no cost.
Same extraction logic as the ENYAL server but running locally.

Dependencies: sqlite3 (stdlib), json, hashlib, uuid (all stdlib)
Optional: ollama for LLM-based entity extraction (if installed)
"""

import hashlib
import json
import os
import shutil
import sqlite3
import time
import uuid


class LocalKnowledgeGraph:
    """On-device knowledge graph with SQLite storage."""

    def __init__(self, db_path="~/.enyal/knowledge.db"):
        self.db_path = os.path.expanduser(db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.execute("SELECT 1")
            self._init_tables()
        except sqlite3.DatabaseError:
            corrupt_path = f"{self.db_path}.corrupt.{int(time.time())}"
            shutil.move(self.db_path, corrupt_path)
            self.conn = sqlite3.connect(self.db_path)
            self._init_tables()

    def _init_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                node_type TEXT NOT NULL,
                summary TEXT,
                properties TEXT DEFAULT '{}',
                name_hash TEXT,
                chunk_ids TEXT DEFAULT '[]',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS edges (
                id TEXT PRIMARY KEY,
                source_node_id TEXT NOT NULL,
                target_node_id TEXT NOT NULL,
                relationship TEXT NOT NULL,
                evidence TEXT,
                valid_from TEXT DEFAULT CURRENT_TIMESTAMP,
                valid_to TEXT,
                FOREIGN KEY (source_node_id) REFERENCES nodes(id),
                FOREIGN KEY (target_node_id) REFERENCES nodes(id),
                UNIQUE(source_node_id, target_node_id, relationship)
            );
            CREATE TABLE IF NOT EXISTS log (
                id TEXT PRIMARY KEY,
                action TEXT NOT NULL,
                details TEXT DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_nodes_hash
                ON nodes(name_hash);
            CREATE INDEX IF NOT EXISTS idx_nodes_type
                ON nodes(node_type);
            CREATE INDEX IF NOT EXISTS idx_edges_source
                ON edges(source_node_id);
            CREATE INDEX IF NOT EXISTS idx_edges_target
                ON edges(target_node_id);
        """)

    # === REMEMBER (local storage) ===

    def remember(self, name, node_type="entity",
                 summary=None, properties=None):
        """Store a fact locally. Free, private, instant."""
        name_hash = self._hash(name)
        node_id = str(uuid.uuid4())

        existing = self.conn.execute(
            "SELECT id, properties FROM nodes WHERE name_hash = ?",
            (name_hash,)
        ).fetchone()

        if existing:
            old_props = json.loads(existing[1])
            new_props = properties or {}

            contradictions = self._detect_contradictions(
                name, old_props, new_props
            )

            merged_props = {**old_props, **new_props}
            self.conn.execute(
                "UPDATE nodes SET summary=?, properties=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (summary, json.dumps(merged_props), existing[0])
            )
            node_id = existing[0]

            for c in contradictions:
                self._create_edge(
                    node_id, node_id, "contradicts",
                    evidence=c
                )
        else:
            self.conn.execute(
                "INSERT INTO nodes (id, name, node_type, summary, properties, name_hash) VALUES (?,?,?,?,?,?)",
                (node_id, name, node_type, summary,
                 json.dumps(properties or {}), name_hash)
            )

        self.conn.commit()
        self._log("remember", {"name": name, "type": node_type})
        return node_id

    # === RECALL (local search) ===

    def recall(self, query, limit=10):
        """Search local knowledge. Combines exact LIKE match + TF-IDF semantic search."""
        import math
        query_escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

        # Exact matches
        exact = self.conn.execute(
            "SELECT id, name, node_type, summary, properties FROM nodes "
            "WHERE name LIKE ? ESCAPE '\\' OR summary LIKE ? ESCAPE '\\' "
            "ORDER BY updated_at DESC LIMIT ?",
            (f"%{query_escaped}%", f"%{query_escaped}%", limit)
        ).fetchall()
        exact_results = [{"id": r[0], "name": r[1], "node_type": r[2],
                          "summary": r[3], "properties": json.loads(r[4])} for r in exact]

        # TF-IDF semantic search
        all_rows = self.conn.execute("SELECT id, name, node_type, summary, properties FROM nodes").fetchall()
        if not all_rows:
            return exact_results

        def tokenize(text):
            return [w for w in (text or "").lower().split() if len(w) > 2]

        query_tokens = tokenize(query)
        if not query_tokens:
            return exact_results

        docs = [{"id": r[0], "name": r[1], "node_type": r[2], "summary": r[3],
                 "properties": json.loads(r[4]),
                 "text": f"{r[1]} {r[3] or ''} {r[4] or ''}"} for r in all_rows]
        N = len(docs)

        df = {}
        for d in docs:
            for t in set(tokenize(d["text"])):
                df[t] = df.get(t, 0) + 1

        scored = []
        for d in docs:
            tokens = tokenize(d["text"])
            if not tokens:
                continue
            tf = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            score = sum((tf.get(qt, 0) / len(tokens)) * math.log(N / df.get(qt, 1))
                        for qt in query_tokens if tf.get(qt))
            if score > 0:
                scored.append({**d, "score": score})

        scored.sort(key=lambda x: -x["score"])

        # Merge: exact first, then semantic (dedup)
        seen = {r["id"] for r in exact_results}
        merged = list(exact_results)
        for s in scored:
            if s["id"] not in seen:
                merged.append(s)
                seen.add(s["id"])
        return merged[:limit]

    # === GRAPH OPERATIONS ===

    def connections(self, node_id, hops=2):
        """Traverse local graph N hops from a node."""
        visited = set()
        current = {node_id}
        seen_edge_ids = set()
        all_nodes = []
        all_edges = []

        for hop in range(hops):
            next_level = set()
            for nid in current:
                if nid in visited:
                    continue
                visited.add(nid)

                edges = self.conn.execute(
                    "SELECT id, source_node_id, target_node_id, relationship, evidence "
                    "FROM edges WHERE source_node_id = ? OR target_node_id = ?",
                    (nid, nid)
                ).fetchall()

                for e in edges:
                    if e[0] not in seen_edge_ids:
                        seen_edge_ids.add(e[0])
                        all_edges.append({
                            "id": e[0], "source": e[1],
                            "target": e[2], "relationship": e[3],
                            "evidence": e[4]
                        })
                    other = e[2] if e[1] == nid else e[1]
                    next_level.add(other)

            current = next_level - visited

        for nid in visited:
            node = self.conn.execute(
                "SELECT id, name, node_type, summary FROM nodes WHERE id = ?",
                (nid,)
            ).fetchone()
            if node:
                all_nodes.append({
                    "id": node[0], "name": node[1],
                    "node_type": node[2], "summary": node[3]
                })

        return {"nodes": all_nodes, "edges": all_edges}

    def contradictions(self):
        """List all contradictions in local graph."""
        edges = self.conn.execute(
            "SELECT e.id, e.source_node_id, e.evidence, n.name "
            "FROM edges e JOIN nodes n ON e.source_node_id = n.id "
            "WHERE e.relationship = 'contradicts'"
        ).fetchall()
        return [{"id": e[0], "node_id": e[1],
                 "evidence": e[2], "node_name": e[3]}
                for e in edges]

    def health(self):
        """Knowledge base health check."""
        total_nodes = self.conn.execute("SELECT count(*) FROM nodes").fetchone()[0]
        total_edges = self.conn.execute("SELECT count(*) FROM edges").fetchone()[0]
        contradiction_count = self.conn.execute(
            "SELECT count(*) FROM edges WHERE relationship = 'contradicts'"
        ).fetchone()[0]
        orphans = self.conn.execute(
            "SELECT count(*) FROM nodes n WHERE NOT EXISTS "
            "(SELECT 1 FROM edges e WHERE e.source_node_id = n.id OR e.target_node_id = n.id)"
        ).fetchone()[0]

        status = "healthy"
        if contradiction_count > 5 or orphans > 10:
            status = "unhealthy"
        elif contradiction_count > 0 or orphans > 3:
            status = "needs_attention"

        return {
            "status": status,
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "contradictions": contradiction_count,
            "orphan_nodes": orphans
        }

    def index(self):
        """Grouped overview of entire knowledge base."""
        rows = self.conn.execute(
            "SELECT n.id, n.name, n.node_type, n.summary, COUNT(e.id) as connections "
            "FROM nodes n LEFT JOIN ("
            "SELECT id, source_node_id as nid FROM edges "
            "UNION ALL SELECT id, target_node_id FROM edges"
            ") e ON e.nid = n.id GROUP BY n.id ORDER BY connections DESC"
        ).fetchall()

        grouped = {}
        for r in rows:
            t = r[2]
            if t not in grouped:
                grouped[t] = []
            grouped[t].append({
                "id": r[0], "name": r[1],
                "summary": r[3], "connections": r[4]
            })
        return grouped

    def context(self, depth=1, topic=None):
        """Layered loading for agent context."""
        if depth == 0:
            return {
                "layer": 0,
                "total_nodes": self.conn.execute("SELECT count(*) FROM nodes").fetchone()[0],
                "total_edges": self.conn.execute("SELECT count(*) FROM edges").fetchone()[0],
                "top_entities": [r[0] for r in self.conn.execute(
                    "SELECT name FROM nodes ORDER BY updated_at DESC LIMIT 5"
                ).fetchall()]
            }

        result = self.context(depth=0)
        result["layer"] = depth

        if depth >= 1:
            result["top_nodes"] = self.conn.execute(
                "SELECT n.name, n.node_type, n.summary, COUNT(e.id) "
                "FROM nodes n LEFT JOIN ("
                "SELECT id, source_node_id as nid FROM edges "
                "UNION ALL SELECT id, target_node_id FROM edges"
                ") e ON e.nid = n.id GROUP BY n.id ORDER BY COUNT(e.id) DESC LIMIT 10"
            ).fetchall()
            result["contradictions"] = self.contradictions()

        if depth >= 2 and topic:
            result["topic_nodes"] = self.recall(topic, limit=50)

        if depth >= 3:
            result["all_nodes"] = self.conn.execute(
                "SELECT id, name, node_type, summary FROM nodes"
            ).fetchall()

        return result

    def compact(self, depth=1, topic=None):
        """Compact format for agent context loading."""
        ctx = self.context(depth, topic)
        lines = []

        if "top_nodes" in ctx:
            for n in ctx["top_nodes"]:
                prefix = n[1][0].upper() if n[1] else "?"
                lines.append(f"{prefix}:{n[0]}|{n[2] or ''}|{n[3]}c")

        for c in ctx.get("contradictions", []):
            lines.append(f"!{c['node_name']}:{c['evidence']}")

        return "\n".join(lines)

    # === INTERNAL ===

    def _hash(self, name):
        normalised = name.lower().strip()
        for suffix in [' ltd', ' inc', ' gmbh', ' limited', ' corp']:
            if normalised.endswith(suffix):
                normalised = normalised[:-len(suffix)].strip()
        return hashlib.sha256(normalised.encode()).hexdigest()

    def _detect_contradictions(self, name, old_props, new_props):
        contradictions = []
        for key in set(old_props.keys()) & set(new_props.keys()):
            if old_props[key] != new_props[key]:
                contradictions.append(
                    f"{key}: was {old_props[key]}, now {new_props[key]}"
                )
        return contradictions

    def _create_edge(self, source, target, relationship, evidence=None):
        try:
            self.conn.execute(
                "INSERT OR IGNORE INTO edges (id, source_node_id, target_node_id, relationship, evidence) VALUES (?,?,?,?,?)",
                (str(uuid.uuid4()), source, target, relationship, evidence)
            )
            self.conn.commit()
        except sqlite3.IntegrityError:
            pass  # Duplicate edge, expected
        except Exception as e:
            self._log("edge_error", {"error": str(e)})

    def _log(self, action, details):
        self.conn.execute(
            "INSERT INTO log (id, action, details) VALUES (?,?,?)",
            (str(uuid.uuid4()), action, json.dumps(details))
        )
        self.conn.commit()
