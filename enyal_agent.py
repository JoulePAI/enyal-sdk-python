"""
EnyalAgent — local brain + permanent proof in one interface.

    agent = EnyalAgent(api_key="eyl_xxx")

    # Local (free, private, instant)
    agent.remember("SpaceX has 90 launches per year")
    results = agent.recall("SpaceX capacity")

    # Permanent proof (costs joules)
    agent.archive(chunk_type="decision_record",
                  data={"decision": "Invest in space"})

    # Context for AI
    context = agent.compact()
    # Paste into any LLM system prompt

Dependencies: sqlite3 (stdlib), json, hashlib (all stdlib)
Optional: httpx or urllib for Ollama LLM extraction
"""

import base64
import datetime
import hashlib
import json
import os
import shutil
import sys
import time

# Allow importing from same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from local_knowledge import LocalKnowledgeGraph

# Import the existing ENYAL client functions
from importlib.util import spec_from_file_location, module_from_spec
_client_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "enyal_client.py")
_spec = spec_from_file_location("enyal_client", _client_path)
_client = module_from_spec(_spec)
_spec.loader.exec_module(_client)

archive = _client.archive
search = _client.search
prove = _client.prove
disclose = _client.disclose
send_message = _client.send_message
get_inbox = _client.get_inbox
get_thread = _client.get_thread
mark_read = _client.mark_read
get_knowledge_nodes = _client.get_knowledge_nodes
get_knowledge_node = _client.get_knowledge_node
get_knowledge_connections = _client.get_knowledge_connections
get_contradictions = _client.get_contradictions
get_knowledge_stats = _client.get_knowledge_stats
get_knowledge_index = _client.get_knowledge_index
get_knowledge_health = _client.get_knowledge_health
synthesise_knowledge = _client.synthesise_knowledge

MAX_SYNC_PAGES = 50


class EnyalAgent:
    def __init__(self, api_key, local_db=None, base_url=None):
        self.api_key = api_key
        self.base_url = base_url or "https://api.enyal.ai"
        db_path = local_db or os.path.expanduser("~/.enyal/knowledge.db")
        self.local = LocalKnowledgeGraph(db_path)
        self._validated = False

    # === LOCAL MEMORY (free, private) ===

    def remember(self, text, node_type="entity",
                 summary=None, properties=None):
        """Store locally. Free. Private. Instant.

        Two modes:
          - Natural language: agent.remember("Tesla has 100GWh capacity")
            Extracts entity name via Ollama (if available) or uses full text.
          - Explicit: agent.remember("Tesla", summary="...", properties={...})
        """
        if properties is None and summary is None:
            name, props = self._extract_from_text(text)
            return self.local.remember(name, node_type, text, props)
        else:
            return self.local.remember(text, node_type, summary, properties)

    def recall(self, query, limit=10):
        """Search local knowledge."""
        return self.local.recall(query, limit)

    def connections(self, node_id, hops=2):
        """Traverse local graph."""
        return self.local.connections(node_id, hops)

    def contradictions(self):
        """List local contradictions."""
        return self.local.contradictions()

    def health(self):
        """Local knowledge health check."""
        return self.local.health()

    def index(self):
        """Local knowledge index."""
        return self.local.index()

    def context(self, depth=1, topic=None):
        """Layered context from local graph."""
        return self.local.context(depth, topic)

    def compact(self, depth=1, topic=None):
        """Compact format for agent prompts."""
        return self.local.compact(depth, topic)

    def relate(self, source_id, target_id, relationship,
               evidence=None):
        """Create a relationship between two nodes."""
        self.local._create_edge(
            source_id, target_id, relationship, evidence
        )

    def forget(self, node_id):
        """Remove from local graph. Does NOT affect ENYAL archives."""
        self.local.conn.execute(
            "DELETE FROM edges WHERE source_node_id = ? OR target_node_id = ?",
            (node_id, node_id)
        )
        cursor = self.local.conn.execute(
            "DELETE FROM nodes WHERE id = ?", (node_id,)
        )
        self.local.conn.commit()
        self.local._log("forget", {"node_id": node_id})
        return cursor.rowcount > 0

    # === PERMANENT PROOF (costs joules) ===

    def archive(self, chunk_type, chunk_key, data,
                agent_id=None, idempotency_key=None, retry=True):
        """Archive to ENYAL. Permanent. Encrypted. Provable.

        Args:
            idempotency_key: Key for safe retries. Auto-generated if omitted.
            retry: Set False to disable automatic retries.
        """
        result = archive(
            self.api_key,
            agent_id=agent_id or "sdk-agent",
            chunk_type=chunk_type,
            chunk_key=chunk_key,
            data=data,
            base_url=self.base_url,
            idempotency_key=idempotency_key,
            retry=retry,
        )

        name = data.get("name") or data.get("decision") or chunk_key
        self.local.remember(
            name,
            node_type=self._type_from_chunk(chunk_type),
            summary=str(data)[:200],
            properties=data
        )

        return result

    def prove(self, resource_type, idempotency_key=None, retry=True, **kwargs):
        """Generate ZK proof.

        Args:
            idempotency_key: Key for safe retries. Auto-generated if omitted.
            retry: Set False to disable automatic retries.
        """
        return prove(self.api_key, resource_type, base_url=self.base_url,
                     idempotency_key=idempotency_key, retry=retry, **kwargs)

    def disclose(self, chunk_ids, recipient_pubkey, purpose,
                 idempotency_key=None, retry=True):
        """Selective disclosure.

        Args:
            idempotency_key: Key for safe retries. Auto-generated if omitted.
            retry: Set False to disable automatic retries.
        """
        return disclose(
            self.api_key, chunk_ids,
            recipient_pubkey, purpose,
            base_url=self.base_url,
            idempotency_key=idempotency_key,
            retry=retry,
        )

    # === MESSAGING ===

    def send(self, sender_id, thread_id, recipient_id,
             message_type, payload, idempotency_key=None, retry=True):
        """Send agent message via ENYAL.

        Args:
            idempotency_key: Key for safe retries. Auto-generated if omitted.
            retry: Set False to disable automatic retries.
        """
        return send_message(
            self.api_key, sender_id, thread_id,
            recipient_id, message_type, payload,
            base_url=self.base_url,
            idempotency_key=idempotency_key,
            retry=retry,
        )

    def inbox(self, agent_id, **kwargs):
        """Get messages."""
        return get_inbox(self.api_key, agent_id, base_url=self.base_url, **kwargs)

    # === SYNC ===

    def sync_to_enyal(self, password):
        """Encrypt local graph client-side and archive to ENYAL.

        The snapshot is encrypted with a key derived from your password.
        ENYAL stores the encrypted blob — it cannot read it.
        Password is mandatory. No server key path.

        WARNING: Snapshots are encrypted with your current password.
        If you change your password, old snapshots become undecryptable.
        Re-sync immediately after any password change.

        Args:
            password: ENYAL account password (required).
        """
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        if not password or not password.strip():
            raise ValueError("Password cannot be empty")

        nodes = self.local.conn.execute(
            "SELECT id, name, node_type, summary, properties, chunk_ids, created_at, updated_at FROM nodes"
        ).fetchall()
        edges = self.local.conn.execute(
            "SELECT id, source_node_id, target_node_id, relationship, evidence, valid_from, valid_to FROM edges"
        ).fetchall()

        # name_hash deliberately excluded — prevents rainbow table attack
        snapshot = {
            "nodes": [
                {"id": n[0], "name": n[1], "node_type": n[2],
                 "summary": n[3], "properties": n[4],
                 "chunk_ids": n[5], "created_at": n[6], "updated_at": n[7]}
                for n in nodes
            ],
            "edges": [
                {"id": e[0], "source": e[1], "target": e[2],
                 "relationship": e[3], "evidence": e[4],
                 "valid_from": e[5], "valid_to": e[6]}
                for e in edges
            ],
            "node_count": len(nodes),
            "edge_count": len(edges),
            "exported_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

        plaintext = json.dumps(snapshot).encode()
        plaintext_hash = hashlib.sha256(plaintext).hexdigest()

        # Derive key: HKDF-SHA256, salt includes account identity
        key = bytearray(self._derive_snapshot_key(password))

        # AES-256-GCM encrypt
        aesgcm = AESGCM(bytes(key))

        # Verify key derivation before encrypting the real data
        test_pt = b"enyal-key-verification"
        test_iv = os.urandom(12)
        test_ct = aesgcm.encrypt(test_iv, test_pt, None)
        try:
            assert aesgcm.decrypt(test_iv, test_ct, None) == test_pt
        except Exception:
            for i in range(len(key)):
                key[i] = 0
            raise RuntimeError(
                "Key derivation verification failed. Please report this bug."
            )

        iv = os.urandom(12)
        ciphertext = aesgcm.encrypt(iv, plaintext, None)
        encrypted_blob = base64.b64encode(iv + ciphertext).decode()

        result = archive(
            self.api_key,
            agent_id="sdk-sync",
            chunk_type="knowledge_graph_snapshot",
            chunk_key=f"kg-snapshot:{datetime.datetime.now(datetime.timezone.utc).isoformat()}",
            data={
                "encrypted_snapshot": encrypted_blob,
                "plaintext_hash": plaintext_hash,
                "node_count": len(nodes),
                "edge_count": len(edges),
                "encryption": "AES-256-GCM",
                "key_derivation": "HKDF-SHA256",
                "version": 2,
            },
            base_url=self.base_url,
        )

        self.local._log("sync_to_enyal", {
            "nodes": len(nodes), "edges": len(edges), "encrypted": True,
        })

        # Best-effort memory clearing
        for i in range(len(key)):
            key[i] = 0
        del plaintext

        return result

    def restore_from_enyal(self, password):
        """Download encrypted snapshot from ENYAL, decrypt locally,
        restore local knowledge graph.

        Use this on a new device or after data loss.
        Requires your ENYAL password — the server cannot decrypt it.

        Args:
            password: ENYAL account password (required).
        """
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        if not password:
            raise ValueError("Password required for restore")

        # Find the most recent snapshot
        results = search(
            self.api_key,
            chunk_type="knowledge_graph_snapshot",
            limit=1,
            base_url=self.base_url,
        )
        chunks = results.get("chunks") or results.get("results") or []
        if not chunks:
            raise RuntimeError("No knowledge graph snapshot found on ENYAL")

        chunk = chunks[0]
        data = chunk.get("data", {})
        if isinstance(data, str):
            data = json.loads(data)

        if "encrypted_snapshot" not in data:
            raise RuntimeError("Snapshot is not encrypted — legacy format")
        version = data.get("version", 1)
        if version < 2:
            raise RuntimeError(
                f"Snapshot version {version} not supported. Re-sync with latest SDK."
            )

        # Backup current DB before clearing
        backup_path = f"{self.local.db_path}.pre-restore.{int(time.time())}"
        if os.path.exists(self.local.db_path):
            shutil.copy2(self.local.db_path, backup_path)

        key = bytearray(self._derive_snapshot_key(password))

        try:
            # Decrypt
            encrypted_blob = base64.b64decode(data["encrypted_snapshot"])
            iv = encrypted_blob[:12]
            ciphertext = encrypted_blob[12:]

            aesgcm = AESGCM(bytes(key))
            try:
                plaintext = aesgcm.decrypt(iv, ciphertext, None)
            except Exception:
                for i in range(len(key)):
                    key[i] = 0
                raise RuntimeError(
                    "Decryption failed. Wrong password or corrupted snapshot."
                )

            # Verify integrity hash
            stored_hash = data.get("plaintext_hash")
            if stored_hash:
                restored_hash = hashlib.sha256(plaintext).hexdigest()
                if restored_hash != stored_hash:
                    raise RuntimeError(
                        "Snapshot integrity check failed — data may have been tampered with"
                    )

            snapshot = json.loads(plaintext.decode())

            # Verify counts
            expected_nodes = data.get("node_count", 0)
            expected_edges = data.get("edge_count", 0)
            actual_nodes = len(snapshot.get("nodes", []))
            actual_edges = len(snapshot.get("edges", []))
            if actual_nodes != expected_nodes or actual_edges != expected_edges:
                raise RuntimeError(
                    f"Snapshot count mismatch. Expected {expected_nodes} nodes/"
                    f"{expected_edges} edges, got {actual_nodes}/{actual_edges}"
                )

            # Clear local graph
            self.local.conn.execute("DELETE FROM edges")
            self.local.conn.execute("DELETE FROM nodes")
            self.local.conn.execute("DELETE FROM log")
            self.local.conn.commit()

            # Restore nodes (count actual inserts)
            nodes_restored = 0
            for node in snapshot.get("nodes", []):
                try:
                    self.local.conn.execute(
                        "INSERT OR REPLACE INTO nodes "
                        "(id, name, node_type, summary, properties, name_hash, chunk_ids, created_at, updated_at) "
                        "VALUES (?,?,?,?,?,?,?,?,?)",
                        (
                            node.get("id", ""), node.get("name", ""),
                            node.get("node_type", "entity"),
                            node.get("summary", ""),
                            node.get("properties", "{}"),
                            self.local._hash(node.get("name", "")),
                            node.get("chunk_ids", "[]"),
                            node.get("created_at", ""),
                            node.get("updated_at", ""),
                        )
                    )
                    nodes_restored += 1
                except Exception:
                    pass

            # Restore edges
            edges_restored = 0
            for edge in snapshot.get("edges", []):
                try:
                    self.local.conn.execute(
                        "INSERT OR REPLACE INTO edges "
                        "(id, source_node_id, target_node_id, relationship, evidence, valid_from, valid_to) "
                        "VALUES (?,?,?,?,?,?,?)",
                        (
                            edge.get("id", ""), edge.get("source", ""),
                            edge.get("target", ""),
                            edge.get("relationship", ""),
                            edge.get("evidence"),
                            edge.get("valid_from"),
                            edge.get("valid_to"),
                        )
                    )
                    edges_restored += 1
                except Exception:
                    pass

            self.local.conn.commit()

            self.local._log("restore_from_enyal", {
                "nodes_restored": nodes_restored,
                "edges_restored": edges_restored,
                "snapshot_date": snapshot.get("exported_at"),
            })

        except Exception:
            # Restore failed — put backup back
            if os.path.exists(backup_path):
                self.local.conn.close()
                shutil.move(backup_path, self.local.db_path)
                self.local = LocalKnowledgeGraph(self.local.db_path)
            raise
        finally:
            for i in range(len(key)):
                key[i] = 0

        return {
            "nodes_restored": nodes_restored,
            "edges_restored": edges_restored,
            "nodes_expected": actual_nodes,
            "edges_expected": actual_edges,
            "snapshot_date": snapshot.get("exported_at"),
        }

    def sync_from_enyal(self, since=None, limit=100,
                        strategy="remote_wins"):
        """Pull remote knowledge into local graph.

        Args:
            since: ISO timestamp. Only pull nodes updated after this.
                   If None, uses last sync timestamp from local DB.
            limit: Max nodes per page. Paginates automatically.
            strategy: Conflict resolution.
                "remote_wins" — remote overwrites local on conflict (default)
                "local_wins" — skip remote updates if local was modified
        """
        # Get last sync time once (not per-node)
        last_sync_time = ""
        if since is None:
            row = self.local.conn.execute(
                "SELECT details FROM log WHERE action = 'sync_from_enyal' "
                "ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            if row:
                details = json.loads(row[0])
                since = details.get("last_updated")
                last_sync_time = since or ""
        else:
            last_sync_time = since

        total_synced = 0
        conflicts = 0
        offset = 0
        page = 0

        while True:
            page += 1
            if page > MAX_SYNC_PAGES:
                self.local._log("sync_truncated", {
                    "reason": f"Hit {MAX_SYNC_PAGES} page limit",
                    "synced_so_far": total_synced
                })
                break

            remote = get_knowledge_nodes(
                self.api_key,
                since=since,
                limit=limit,
                offset=offset,
                base_url=self.base_url,
            )
            nodes = remote if isinstance(remote, list) else remote.get('nodes', [])
            if not nodes:
                break

            for node in nodes:
                name_hash = self.local._hash(node['name'])
                existing = self.local.conn.execute(
                    "SELECT id, name, updated_at FROM nodes WHERE name_hash = ?",
                    (name_hash,)
                ).fetchone()

                local_node = None
                if existing:
                    local_node = {
                        "id": existing[0],
                        "name": existing[1],
                        "updated_at": existing[2]
                    }

                action = self._merge_node(
                    local_node, node, last_sync_time, strategy
                )

                if action == "conflict_local_wins":
                    conflicts += 1
                    continue
                elif action in ("created", "updated", "conflict_remote_wins"):
                    props = node.get('properties', '{}')
                    if isinstance(props, str):
                        props = json.loads(props)
                    self.local.remember(
                        node['name'], node['node_type'],
                        node.get('summary'), props
                    )
                    total_synced += 1
                    if "conflict" in action:
                        conflicts += 1

            offset += limit
            if len(nodes) < limit:
                break

        self.local._log("sync_from_enyal", {
            "nodes_synced": total_synced,
            "conflicts": conflicts,
            "since": since,
            "last_updated": datetime.datetime.now(datetime.timezone.utc).isoformat()
        })

        return {"synced": total_synced, "conflicts": conflicts}

    # === INTERNAL ===

    def _derive_snapshot_key(self, password):
        """Derive AES-256 key from password + account identity.
        Salt includes API key hash so different accounts with
        the same password produce different keys."""
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF
        from cryptography.hazmat.primitives import hashes
        account_salt = hashlib.sha256(self.api_key.encode()).hexdigest()[:16]
        return HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=f"enyal-knowledge-snapshot:{account_salt}".encode(),
            info=b"client-side-encryption",
        ).derive(password.encode())

    def _merge_node(self, local_node, remote_node,
                    last_sync_time, strategy):
        """Three-way merge: local vs remote vs last sync time."""
        if local_node is None:
            return "created"

        local_updated = local_node.get("updated_at", "")
        local_modified = (
            self._normalise_ts(local_updated) >
            self._normalise_ts(last_sync_time)
        )

        if not local_modified:
            return "updated"

        # Conflict: local was modified since last sync
        if strategy == "local_wins":
            self.local._log("sync_conflict", {
                "node_name": remote_node.get("name"),
                "resolution": "local_wins"
            })
            return "conflict_local_wins"
        else:
            self.local._log("sync_conflict", {
                "node_name": remote_node.get("name"),
                "resolution": "remote_wins"
            })
            return "conflict_remote_wins"

    def _normalise_ts(self, ts_string):
        """Parse any timestamp format to comparable datetime."""
        if not ts_string:
            return datetime.datetime.min
        for fmt in [
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
        ]:
            try:
                return datetime.datetime.strptime(ts_string, fmt)
            except ValueError:
                continue
        return datetime.datetime.min

    def _extract_from_text(self, text):
        """Extract entity name from natural language text.

        Uses local Ollama if available, falls back to full text as name.
        """
        model = os.environ.get("ENYAL_LOCAL_MODEL", "mistral-nemo")
        try:
            import urllib.request
            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=json.dumps({
                    "model": model,
                    "prompt": (
                        "Extract the entity name from this text. "
                        "Return ONLY the name, nothing else.\n\n"
                        f"Text: {text}"
                    ),
                    "stream": False
                }).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = json.loads(resp.read())
                name = body.get("response", "").strip()
                if name:
                    return name, {"raw_text": text}
        except Exception:
            pass

        # Fallback: use full text as name
        return text.strip(), {"raw_text": text}

    def _type_from_chunk(self, chunk_type):
        mapping = {
            "entity_snapshot": "entity",
            "decision_record": "decision",
            "verification_result": "event",
            "agreement": "event",
            "timestamp": "source",
            "credential": "event",
            "agent_message": "event",
        }
        return mapping.get(chunk_type, "entity")

    def _validate_key_once(self):
        if not self._validated:
            try:
                get_knowledge_stats(self.api_key, base_url=self.base_url)
                self._validated = True
            except Exception:
                raise ValueError("Invalid ENYAL API key")
