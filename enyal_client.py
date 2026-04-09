"""
ENYAL Client SDK — Three-tier client-side disclosure, verification, and proof.

TIER 1 — Local verification (zero trust in ENYAL)
  request_client_disclosure, decrypt_custodial_share, combine_shares_and_decrypt, verify_share_combination

TIER 2 — Proof server (trust ENYAL during proof generation, proof for auditors)
  request_share_proof

TIER 3 — Self-hosted (documented in README, not in SDK)
  Clone shamir-circuit repo, cargo build --release, run locally.

Dependencies: cryptography (pip install cryptography), requests
"""

import base64
import hashlib
import hmac
import json
import struct
import urllib.request
from typing import Optional

# ────────────────────────────────────────────────────────────────
# GF(256) Arithmetic — identical to enyal/shamir.py
# Generator = 3, irreducible polynomial = 0x11B (same as AES)
# ────────────────────────────────────────────────────────────────

_GF256_EXP = [0] * 512
_GF256_LOG = [0] * 256

def _init_gf256():
    x = 1
    for i in range(255):
        _GF256_EXP[i] = x
        _GF256_LOG[x] = i
        hi = x << 1
        if hi & 0x100:
            hi ^= 0x11B
        x = hi ^ x
    for i in range(255, 512):
        _GF256_EXP[i] = _GF256_EXP[i - 255]

_init_gf256()


def _gf_mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return _GF256_EXP[(_GF256_LOG[a] + _GF256_LOG[b]) % 255]


def _gf_inv(a: int) -> int:
    if a == 0:
        raise ValueError("Zero has no inverse in GF(256)")
    return _GF256_EXP[255 - _GF256_LOG[a]]


def shamir_combine(share1: bytes, share2: bytes) -> bytes:
    """Shamir Lagrange interpolation at x=0 for two shares.

    Each share: [index_byte, data_0, ..., data_31] = 33 bytes.
    Returns: 32-byte reconstructed secret.
    """
    if len(share1) != 33 or len(share2) != 33:
        raise ValueError("Share combination failed — each share must be 33 bytes")
    x1, x2 = share1[0], share2[0]
    if x1 == 0 or x2 == 0 or x1 == x2:
        raise ValueError("Share combination failed — invalid share indices")
    d = x1 ^ x2
    d_inv = _gf_inv(d)
    secret = bytearray(32)
    for i in range(32):
        y1, y2 = share1[1 + i], share2[1 + i]
        num = _gf_mul(y1, x2) ^ _gf_mul(y2, x1)
        secret[i] = _gf_mul(num, d_inv)
    return bytes(secret)


# ────────────────────────────────────────────────────────────────
# Crypto Helpers
# ────────────────────────────────────────────────────────────────

def _memory_kdf(shared_secret: bytes) -> bytes:
    """HKDF-SHA256 matching enyal's bsv_memory._memory_kdf.

    PRK = HMAC-SHA256(salt=zeros(32), IKM=shared_secret)
    OKM = HMAC-SHA256(PRK, "joulepai-memory-v1" || 0x01)
    """
    prk = hmac.new(b"\x00" * 32, shared_secret, hashlib.sha256).digest()
    okm = hmac.new(prk, b"joulepai-memory-v1\x01", hashlib.sha256).digest()
    return okm


def _ecdh_shared_secret(private_key_bytes: bytes, peer_public_key_compressed: bytes) -> bytes:
    """P-256 ECDH: scalar multiply private key × peer public key.

    Returns 32-byte x-coordinate of shared point.
    Requires: pip install cryptography
    """
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
    from cryptography.hazmat.backends import default_backend

    # Import private key
    private_key = ec.derive_private_key(
        int.from_bytes(private_key_bytes, "big"),
        ec.SECP256R1(),
        default_backend(),
    )

    # Decompress public key if needed
    if len(peer_public_key_compressed) == 33:
        peer_pub = ec.EllipticCurvePublicKey.from_encoded_point(
            ec.SECP256R1(), peer_public_key_compressed
        )
    elif len(peer_public_key_compressed) == 65:
        peer_pub = ec.EllipticCurvePublicKey.from_encoded_point(
            ec.SECP256R1(), peer_public_key_compressed
        )
    else:
        raise ValueError(f"Invalid public key length: {len(peer_public_key_compressed)}")

    # ECDH
    shared_key = private_key.exchange(ec.ECDH(), peer_pub)
    return shared_key  # 32 bytes (x-coordinate)


def _aes_gcm_decrypt(key: bytes, iv: bytes, ciphertext: bytes, tag: bytes) -> bytes:
    """AES-256-GCM decrypt. On auth tag mismatch, raises user-friendly error."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    aes = AESGCM(key)
    try:
        return aes.decrypt(iv, ciphertext + tag, None)
    except Exception:
        raise ValueError(
            "Share combination failed — invalid recovery phrase or share. "
            "Please verify your recovery phrase and try again."
        )


# ────────────────────────────────────────────────────────────────
# TIER 1 — Client-side (zero trust in ENYAL)
# ────────────────────────────────────────────────────────────────

def request_client_disclosure(
    api_key: str,
    base_url: str,
    chunk_ids: list[str],
    purpose: str,
) -> dict:
    """1. Request client-side disclosure materials from ENYAL.

    Returns encrypted chunks + ECDH-encrypted custodial share + poseidon_key_hash.
    No decryption on server. Customer share is NOT sent.
    """
    body = json.dumps({"chunk_ids": chunk_ids, "purpose": purpose}).encode()
    req = urllib.request.Request(
        f"{base_url}/api/v1/disclose/client-side",
        data=body,
        headers={"Content-Type": "application/json", "X-API-Key": api_key},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        detail = json.loads(e.read()).get("detail", str(e)) if e.fp else str(e)
        raise RuntimeError(f"Disclosure failed ({e.code}): {detail}")


def decrypt_custodial_share(
    encrypted_share: dict,
    customer_private_key_bytes: bytes,
) -> bytes:
    """2. Decrypt the custodial share using your P-256 private key.

    The custodial share was ECDH-encrypted with your registered public key.

    Args:
        encrypted_share: custodial_share dict from disclosure response
        customer_private_key_bytes: 32-byte P-256 private key

    Returns:
        33-byte custodial share (index + data)
    """
    ephem_pub = bytes.fromhex(encrypted_share["ephemeral_pubkey_hex"])
    iv = bytes.fromhex(encrypted_share["iv_hex"])
    tag = bytes.fromhex(encrypted_share["tag_hex"])
    ct = base64.b64decode(encrypted_share["encrypted_share"])

    shared_secret = _ecdh_shared_secret(customer_private_key_bytes, ephem_pub)
    aes_key = _memory_kdf(shared_secret)
    return _aes_gcm_decrypt(aes_key, iv, ct, tag)


def combine_shares_and_decrypt(
    customer_share: bytes,
    custodial_share: bytes,
    chunk: dict,
) -> bytes:
    """3. Combine shares and decrypt a chunk.

    GF(256) Lagrange interpolation → reconstruct private key → ECDH → AES-GCM decrypt.
    On wrong share: AES-GCM auth tag mismatch → clear error message.

    Args:
        customer_share: 33-byte customer share (index + data)
        custodial_share: 33-byte custodial share
        chunk: chunk dict from disclosure response (encrypted_payload + encryption_metadata)

    Returns:
        Decrypted plaintext bytes
    """
    private_key = shamir_combine(customer_share, custodial_share)
    meta = chunk["encryption_metadata"]
    ephem_pub = bytes.fromhex(meta["ecdh_public_key_hex"])
    iv = bytes.fromhex(meta["iv_hex"])
    tag = bytes.fromhex(meta["tag_hex"])
    ct = base64.b64decode(chunk["encrypted_payload"])

    shared_secret = _ecdh_shared_secret(private_key, ephem_pub)
    aes_key = _memory_kdf(shared_secret)
    return _aes_gcm_decrypt(aes_key, iv, ct, tag)


def verify_share_combination(
    customer_share: bytes,
    custodial_share: bytes,
    poseidon_key_hash: str,
    binary_path: Optional[str] = None,
) -> dict:
    """4. Verify share combination locally.

    Reconstructs the secret via GF(256) Lagrange interpolation, computes
    content integrity hash, and compares against expected hash.

    For hash computation, calls the shamir-circuit binary if available,
    otherwise falls back to a pure-Python field operation check.

    Args:
        customer_share: 33-byte customer share
        custodial_share: 33-byte custodial share
        poseidon_key_hash: 64-char hex of expected key hash
        binary_path: optional path to shamir-circuit binary for hash computation

    Returns:
        {"valid": bool, "reconstructed_hash": str, "expected_hash": str}
    """
    secret = shamir_combine(customer_share, custodial_share)

    if binary_path:
        # Use the Rust binary for exact Poseidon hash
        import subprocess
        input_json = json.dumps({
            "command": "poseidon_hash",
            "data_hex": secret.hex(),
        })
        proc = subprocess.run(
            [binary_path],
            input=input_json, capture_output=True, text=True, timeout=10,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Hash computation failed: {proc.stderr}")
        result = json.loads(proc.stdout)
        computed_hash = result["poseidon_hash_hex"]
    else:
        # Without the binary, we can't compute Poseidon locally in Python.
        # Return a partial result indicating the shares combine successfully
        # but content integrity hash requires the binary or WASM module.
        return {
            "valid": None,
            "reconstructed_key_hex": secret.hex(),
            "expected_hash": poseidon_key_hash,
            "note": "Hash verification requires shamir-circuit binary or WASM module. "
                    "Provide binary_path for full verification.",
        }

    valid = computed_hash.lower() == poseidon_key_hash.strip().lower()
    return {
        "valid": valid,
        "reconstructed_hash": computed_hash,
        "expected_hash": poseidon_key_hash,
    }


# ────────────────────────────────────────────────────────────────
# TIER 2 — Proof server (trust ENYAL during proof generation)
# ────────────────────────────────────────────────────────────────

def request_share_proof(
    api_key: str,
    base_url: str,
    customer_share_hex: str,
    poseidon_key_hash: Optional[str] = None,
) -> dict:
    """5. Request a cryptographic share combination proof from ENYAL (Tier 2).

    Customer sends their share to the server. ENYAL retrieves its custodial share,
    generates a zero-knowledge proof, then wipes both shares from memory.

    NOTE: This is NOT zero-knowledge to ENYAL. ENYAL sees the share during
    proof generation. The proof is for third-party auditors.
    For zero-trust verification, use verify_share_combination (Tier 1).
    For zero-trust proof generation, use Tier 3 (self-hosted Rust binary).

    Returns:
        Proof result dict including share_attestation with wipe confirmation.
    """
    body_dict = {"customer_share_hex": customer_share_hex}
    if poseidon_key_hash:
        body_dict["poseidon_key_hash"] = poseidon_key_hash
    body = json.dumps(body_dict).encode()

    req = urllib.request.Request(
        f"{base_url}/api/v1/prove/share-combination",
        data=body,
        headers={"Content-Type": "application/json", "X-API-Key": api_key},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        detail = json.loads(e.read()).get("detail", str(e)) if e.fp else str(e)
        raise RuntimeError(f"Proof generation failed ({e.code}): {detail}")


# ────────────────────────────────────────────────────────────────
# API Wrappers — Core Operations
# ────────────────────────────────────────────────────────────────

DEFAULT_BASE_URL = "https://api.enyal.ai"


def _api_call(api_key: str, method: str, path: str, body: dict = None,
              params: dict = None, base_url: str = DEFAULT_BASE_URL, timeout: int = 30) -> dict:
    """Generic API call helper. Returns parsed JSON response."""
    url = f"{base_url}{path}"
    if params:
        qs = "&".join(f"{k}={urllib.parse.quote(str(v))}" for k, v in params.items() if v is not None)
        url = f"{url}?{qs}"
    data = json.dumps(body).encode() if body else None
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        detail = json.loads(e.read()).get("detail", str(e)) if e.fp else str(e)
        raise RuntimeError(f"API call failed ({e.code}): {detail}")


def archive(api_key: str, agent_id: str, chunk_type: str, chunk_key: str,
            data, base_url: str = DEFAULT_BASE_URL, **metadata) -> dict:
    """Archive to ENYAL's immutable ledger. Returns {chunk_id, status, joule_cost}."""
    body = {"agent_id": agent_id, "chunk_type": chunk_type, "chunk_key": chunk_key, "data": data, **metadata}
    return _api_call(api_key, "POST", "/api/v1/archive", body, base_url=base_url)


def search(api_key: str, query: str = None, chunk_type: str = None, entity: str = None,
           since: str = None, until: str = None, limit: int = 20,
           base_url: str = DEFAULT_BASE_URL) -> dict:
    """Search archived intelligence. Returns {results, total, search_mode}."""
    params = {k: v for k, v in {"q": query, "chunk_type": chunk_type, "entity": entity,
              "since": since, "until": until, "limit": limit}.items() if v is not None}
    return _api_call(api_key, "GET", "/api/v1/search", params=params, base_url=base_url)


def prove(api_key: str, resource_type: str, geographic_region: str = None,
          quantum_resistant: bool = False, base_url: str = DEFAULT_BASE_URL) -> dict:
    """Generate a ZK proof of archived intelligence. Returns proof + metadata."""
    body = {"resource_type": resource_type, "quantum_resistant": quantum_resistant}
    if geographic_region:
        body["geographic_region"] = geographic_region
    return _api_call(api_key, "POST", "/api/v1/prove", body, base_url=base_url, timeout=60)


def disclose(api_key: str, chunk_ids: list, recipient_pubkey_hex: str, purpose: str,
             include_content_proof: bool = False, proof_hash_type: str = "poseidon",
             base_url: str = DEFAULT_BASE_URL) -> dict:
    """Server-side disclosure — re-encrypts chunks for a recipient."""
    body = {"chunk_ids": chunk_ids, "recipient_pubkey_hex": recipient_pubkey_hex,
            "purpose": purpose, "include_content_proof": include_content_proof,
            "proof_hash_type": proof_hash_type}
    return _api_call(api_key, "POST", "/api/v1/disclose", body, base_url=base_url, timeout=60)


# ────────────────────────────────────────────────────────────────
# API Wrappers — Trust Endpoints
# ────────────────────────────────────────────────────────────────

def timestamp(api_key: str, payload, description: str = None,
              base_url: str = DEFAULT_BASE_URL) -> dict:
    """Timestamp anchored to ENYAL's immutable ledger. Returns {chunk_id, transaction_id}."""
    body = {"payload": payload}
    if description:
        body["description"] = description
    return _api_call(api_key, "POST", "/api/v1/timestamp", body, base_url=base_url)


def create_agreement(api_key: str, terms: str, parties: list, title: str = None,
                     base_url: str = DEFAULT_BASE_URL) -> dict:
    """Create a multi-party agreement anchored to ENYAL's immutable ledger."""
    body = {"terms": terms, "parties": parties}
    if title:
        body["title"] = title
    return _api_call(api_key, "POST", "/api/v1/agreement/create", body, base_url=base_url)


def verify_agreement(api_key: str, agreement_chunk_id: str, terms: str,
                     base_url: str = DEFAULT_BASE_URL) -> dict:
    """Verify an agreement against its terms anchored to ENYAL's immutable ledger."""
    return _api_call(api_key, "POST", "/api/v1/agreement/verify",
                     {"agreement_chunk_id": agreement_chunk_id, "terms": terms}, base_url=base_url)


def get_lineage(api_key: str, chunk_id: str, base_url: str = DEFAULT_BASE_URL) -> dict:
    """Get the provenance lineage chain for a chunk."""
    return _api_call(api_key, "GET", f"/api/v1/lineage/{chunk_id}", base_url=base_url)


def compliance_attest(api_key: str, period_start: str, period_end: str, systems: list,
                      base_url: str = DEFAULT_BASE_URL) -> dict:
    """Generate a compliance attestation report. Returns {attestation_id, tx_id}."""
    return _api_call(api_key, "POST", "/api/v1/compliance/attest",
                     {"period_start": period_start, "period_end": period_end, "systems": systems},
                     base_url=base_url)


# ────────────────────────────────────────────────────────────────
# API Wrappers — Agent Messaging
# ────────────────────────────────────────────────────────────────

def send_message(api_key: str, sender_agent_id: str, thread_id: str,
                 recipient_agent_id: str, message_type: str, payload: dict,
                 expires_at: str = None, base_url: str = DEFAULT_BASE_URL) -> dict:
    """Send an agent-to-agent message. Cost: 10 joules."""
    body = {"sender_agent_id": sender_agent_id, "thread_id": thread_id,
            "recipient_agent_id": recipient_agent_id, "message_type": message_type,
            "payload": payload}
    if expires_at:
        body["expires_at"] = expires_at
    return _api_call(api_key, "POST", "/api/v1/message/send", body, base_url=base_url)


def get_inbox(api_key: str, agent_id: str, direction: str = "inbox",
              thread_id: str = None, message_type: str = None,
              since: str = None, limit: int = 20,
              base_url: str = DEFAULT_BASE_URL) -> dict:
    """Retrieve messages for an agent. direction: inbox|outbox|all."""
    params = {"agent_id": agent_id, "direction": direction, "limit": limit}
    if thread_id:
        params["thread_id"] = thread_id
    if message_type:
        params["message_type"] = message_type
    if since:
        params["since"] = since
    return _api_call(api_key, "GET", "/api/v1/message/inbox", params=params, base_url=base_url)


def get_thread(api_key: str, thread_id: str,
               base_url: str = DEFAULT_BASE_URL) -> dict:
    """Retrieve all messages in a thread, ordered by sequence."""
    return _api_call(api_key, "GET", f"/api/v1/message/thread/{thread_id}", base_url=base_url)


def mark_read(api_key: str, message_ids: list,
              base_url: str = DEFAULT_BASE_URL) -> dict:
    """Mark messages as read."""
    return _api_call(api_key, "POST", "/api/v1/message/read",
                     {"message_ids": message_ids}, base_url=base_url)


# ────────────────────────────────────────────────────────────────
# Knowledge Base — browsable wiki auto-built from archived chunks
# ────────────────────────────────────────────────────────────────

def get_knowledge_nodes(api_key: str, node_type: str = None, search: str = None,
                        limit: int = 50, offset: int = 0, since: str = None,
                        base_url: str = DEFAULT_BASE_URL) -> list:
    """List knowledge nodes. Filter by node_type, search by name, paginate with offset/limit.

    Args:
        since: ISO timestamp — only return nodes updated after this time.
        offset: Skip first N results (for pagination).
    """
    params = {k: v for k, v in {"node_type": node_type, "search": search,
              "limit": limit, "offset": offset, "since": since}.items() if v is not None}
    if "offset" in params and params["offset"] == 0:
        del params["offset"]
    return _api_call(api_key, "GET", "/api/v1/knowledge/nodes", params=params, base_url=base_url)


def get_knowledge_node(api_key: str, node_id: str,
                       base_url: str = DEFAULT_BASE_URL) -> dict:
    """Get a single node with all its edges."""
    return _api_call(api_key, "GET", f"/api/v1/knowledge/node/{node_id}", base_url=base_url)


def get_knowledge_connections(api_key: str, node_id: str, hops: int = 2,
                              base_url: str = DEFAULT_BASE_URL) -> dict:
    """Get nodes connected within N hops. Returns {seed, nodes, edges}."""
    return _api_call(api_key, "GET", f"/api/v1/knowledge/node/{node_id}/connections",
                     params={"hops": hops}, base_url=base_url)


def get_contradictions(api_key: str,
                       base_url: str = DEFAULT_BASE_URL) -> list:
    """Get all contradiction edges — the 'what disagrees' view."""
    return _api_call(api_key, "GET", "/api/v1/knowledge/contradictions", base_url=base_url)


def get_knowledge_stats(api_key: str,
                        base_url: str = DEFAULT_BASE_URL) -> dict:
    """Knowledge base summary: node counts by type, edge counts, contradictions."""
    return _api_call(api_key, "GET", "/api/v1/knowledge/stats", base_url=base_url)


def get_knowledge_index(api_key: str,
                        base_url: str = DEFAULT_BASE_URL) -> dict:
    """Grouped overview of the knowledge base with connection counts.

    Returns nodes grouped by type (entities, decisions, events, etc.),
    plus contradictions, total_nodes, total_edges, and last_updated.
    Results are cached server-side for 60 seconds.
    """
    return _api_call(api_key, "GET", "/api/v1/knowledge/index", base_url=base_url)


def get_knowledge_health(api_key: str,
                         base_url: str = DEFAULT_BASE_URL) -> dict:
    """Knowledge base health assessment.

    Returns:
        status: 'healthy' | 'needs_attention' | 'unhealthy'
        contradictions: int
        orphan_nodes: int
        gaps: list of entities mentioned in decisions but missing entity_snapshot
        stale_nodes: int (not updated in 30+ days)
        suggested_actions: list of remediation strings
    """
    return _api_call(api_key, "GET", "/api/v1/knowledge/health", base_url=base_url)


def synthesise_knowledge(api_key: str, query: str, node_ids: list,
                         base_url: str = DEFAULT_BASE_URL) -> dict:
    """Combine multiple knowledge nodes into a synthesis. Cost: 5 joules.

    Creates a new 'synthesis' node with 'informed_by' edges to each source.

    Args:
        query: synthesis question (max 500 chars)
        node_ids: list of 1-20 node UUIDs to synthesise

    Returns:
        {id, name, node_type, summary, source_nodes, edges_created, cost}
    """
    return _api_call(api_key, "POST", "/api/v1/knowledge/synthesise",
                     {"query": query, "node_ids": node_ids}, base_url=base_url)
