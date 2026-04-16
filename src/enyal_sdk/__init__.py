"""
ENYAL SDK — encrypted knowledge graph, local memory, permanent proof.

    from enyal_sdk import EnyalAgent

    agent = EnyalAgent(api_key="eyl_your_key")
    agent.remember("Tesla has 100GWh capacity")
    agent.archive(chunk_type="decision_record",
                  chunk_key="my-agent:decision:001",
                  data={"decision": "Invest in energy"})
"""

__version__ = "2.1.0"

# Client functions (API wrappers, crypto, retry)
from .client import (
    archive,
    search,
    prove,
    disclose,
    timestamp,
    create_agreement,
    verify_agreement,
    get_lineage,
    compliance_attest,
    send_message,
    get_inbox,
    get_thread,
    mark_read,
    get_knowledge_nodes,
    get_knowledge_node,
    get_knowledge_connections,
    get_contradictions,
    get_knowledge_stats,
    get_knowledge_index,
    get_knowledge_health,
    synthesise_knowledge,
    # Tier 1/2 disclosure + proof
    request_client_disclosure,
    decrypt_custodial_share,
    combine_shares_and_decrypt,
    verify_share_combination,
    request_share_proof,
    # Crypto utilities
    shamir_combine,
)

# Agent class
from .agent import EnyalAgent

# Local knowledge graph
from .local_knowledge import LocalKnowledgeGraph

__all__ = [
    "__version__",
    "EnyalAgent",
    "LocalKnowledgeGraph",
    "archive",
    "search",
    "prove",
    "disclose",
    "timestamp",
    "create_agreement",
    "verify_agreement",
    "get_lineage",
    "compliance_attest",
    "send_message",
    "get_inbox",
    "get_thread",
    "mark_read",
    "get_knowledge_nodes",
    "get_knowledge_node",
    "get_knowledge_connections",
    "get_contradictions",
    "get_knowledge_stats",
    "get_knowledge_index",
    "get_knowledge_health",
    "synthesise_knowledge",
    "request_client_disclosure",
    "decrypt_custodial_share",
    "combine_shares_and_decrypt",
    "verify_share_combination",
    "request_share_proof",
    "shamir_combine",
]
