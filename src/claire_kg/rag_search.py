"""
RAG (retrieval-augmented generation) search: vector similarity over KG nodes.

Uses OpenAI embeddings for the query (and optionally a reference node) and
Neo4j Cypher to compute cosine similarity against stored node embeddings.
Used by LLMOrchestrator when QuestionClassifier.should_use_rag() is True
(e.g. "vulnerabilities similar to CVE-X", conceptual queries).

Module layout: SimilarityResult dataclass → RAGSearch (find_similar,
find_similar_by_uid, check_embeddings_available). Requires OPENAI_API_KEY and
nodes with an `embedding` property (see ingest embedding generation).
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .database import Neo4jConnection

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# SimilarityResult and RAGSearch
# -----------------------------------------------------------------------------


@dataclass
class SimilarityResult:
    """Single hit from vector similarity search: uid, name, score, optional description."""

    uid: str
    name: str
    similarity: float
    description: Optional[str] = None


class RAGSearch:
    """Vector similarity search over KG nodes using OpenAI embeddings and Neo4j Cypher."""

    def __init__(self, db: Neo4jConnection, model: str = "text-embedding-3-small"):
        """Initialize with Neo4j connection and OpenAI embedding model name.

        Args:
            db: Neo4j database connection
            model: OpenAI embedding model (default: text-embedding-3-small)
        """
        self.db = db
        self.model_name = model
        self._openai_client = None

    @property
    def openai_client(self):
        """Lazy load OpenAI client."""
        if self._openai_client is None:
            from openai import OpenAI
            import os
            from dotenv import load_dotenv

            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")

            self._openai_client = OpenAI(api_key=api_key)
            logger.info(
                f"Initialized OpenAI client for embeddings (model: {self.model_name})"
            )
        return self._openai_client

    def find_similar(
        self,
        query_text: str,
        node_type: str = "Vulnerability",
        top_k: int = 10,
        min_similarity: float = 0.0,
    ) -> List[SimilarityResult]:
        """Find similar nodes using vector similarity search.

        Args:
            query_text: Natural language query text
            node_type: Node type to search (Vulnerability, AttackPattern, Technique, Weakness)
            top_k: Number of results to return
            min_similarity: Minimum similarity score threshold (0.0-1.0)

        Returns:
            List of SimilarityResult objects sorted by similarity (descending)
        """
        response = self.openai_client.embeddings.create(
            model=self.model_name,
            input=[query_text],
        )
        query_embedding = response.data[0].embedding

        # Cosine similarity in Cypher: dot product / (||query|| * ||node||)
        cypher = f"""
        MATCH (n:{node_type})
        WHERE n.embedding IS NOT NULL
        WITH n, 
             reduce(dot = 0.0, i in range(0, size($query_embedding)-1) | 
               dot + $query_embedding[i] * n.embedding[i]) as dot_product,
             sqrt(reduce(sum1 = 0.0, x in $query_embedding | sum1 + x * x)) as query_mag,
             sqrt(reduce(sum2 = 0.0, x in n.embedding | sum2 + x * x)) as node_mag
        WHERE query_mag > 0 AND node_mag > 0
        WITH n, dot_product / (query_mag * node_mag) as similarity
        WHERE similarity >= $min_similarity
        ORDER BY similarity DESC
        LIMIT $top_k
        RETURN n.uid as uid, 
               COALESCE(n.name, '') as name, 
               similarity,
               COALESCE(n.description, n.descriptions, '') as description
        """

        # Ensure list for Neo4j parameter (OpenAI returns list; support numpy if passed)
        if hasattr(query_embedding, "tolist"):
            query_embedding_list = query_embedding.tolist()
        else:
            query_embedding_list = query_embedding

        results = self.db.execute_cypher(
            cypher,
            {
                "query_embedding": query_embedding_list,
                "top_k": top_k,
                "min_similarity": min_similarity,
            },
        )

        return [
            SimilarityResult(
                uid=r.get("uid", ""),
                name=r.get("name", ""),
                similarity=float(r.get("similarity", 0.0)),
                description=r.get("description") or None,
            )
            for r in results
        ]

    def find_similar_by_uid(
        self,
        reference_uid: str,
        node_type: str = "Vulnerability",
        top_k: int = 10,
        min_similarity: float = 0.0,
    ) -> List[SimilarityResult]:
        """Find nodes similar to a reference node by UID (uses reference node's embedding).

        Args:
            reference_uid: UID of the reference node (e.g., 'CVE-2024-20439')
            node_type: Node type to search (Vulnerability, AttackPattern, Technique, Weakness)
            top_k: Number of results to return
            min_similarity: Minimum similarity score threshold (0.0-1.0)

        Returns:
            List of SimilarityResult objects sorted by similarity (descending)
        """
        # Cosine similarity: query = reference node's embedding; exclude self
        cypher = f"""
        MATCH (ref:{node_type} {{uid: $reference_uid}})
        WHERE ref.embedding IS NOT NULL
        MATCH (n:{node_type})
        WHERE n.embedding IS NOT NULL AND n.uid <> $reference_uid
        WITH ref, n,
             reduce(dot = 0.0, i in range(0, size(ref.embedding)-1) | 
               dot + ref.embedding[i] * n.embedding[i]) as dot_product,
             sqrt(reduce(sum1 = 0.0, x in ref.embedding | sum1 + x * x)) as ref_mag,
             sqrt(reduce(sum2 = 0.0, x in n.embedding | sum2 + x * x)) as node_mag
        WHERE ref_mag > 0 AND node_mag > 0
        WITH n, dot_product / (ref_mag * node_mag) as similarity
        WHERE similarity >= $min_similarity
        ORDER BY similarity DESC
        LIMIT $top_k
        RETURN n.uid as uid, 
               COALESCE(n.name, '') as name, 
               similarity,
               COALESCE(n.description, n.descriptions, '') as description
        """

        results = self.db.execute_cypher(
            cypher,
            {
                "reference_uid": reference_uid,
                "top_k": top_k,
                "min_similarity": min_similarity,
            },
        )

        return [
            SimilarityResult(
                uid=r.get("uid", ""),
                name=r.get("name", ""),
                similarity=float(r.get("similarity", 0.0)),
                description=r.get("description") or None,
            )
            for r in results
        ]

    def check_embeddings_available(
        self, node_type: str = "Vulnerability"
    ) -> Dict[str, Any]:
        """Return counts and coverage for node type's embedding property.

        Args:
            node_type: Label to check (e.g. Vulnerability, Weakness)

        Returns:
            Dict with total_nodes, nodes_with_embeddings, coverage_percent, embedding_size, available
        """
        cypher = f"""
        MATCH (n:{node_type})
        RETURN count(n) as total,
               count(n.embedding) as with_embeddings,
               CASE WHEN count(n.embedding) > 0 
                    THEN collect(DISTINCT size(n.embedding))[0]
                    ELSE NULL END as embedding_size
        """

        result = self.db.execute_cypher_single(cypher)

        if result:
            total = result.get("total", 0)
            with_embeddings = result.get("with_embeddings", 0)
            embedding_size = result.get("embedding_size")

            return {
                "node_type": node_type,
                "total_nodes": total,
                "nodes_with_embeddings": with_embeddings,
                "coverage_percent": (
                    (with_embeddings / total * 100) if total > 0 else 0.0
                ),
                "embedding_size": embedding_size,
                "available": with_embeddings > 0,
            }

        return {
            "node_type": node_type,
            "total_nodes": 0,
            "nodes_with_embeddings": 0,
            "coverage_percent": 0.0,
            "embedding_size": None,
            "available": False,
        }
