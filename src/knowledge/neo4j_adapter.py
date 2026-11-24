"""
Neo4j Graph Database Adapter for Epilepsy Knowledge Graph.
Provides native Neo4j connectivity, Cypher querying, triplet ingestion,
and multi-hop relationship discovery with graceful offline fallback.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
DEFAULT_NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
DEFAULT_NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


class Neo4jGraphDatabase:
    """Manages connection and Cypher operations for Neo4j graph database."""

    def __init__(
        self,
        uri: str = DEFAULT_NEO4J_URI,
        user: str = DEFAULT_NEO4J_USER,
        password: str = DEFAULT_NEO4J_PASSWORD,
    ) -> None:
        self.uri = uri
        self.user = user
        self.password = password
        self._driver = None
        self._connected: Optional[bool] = None

    def connect(self) -> bool:
        """Attempt connection to Neo4j database."""
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            with self._driver.session() as session:
                session.run("RETURN 1 AS test")
            self._connected = True
            logger.info("Successfully connected to Neo4j database at %s", self.uri)
            return True
        except Exception as e:
            self._connected = False
            logger.warning("Neo4j database not reachable at %s (%s). Using local graph engine.", self.uri, e)
            return False

    def is_connected(self) -> bool:
        """Return True if Neo4j driver is active and server is responding."""
        if self._connected is None:
            self.connect()
        return bool(self._connected)

    def close(self) -> None:
        """Close active driver connection."""
        if self._driver:
            self._driver.close()
            self._driver = None
            self._connected = False

    def init_schema(self) -> None:
        """Ensure uniqueness constraints and indexes exist in Neo4j."""
        if not self.is_connected() or not self._driver:
            return
        with self._driver.session() as session:
            session.run("CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
            session.run("CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.name)")

    def import_triplets(self, triplets_df: pd.DataFrame, batch_size: int = 1000) -> int:
        """Bulk import triplets [head, relation, tail] into Neo4j."""
        if not self.is_connected() or not self._driver:
            logger.warning("Cannot import to Neo4j: database is offline.")
            return 0

        self.init_schema()
        imported = 0
        total = len(triplets_df)

        query = """
        UNWIND $batch AS row
        MERGE (h:Entity {name: row.head})
        MERGE (t:Entity {name: row.tail})
        MERGE (h)-[r:RELATION {type: row.relation}]->(t)
        """

        records = [
            {
                "head": str(row.head).strip(),
                "relation": re.sub(r"[^A-Za-z0-9_]", "_", str(row.relation).strip().upper()),
                "tail": str(row.tail).strip(),
            }
            for row in triplets_df.itertuples(index=False)
        ]

        with self._driver.session() as session:
            for i in range(0, total, batch_size):
                batch = records[i:i + batch_size]
                session.run(query, batch=batch)
                imported += len(batch)

        logger.info("Imported %d triplets into Neo4j.", imported)
        return imported

    def get_subgraph(self, entity_name: str, hops: int = 1, limit: int = 50) -> List[str]:
        """Retrieve k-hop subgraph triplets formatted as natural language statements."""
        if not self.is_connected() or not self._driver:
            return []

        cypher = f"""
        MATCH (e:Entity {{name: $name}})-[r:RELATION*1..{hops}]-(target:Entity)
        UNWIND r AS rel
        WITH startNode(rel) AS h, rel, endNode(rel) AS t
        RETURN DISTINCT h.name AS head, rel.type AS relation, t.name AS tail
        LIMIT $limit
        """
        sentences = []
        with self._driver.session() as session:
            result = session.run(cypher, name=entity_name, limit=limit)
            for record in result:
                rel_clean = str(record["relation"]).replace("_", " ").lower()
                sentences.append(f"{record['head']} {rel_clean} {record['tail']}")
        return sorted(set(sentences))

    def find_paths(self, source_name: str, target_name: str, max_depth: int = 3) -> List[List[str]]:
        """Find shortest paths between source and target entities."""
        if not self.is_connected() or not self._driver:
            return []

        cypher = f"""
        MATCH p = allShortestPaths((a:Entity {{name: $src}})-[*..{max_depth}]-(b:Entity {{name: $tgt}}))
        RETURN [n in nodes(p) | n.name] AS path
        LIMIT 10
        """
        paths = []
        with self._driver.session() as session:
            result = session.run(cypher, src=source_name, tgt=target_name)
            for record in result:
                paths.append(record["path"])
        return paths

    def run_cypher(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute arbitrary Cypher query and return records as dictionaries."""
        if not self.is_connected() or not self._driver:
            return []
        with self._driver.session() as session:
            result = session.run(cypher, **(params or {}))
            return [record.data() for record in result]
