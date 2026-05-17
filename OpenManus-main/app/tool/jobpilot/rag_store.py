"""RAGStore — retrieval-augmented generation knowledge base for JobPilot agents.

Uses ChromaDB as the primary backend when available, with an automatic
fallback to a lightweight in-memory Jaccard-similarity store so the system
works without any extra installation.

Backends
--------
* **ChromaBackend** — persistent ChromaDB collections with embedding-based
  semantic search (requires ``pip install chromadb``).
* **InMemoryBackend** — zero-dependency, bag-of-words Jaccard similarity
  over documents kept in plain Python lists.  Best suited for development,
  testing, and environments where chromadb is not available.

Knowledge-base categories
-------------------------
* ``resume_bullets``       — for ResumeOptimizerAgent
* ``interview_qa``         — for InterviewAgent
* ``skill_graph``          — for JDParserAgent
* ``cover_letter_templates`` — for CoverLetterAgent

Typical usage
-------------
>>> store = get_rag_store()
>>> store.add_documents(["Built REST API with FastAPI"], "resume_bullets")
>>> results = store.retrieve("FastAPI backend service", "resume_bullets")
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

from app.logger import logger

# ---------------------------------------------------------------------------
# Built-in seed data  (auto-loaded on first use per category)
# ---------------------------------------------------------------------------

_BUILTIN_SEEDS: Dict[str, List[str]] = {
    "resume_bullets": [
        # Backend
        "Designed and implemented a high-throughput REST API using FastAPI, serving 50K+ requests/day with p99 latency under 80ms.",
        "Refactored monolithic Django application into microservices; reduced deployment time by 40%.",
        "Developed gRPC services in Go for inter-service communication, cutting cross-service latency by 35%.",
        "Built an event-driven pipeline with Kafka and Python, processing 1M+ messages/hour reliably.",
        "Implemented Redis-backed caching layer, reducing database load by 60% and improving API response time by 3×.",
        # ML / AI
        "Trained a BERT-based text classification model achieving 94% F1 on production data; deployed via TorchServe.",
        "Built end-to-end ML pipeline (data ingestion → feature engineering → model training → serving) using MLflow and Airflow.",
        "Implemented RAG system with LangChain and ChromaDB, improving answer relevance by 28% over vanilla LLM baseline.",
        "Reduced model inference latency by 2.5× through ONNX export and TensorRT optimization.",
        "Designed feature store using Feast; enabled real-time and batch feature serving for 10+ ML models.",
        # Data Engineering
        "Built Spark ETL jobs processing 500GB/day of raw logs; cut reporting latency from 24 hours to 2 hours.",
        "Migrated on-prem Hadoop cluster to AWS EMR, reducing infrastructure cost by 30%.",
        "Designed dbt data models with 100+ tests, achieving 99.8% data quality SLA.",
        "Created real-time analytics dashboard with Apache Flink and Grafana, enabling sub-second KPI monitoring.",
        # Frontend
        "Developed React component library used by 6 product teams; improved design consistency and reduced duplication.",
        "Optimized web bundle size from 4.2MB to 890KB through code splitting and tree shaking.",
        "Implemented server-side rendering with Next.js; improved Core Web Vitals LCP by 45%.",
        # Product / General
        "Led cross-functional project with 8 stakeholders, delivering feature 2 weeks ahead of schedule.",
        "Improved CI/CD pipeline reliability from 82% to 99% by adding retry logic and better test isolation.",
        "Wrote comprehensive API documentation; reduced support tickets by 25% within one quarter.",
    ],
    "interview_qa": [
        # System design
        "Q: Design a URL shortener like bit.ly.\nIntent: Assess scalability, hashing, database choice, cache strategy.\nSTAR framework: Discuss hash collision, base62 encoding, Redis cache for hot URLs, sharding for scale, analytics pipeline.",
        "Q: How would you design a distributed rate limiter?\nIntent: Assess knowledge of token bucket/sliding window algorithms, Redis, distributed systems trade-offs.\nSTAR framework: Mention Redis INCR+EXPIRE for simple cases, or Redis + Lua script for atomicity; edge cases: clock skew, Redis failover.",
        "Q: Design a real-time collaborative document editor (like Google Docs).\nIntent: Evaluate operational transformation or CRDT knowledge, WebSocket usage, conflict resolution.\nSTAR framework: Operational Transformation vs CRDT; WebSocket for low-latency sync; optimistic UI; backend merging strategy.",
        # Algorithms
        "Q: Given a list of intervals, merge all overlapping intervals.\nIntent: Sorting, greedy, edge cases.\nSTAR framework: Sort by start time O(n log n); iterate merging if current.start <= prev.end; handle empty list.",
        "Q: Implement LRU Cache with O(1) get and put.\nIntent: Data structure design — doubly linked list + hashmap.\nSTAR framework: HashMap stores key→node; DLL keeps access order; on get, move node to head; on put, evict tail if full.",
        # Behavioral
        "Q: Tell me about a time you disagreed with a technical decision and how you handled it.\nIntent: Assess maturity, communication, data-driven argumentation.\nSTAR: Situation — describe context. Task — your role and concern. Action — how you raised it (data, prototype, alternatives). Result — outcome.",
        "Q: Describe a situation where you had to learn something new quickly to meet a deadline.\nIntent: Learning agility, resourcefulness, ownership.\nSTAR: Situation — tight deadline + unfamiliar tech. Task — your specific deliverable. Action — resources used, pairing sessions. Result — delivered + knowledge sharing afterward.",
        "Q: How do you handle conflicting priorities from multiple stakeholders?\nIntent: Prioritization, stakeholder management, communication.\nSTAR: Acknowledge conflict early, facilitate joint prioritization meeting, use impact/effort matrix, communicate trade-offs clearly.",
        # ML-specific
        "Q: How do you handle class imbalance in a classification problem?\nIntent: Practical ML knowledge — oversampling, undersampling, class weights, threshold tuning.\nSTAR framework: Diagnose imbalance ratio first; SMOTE / class_weight='balanced'; tune decision threshold using PR curve not ROC; mention F1 / MCC as better metrics than accuracy.",
        "Q: Explain the bias-variance trade-off and how it affects model selection.\nIntent: Fundamental ML theory with practical implications.\nSTAR framework: Bias = underfitting (high train error); Variance = overfitting (low train, high val error); regularization, ensemble methods, cross-validation as tools.",
        # Backend-specific
        "Q: What are the trade-offs between SQL and NoSQL databases?\nIntent: Evaluate breadth of data engineering knowledge.\nSTAR framework: SQL — ACID, schema, joins, vertical scale; NoSQL — eventual consistency, flexible schema, horizontal scale; choose based on access pattern + consistency needs.",
        "Q: How would you optimize a slow database query?\nIntent: Assess practical DB tuning knowledge.\nSTAR framework: EXPLAIN / EXPLAIN ANALYZE; add indexes on filter/join columns; avoid SELECT *; consider denormalization or materialized views for read-heavy workloads.",
        # Product / culture
        "Q: Questions to ask the interviewer about engineering culture.\nIntent: Genuine interest in role and team health.\nSample: 'What does a typical oncall rotation look like?', 'How are technical decisions made — bottom-up or top-down?', 'What does the onboarding process look like for new engineers?'",
    ],
    "skill_graph": [
        # Backend
        "Backend Engineer required skills: Python or Go or Java, RESTful API design, SQL (PostgreSQL or MySQL), Git, Linux/Bash, understanding of HTTP and networking fundamentals.",
        "Backend Engineer nice-to-have: Kubernetes/Docker, Redis, Kafka/RabbitMQ, gRPC, CI/CD pipelines, cloud (AWS/GCP/Azure), microservices architecture.",
        "Backend Engineer common responsibilities: design and implement backend services and APIs, optimize database queries, participate in code review, write unit and integration tests, on-call rotation.",
        # ML / AI
        "ML Engineer required skills: Python, PyTorch or TensorFlow, Scikit-learn, SQL, statistics fundamentals, experience training and evaluating supervised models.",
        "ML Engineer nice-to-have: MLflow, Airflow, Spark, feature stores (Feast/Tecton), model serving (TorchServe/BentoML), LLM experience, CUDA/GPU optimization.",
        "ML Engineer responsibilities: develop and maintain ML pipelines, collaborate with data team on feature engineering, monitor model performance in production, run A/B experiments.",
        # Data Engineering
        "Data Engineer required skills: Python or Scala, SQL, Apache Spark or Flink, data warehousing (Snowflake/BigQuery/Redshift), ETL pipeline design.",
        "Data Engineer nice-to-have: dbt, Kafka, Airflow, Terraform, data modeling, stream processing, data quality frameworks.",
        "Data Engineer responsibilities: build and maintain scalable data pipelines, ensure data quality and lineage, collaborate with analytics and ML teams.",
        # Frontend
        "Frontend Engineer required skills: JavaScript/TypeScript, React or Vue, CSS/HTML, REST API consumption, Git.",
        "Frontend Engineer nice-to-have: Next.js, Webpack/Vite, unit testing (Jest/Vitest), accessibility (WCAG), performance optimization (Lighthouse), GraphQL.",
        "Frontend Engineer responsibilities: build responsive web UIs, work closely with design team, write component tests, optimize page performance.",
        # Product Management
        "Product Manager required skills: product strategy, user research, data-driven decision making, roadmap planning, stakeholder communication, Agile/Scrum.",
        "Product Manager nice-to-have: SQL for self-service analytics, A/B testing, UX design basics, technical understanding of APIs and system architecture.",
        # DevOps / SRE
        "DevOps/SRE required skills: Kubernetes, Docker, CI/CD (GitHub Actions/Jenkins), Linux, cloud platforms (AWS/GCP/Azure), monitoring (Prometheus/Grafana).",
        "DevOps/SRE nice-to-have: Terraform/Pulumi, Helm, service mesh (Istio), Ansible, security best practices (IAM, secrets management).",
    ],
    "cover_letter_templates": [
        # Opening hooks
        "Hook (startup): 'When I read that [Company] is building [mission], I immediately thought of the challenge I tackled at [previous company] — [brief parallel]. That alignment between your roadmap and my experience is why I'm excited to apply for this [Role] position.'",
        "Hook (enterprise): 'With [X] years of hands-on experience in [domain], I have consistently delivered [key outcome], and I am eager to bring this expertise to [Company]'s [Team/Initiative].'",
        "Hook (early career): 'As a [Degree] student who has spent the past [time] building [project], I am excited by [Company]'s mission to [goal] and confident that my foundation in [skills] makes me a strong candidate for this [Role] opportunity.'",
        # Value proposition paragraphs
        "Value proposition (backend): 'In my [role] at [company], I designed and shipped [feature/system] that [quantified impact]. This project sharpened my skills in [tech stack], which aligns directly with your requirement for [JD skill].'",
        "Value proposition (ML): 'My most recent project involved [ML task] — I [specific action] and achieved [metric]. Beyond model accuracy, I care deeply about production reliability: I built [MLflow/monitoring/pipeline element] to ensure the system remained healthy in production.'",
        "Value proposition (data): 'I transformed [data challenge] into [business outcome] by designing [pipeline/model/architecture]. The result was [measurable impact], giving [stakeholder] the real-time visibility they needed to [business goal].'",
        # Closing paragraphs
        "Closing (general): 'I would welcome the opportunity to discuss how my background in [skill] and passion for [domain] can contribute to [Company]'s goals. Thank you for your time and consideration — I look forward to connecting.'",
        "Closing (startup): 'I thrive in fast-paced environments where ownership matters and iteration is celebrated. I believe [Company] is exactly the kind of place where I can do my best work, and I am genuinely excited to contribute. Happy to chat at your convenience.'",
        "Closing (enterprise): 'I am confident that my experience in [area] and my commitment to [quality/scale/reliability] align with [Company]'s standards. I look forward to the opportunity to discuss how I can contribute to your [team/division] in detail.'",
        # Subject lines for email
        "Email subject (referral): 'Referred by [Name] — [Role] Application — [Your Name]'",
        "Email subject (direct): 'Application for [Role] — [Your Name] | [Key skill or credential]'",
    ],
}

# ---------------------------------------------------------------------------
# In-memory backend (zero dependencies)
# ---------------------------------------------------------------------------


class _InMemoryBackend:
    """Simple keyword-overlap (Jaccard) similarity store.

    Uses a bag-of-words approach with numpy-free pure Python scoring.
    Suitable for testing and environments without chromadb.
    """

    def __init__(self) -> None:
        self._store: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> set:
        """Lowercase-split text into a set of alpha tokens (≥ 2 chars)."""
        return {w for w in re.split(r"\W+", text.lower()) if len(w) >= 2}

    @classmethod
    def _score(cls, query_tokens: set, doc: str) -> float:
        """Return Jaccard similarity between *query_tokens* and *doc* tokens."""
        doc_tokens = cls._tokenize(doc)
        if not doc_tokens:
            return 0.0
        intersection = len(query_tokens & doc_tokens)
        union = len(query_tokens | doc_tokens)
        return intersection / union if union else 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def count(self, category: str) -> int:
        return len(self._store.get(category, []))

    def add(self, docs: List[str], category: str) -> None:
        self._store.setdefault(category, []).extend(docs)

    def retrieve(self, query: str, category: str, top_k: int) -> List[str]:
        docs = self._store.get(category, [])
        if not docs:
            return []
        query_tokens = self._tokenize(query)
        scored = sorted(
            ((self._score(query_tokens, d), d) for d in docs),
            key=lambda x: x[0],
            reverse=True,
        )
        return [d for _, d in scored[:top_k]]


# ---------------------------------------------------------------------------
# ChromaDB backend (optional)
# ---------------------------------------------------------------------------


class _ChromaBackend:
    """ChromaDB-backed persistent vector store.

    Wraps a separate chromadb collection per category.  Falls back gracefully
    if chromadb is not installed (constructor raises ImportError which the
    RAGStore catches).
    """

    def __init__(self, persist_dir: str) -> None:
        import chromadb  # type: ignore[import]

        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_dir)

    def _collection(self, category: str):  # type: ignore[return]
        return self._client.get_or_create_collection(name=category)

    def count(self, category: str) -> int:
        try:
            return self._collection(category).count()
        except Exception:
            return 0

    def add(self, docs: List[str], category: str) -> None:
        col = self._collection(category)
        existing = col.count()
        ids = [f"{category}_{existing + i}" for i in range(len(docs))]
        col.add(documents=docs, ids=ids)

    def retrieve(self, query: str, category: str, top_k: int) -> List[str]:
        col = self._collection(category)
        if col.count() == 0:
            return []
        results = col.query(query_texts=[query], n_results=min(top_k, col.count()))
        docs = results.get("documents", [[]])[0]
        return docs


# ---------------------------------------------------------------------------
# Public RAGStore facade
# ---------------------------------------------------------------------------


class RAGStore:
    """Knowledge-base facade used by all JobPilot agents.

    On first access to any category the built-in seed data for that category
    is loaded automatically, so the system works out of the box.

    Args:
        persist_dir: Directory for ChromaDB persistence.  If ``None``,
            the store uses ``workspace/jobpilot/rag_db`` relative to the
            project root.  Ignored for the in-memory fallback.
        force_memory: When ``True``, skip ChromaDB and always use the
            in-memory backend.  Useful for testing.
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        *,
        force_memory: bool = False,
    ) -> None:
        self._seeded_categories: set = set()

        if force_memory:
            self._backend: _InMemoryBackend | _ChromaBackend = _InMemoryBackend()
            self._backend_name = "in-memory"
            return

        _dir = persist_dir or str(
            Path(__file__).resolve().parents[4] / "workspace" / "jobpilot" / "rag_db"
        )
        try:
            self._backend = _ChromaBackend(_dir)
            self._backend_name = "chromadb"
            logger.info(f"[RAGStore] Using ChromaDB backend at {_dir}")
        except ImportError:
            self._backend = _InMemoryBackend()
            self._backend_name = "in-memory"
            logger.info(
                "[RAGStore] chromadb not installed — using in-memory fallback. "
                "Install with: pip install chromadb"
            )
        except Exception as exc:
            self._backend = _InMemoryBackend()
            self._backend_name = "in-memory"
            logger.warning(
                f"[RAGStore] ChromaDB init failed ({exc}); falling back to in-memory."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def add_documents(
        self,
        docs: List[str],
        category: str,
        *,
        ids: Optional[List[str]] = None,
    ) -> None:
        """Add *docs* to *category*.

        Args:
            docs: List of document strings.
            category: One of the four JobPilot categories or any custom string.
            ids: Optional list of unique IDs (ChromaDB only; ignored for
                 in-memory backend).
        """
        if not docs:
            return
        self._backend.add(docs, category)
        logger.debug(f"[RAGStore] Added {len(docs)} docs to '{category}'")

    def retrieve(
        self,
        query: str,
        category: str,
        top_k: int = 5,
    ) -> List[str]:
        """Return the top-*k* most relevant documents for *query* in *category*.

        Auto-seeds the category with built-in samples on first call.

        Args:
            query: Natural-language query string.
            category: Knowledge-base category to search.
            top_k: Maximum number of results to return.

        Returns:
            List of document strings ordered by relevance (most relevant first).
        """
        self._ensure_seeded(category)
        results = self._backend.retrieve(query, category, top_k)
        logger.debug(
            f"[RAGStore] Retrieved {len(results)} docs from '{category}' for query: {query[:60]}"
        )
        return results

    def clear(self, category: str) -> None:
        """Clear all documents in *category* (in-memory backend only)."""
        if isinstance(self._backend, _InMemoryBackend):
            self._backend._store.pop(category, None)
        self._seeded_categories.discard(category)

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------

    def _ensure_seeded(self, category: str) -> None:
        """Seed *category* with built-in samples if it is empty."""
        if category in self._seeded_categories:
            return
        if self._backend.count(category) == 0:
            seeds = _BUILTIN_SEEDS.get(category, [])
            if seeds:
                self._backend.add(seeds, category)
                logger.info(
                    f"[RAGStore] Auto-seeded '{category}' with {len(seeds)} built-in documents."
                )
        self._seeded_categories.add(category)

    def seed_from_list(self, docs: List[str], category: str) -> None:
        """Explicitly seed *category* with *docs* (replaces existing for in-memory)."""
        if isinstance(self._backend, _InMemoryBackend):
            self._backend._store[category] = list(docs)
        else:
            self._backend.add(docs, category)
        self._seeded_categories.add(category)
        logger.info(f"[RAGStore] Seeded '{category}' with {len(docs)} documents.")


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_store: Optional[RAGStore] = None


def get_rag_store(
    persist_dir: Optional[str] = None,
    *,
    force_memory: bool = False,
) -> RAGStore:
    """Return the module-level RAGStore singleton, creating it if necessary.

    Args:
        persist_dir: Override the default ChromaDB persistence directory.
            Only applied on the **first** call; subsequent calls return the
            already-created instance.
        force_memory: When ``True``, force the in-memory backend.  Only
            applied on the first call.
    """
    global _store
    if _store is None:
        _store = RAGStore(persist_dir=persist_dir, force_memory=force_memory)
    return _store


def reset_rag_store() -> None:
    """Reset the module-level singleton (useful for testing)."""
    global _store
    _store = None
