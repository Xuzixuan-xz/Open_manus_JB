#!/usr/bin/env python3
"""Build (or rebuild) the JobPilot RAG knowledge base from documents.

Usage
-----
Run from the ``OpenManus-main/`` directory:

    python scripts/build_knowledge_base.py [--reset] [--dir PATH]

Options
-------
--reset         Clear all existing documents before seeding.
--dir PATH      Override the default ChromaDB persistence directory
                (default: workspace/jobpilot/rag_db).

The script seeds the four knowledge-base categories with built-in samples
automatically.  You can also add your own documents by editing the
``CUSTOM_DOCS`` dictionary below — just append strings to each category list.

To add documents from files (PDF, DOCX, plain text), use the helper section
at the bottom of this script.

Categories
----------
* resume_bullets          — high-impact resume bullet point examples
* interview_qa            — interview Q&As with answer frameworks
* skill_graph             — standard skill sets per role type
* cover_letter_templates  — opening hooks, value props, and closing paragraphs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Make sure the app package is importable when running from the repo root
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Mock optional heavy/third-party modules that may not be installed.
# RAGStore itself has no such dependencies; the mocks are only needed because
# importing app.tool.jobpilot.rag_store causes Python to first execute the
# parent app/tool/__init__.py, which eagerly imports browser_use etc.
# ---------------------------------------------------------------------------
for _mod in (
    "browser_use",
    "browser_use.browser",
    "browser_use.browser.context",
    "browser_use.agent",
    "browser_use.agent.service",
    "browser_use.agent.views",
    "browser_use.controller",
    "browser_use.controller.service",
    "browser_use.dom",
    "browser_use.dom.service",
    "playwright",
    "playwright.async_api",
    "browsergym",
    "browsergym.core",
    "browsergym.core.env",
    "crawl4ai",
    "crawl4ai.async_webcrawler",
    "daytona",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()  # type: ignore[assignment]

from app.tool.jobpilot.rag_store import RAGStore, _BUILTIN_SEEDS, reset_rag_store  # noqa: E402


# ---------------------------------------------------------------------------
# Add your custom documents here.
# Each entry is a list of strings.  Leave lists empty to use only the
# built-in samples.
# ---------------------------------------------------------------------------

CUSTOM_DOCS: dict[str, list[str]] = {
    "resume_bullets": [
        # Example (uncomment and edit):
        # "Led migration of legacy Python 2 codebase to Python 3; removed 8K lines of dead code and improved test coverage from 42% to 87%.",
    ],
    "interview_qa": [
        # Example:
        # "Q: Explain CAP theorem and when you would choose consistency over availability.\nIntent: Distributed systems knowledge.\nSTAR: Define CAP; financial/medical systems choose CP; social feeds choose AP; trade-offs.",
    ],
    "skill_graph": [
        # Example:
        # "Security Engineer required skills: network security, penetration testing, OWASP Top-10, Python/Bash scripting, SIEM tools.",
    ],
    "cover_letter_templates": [
        # Example:
        # "Hook (fintech): 'Having spent [X] years building high-reliability payment systems, I am excited by [Company]'s mission to democratize financial access — it aligns directly with the scalability challenges I tackled at [previous company].'",
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_text_file(path: str) -> str:
    """Load a plain-text file and return its content as a single string."""
    return Path(path).read_text(encoding="utf-8", errors="replace").strip()


def _split_into_chunks(text: str, separator: str = "\n\n") -> list[str]:
    """Split *text* by *separator* and return non-empty chunks."""
    return [chunk.strip() for chunk in text.split(separator) if chunk.strip()]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the JobPilot RAG knowledge base."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset in-memory store before seeding (no effect on ChromaDB).",
    )
    parser.add_argument(
        "--dir",
        metavar="PATH",
        default=None,
        help="ChromaDB persistence directory (default: workspace/jobpilot/rag_db).",
    )
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════╗")
    print("║   JobPilot RAG Knowledge Base Builder            ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    # Reset singleton so we can pass our arguments
    reset_rag_store()
    store = RAGStore(persist_dir=args.dir)
    print(f"Backend: {store.backend_name}")
    if args.dir:
        print(f"Directory: {args.dir}")
    print()

    total_added = 0

    for category, builtin_docs in _BUILTIN_SEEDS.items():
        custom = [d for d in CUSTOM_DOCS.get(category, []) if d.strip()]

        if args.reset:
            store.clear(category)

        # Force-seed built-ins (bypasses auto-seed guard)
        store.seed_from_list(builtin_docs + custom, category)
        count = len(builtin_docs) + len(custom)
        total_added += count
        print(f"  ✔ {category:<28} {count:>3} documents")

    print()
    print(f"Total documents loaded: {total_added}")
    print()
    print("Knowledge base is ready.  Agents will use it automatically on next run.")
    print()

    # Smoke-test retrieval
    print("── Smoke test ──────────────────────────────────────")
    for query, cat in [
        ("Python backend API FastAPI", "resume_bullets"),
        ("system design distributed cache behavioral", "interview_qa"),
        ("ML engineer required skills", "skill_graph"),
        ("startup opening hook cover letter", "cover_letter_templates"),
    ]:
        results = store.retrieve(query, cat, top_k=1)
        snippet = results[0][:80].replace("\n", " ") + "…" if results else "(no result)"
        print(f"  [{cat}] '{query[:40]}…'\n    → {snippet}")
    print()
    print("Done ✅")


if __name__ == "__main__":
    main()
