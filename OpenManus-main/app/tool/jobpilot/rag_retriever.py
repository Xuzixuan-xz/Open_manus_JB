"""RAGRetrieverTool — BaseTool wrapper for JobPilot RAG knowledge retrieval.

Agents call this tool to fetch relevant examples from the knowledge base
*before* generating their main output, grounding their responses in curated,
role-specific material instead of relying solely on LLM priors.

Categories
----------
* ``resume_bullets``         — job-impact bullet point examples for ResumeOptimizerAgent
* ``interview_qa``           — interview Q&A templates for InterviewAgent
* ``skill_graph``            — role-standard skill sets for JDParserAgent
* ``cover_letter_templates`` — opening / value / closing snippets for CoverLetterAgent
"""

from __future__ import annotations

from typing import Optional

from app.tool.base import BaseTool, ToolResult
from app.tool.jobpilot.rag_store import get_rag_store

# Valid category constants (surfaced so prompts can reference them)
CATEGORY_RESUME_BULLETS = "resume_bullets"
CATEGORY_INTERVIEW_QA = "interview_qa"
CATEGORY_SKILL_GRAPH = "skill_graph"
CATEGORY_COVER_LETTER = "cover_letter_templates"

_VALID_CATEGORIES = {
    CATEGORY_RESUME_BULLETS,
    CATEGORY_INTERVIEW_QA,
    CATEGORY_SKILL_GRAPH,
    CATEGORY_COVER_LETTER,
}


class RAGRetrieverTool(BaseTool):
    """Retrieve relevant examples from the JobPilot knowledge base.

    Returns the top-k most relevant documents for a given query and category.
    Use this tool *before* generating your main analysis to ground your output
    in real-world examples and templates.
    """

    name: str = "rag_retriever"
    description: str = (
        "Retrieve relevant examples, templates, or skill information from the JobPilot "
        "knowledge base. Use this BEFORE generating your main output to get grounding "
        "examples that improve quality. "
        "Available categories: "
        "'resume_bullets' (job-impact resume examples for resume optimization), "
        "'interview_qa' (interview question + answer frameworks for interview prep), "
        "'skill_graph' (standard skill sets per role type for JD parsing), "
        "'cover_letter_templates' (opening hooks, value props, closings for cover letters)."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Natural-language search query. Be specific: include role type, "
                    "technologies, or topic (e.g. 'Python backend API performance', "
                    "'ML model training behavioral question', 'data engineer Spark skills')."
                ),
            },
            "category": {
                "type": "string",
                "enum": sorted(_VALID_CATEGORIES),
                "description": (
                    "Knowledge-base category to search. Must be one of: "
                    + ", ".join(f"'{c}'" for c in sorted(_VALID_CATEGORIES))
                    + "."
                ),
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum number of results to return (default: 5, max: 10).",
                "default": 5,
            },
        },
        "required": ["query", "category"],
    }

    async def execute(
        self,
        query: str,
        category: str,
        top_k: int = 5,
    ) -> ToolResult:
        """Retrieve top-k relevant documents from the knowledge base.

        Args:
            query: Natural-language search query.
            category: One of the four valid category strings.
            top_k: Number of results to return (clamped to 1–10).

        Returns:
            ToolResult containing the retrieved snippets as formatted text,
            or an error message if the category is invalid.
        """
        if not query or not query.strip():
            return self.fail_response("'query' must be a non-empty string.")

        if category not in _VALID_CATEGORIES:
            return self.fail_response(
                f"Invalid category '{category}'. "
                f"Valid values: {', '.join(sorted(_VALID_CATEGORIES))}."
            )

        top_k = max(1, min(int(top_k), 10))

        try:
            store = get_rag_store()
            results = store.retrieve(query, category, top_k)
        except Exception as exc:
            return self.fail_response(
                f"RAG retrieval failed: {exc}. "
                "Proceed without retrieval context."
            )

        if not results:
            return self.success_response(
                f"No documents found in category '{category}' for query: {query!r}. "
                "Proceed without retrieval context."
            )

        lines = [
            f"Retrieved {len(results)} relevant example(s) from '{category}':",
            "",
        ]
        for i, doc in enumerate(results, start=1):
            lines.append(f"[{i}] {doc}")
            lines.append("")

        return self.success_response("\n".join(lines).strip())
