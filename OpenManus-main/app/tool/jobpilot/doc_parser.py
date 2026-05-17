"""DocParserTool — parses PDF, Docx, or plain-text resume/document files."""
import asyncio
from pathlib import Path
from typing import Optional

from app.tool.base import BaseTool, ToolResult


class DocParserTool(BaseTool):
    """Parse a document file (PDF, DOCX) or accept raw text and return plain text content."""

    name: str = "doc_parser"
    description: str = (
        "Parse a document file (PDF or DOCX) and return its plain text content. "
        "You can provide a file_path to read from disk, or supply the text directly via the 'text' parameter. "
        "Supported formats: .pdf, .docx, .txt, and any plain text."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute or relative path to the document file (PDF, DOCX, or TXT).",
            },
            "text": {
                "type": "string",
                "description": "Raw text to pass through directly (use when the content is already available as text).",
            },
        },
        "required": [],
    }

    async def execute(
        self,
        file_path: Optional[str] = None,
        text: Optional[str] = None,
    ) -> ToolResult:
        """Parse a document or return raw text.

        Args:
            file_path: Path to the document file.
            text: Raw text content (takes precedence over file if both provided).

        Returns:
            ToolResult with the extracted plain text.
        """
        if text:
            return self.success_response(text.strip())

        if not file_path:
            return self.fail_response(
                "Either 'file_path' or 'text' must be provided."
            )

        path = Path(file_path)
        if not path.exists():
            return self.fail_response(f"File not found: {file_path}")

        suffix = path.suffix.lower()
        try:
            if suffix == ".pdf":
                content = await asyncio.get_event_loop().run_in_executor(
                    None, self._parse_pdf, path
                )
            elif suffix == ".docx":
                content = await asyncio.get_event_loop().run_in_executor(
                    None, self._parse_docx, path
                )
            else:
                # Treat as plain text
                content = path.read_text(encoding="utf-8", errors="replace")

            return self.success_response(content.strip())
        except ImportError as e:
            return self.fail_response(
                f"Required library not installed for {suffix} parsing: {e}. "
                "Install 'pypdf' for PDF or 'python-docx' for DOCX, "
                "or paste the text directly using the 'text' parameter."
            )
        except Exception as e:
            return self.fail_response(f"Failed to parse document '{file_path}': {e}")

    @staticmethod
    def _parse_pdf(path: Path) -> str:
        """Extract text from a PDF file using pypdf."""
        import pypdf  # type: ignore[import]

        reader = pypdf.PdfReader(str(path))
        pages = []
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                pages.append(extracted)
        return "\n".join(pages)

    @staticmethod
    def _parse_docx(path: Path) -> str:
        """Extract text from a DOCX file using python-docx."""
        import docx  # type: ignore[import]

        doc = docx.Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)
