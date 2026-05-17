"""MarkdownExporterTool — saves text content to a Markdown file in the workspace."""
import asyncio
import time
from pathlib import Path
from typing import Optional

from app.config import config
from app.tool.base import BaseTool, ToolResult

_DEFAULT_SUBDIR = "jobpilot"


class MarkdownExporterTool(BaseTool):
    """Save text content as a Markdown (.md) file in the workspace directory."""

    name: str = "md_exporter"
    description: str = (
        "Save content to a Markdown file in the workspace/jobpilot directory. "
        "Use this to persist reports, cover letters, interview kits, or any other "
        "structured text output. Returns the absolute path of the saved file."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The Markdown-formatted text content to save.",
            },
            "filename": {
                "type": "string",
                "description": (
                    "Output filename without extension (e.g., 'cover_letter'). "
                    "Defaults to 'jobpilot_report_<timestamp>' if not provided."
                ),
            },
            "subdir": {
                "type": "string",
                "description": (
                    f"Subdirectory under workspace to save the file in. "
                    f"Defaults to '{_DEFAULT_SUBDIR}'."
                ),
                "default": _DEFAULT_SUBDIR,
            },
        },
        "required": ["content"],
    }

    async def execute(
        self,
        content: str,
        filename: Optional[str] = None,
        subdir: str = _DEFAULT_SUBDIR,
    ) -> ToolResult:
        """Save content to a Markdown file.

        Args:
            content: The text content to write.
            filename: Base filename (without .md extension).
            subdir: Subdirectory under workspace root.

        Returns:
            ToolResult with the saved file path or an error.
        """
        if not content or not content.strip():
            return self.fail_response("Content must not be empty.")

        try:
            file_path = await asyncio.get_event_loop().run_in_executor(
                None, self._write_file, content, filename, subdir
            )
            return self.success_response(
                f"Report saved successfully to: {file_path}"
            )
        except Exception as e:
            return self.fail_response(f"Failed to save Markdown file: {e}")

    @staticmethod
    def _write_file(content: str, filename: Optional[str], subdir: str) -> str:
        """Synchronous helper: create directory and write file."""
        output_dir = config.workspace_root / subdir
        output_dir.mkdir(parents=True, exist_ok=True)

        base_name = filename or f"jobpilot_report_{int(time.time())}"
        # Sanitize filename
        safe_name = "".join(
            c if c.isalnum() or c in ("-", "_") else "_" for c in base_name
        )
        file_path = output_dir / f"{safe_name}.md"

        file_path.write_text(content, encoding="utf-8")
        return str(file_path)
