"""WebScraperTool — fetches and cleans text content from a web URL."""
import asyncio
from typing import Optional

import requests
from bs4 import BeautifulSoup

from app.tool.base import BaseTool, ToolResult


_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
_MAX_CONTENT_CHARS = 12000
_REQUEST_TIMEOUT = 15


class WebScraperTool(BaseTool):
    """Fetch the main textual content from a web page URL."""

    name: str = "web_scraper"
    description: str = (
        "Fetch and extract the main text content from a web page. "
        "Useful for retrieving job descriptions from company career pages, "
        "or gathering company information from official websites. "
        "Returns cleaned plain text (scripts and navigation elements are removed)."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The full URL of the web page to scrape (must start with http:// or https://).",
            },
            "max_chars": {
                "type": "integer",
                "description": (
                    f"Maximum number of characters to return (default: {_MAX_CONTENT_CHARS}). "
                    "Increase for longer pages."
                ),
                "default": _MAX_CONTENT_CHARS,
            },
        },
        "required": ["url"],
    }

    async def execute(
        self,
        url: str,
        max_chars: int = _MAX_CONTENT_CHARS,
    ) -> ToolResult:
        """Fetch and clean text content from the given URL.

        Args:
            url: The web page URL to fetch.
            max_chars: Maximum characters of content to return.

        Returns:
            ToolResult containing the extracted text or an error message.
        """
        if not url.startswith(("http://", "https://")):
            return self.fail_response(
                f"Invalid URL '{url}'. URL must start with http:// or https://."
            )

        try:
            content = await asyncio.get_event_loop().run_in_executor(
                None, self._fetch_and_clean, url, max_chars
            )
            if content:
                return self.success_response(content)
            return self.fail_response(
                f"No readable content could be extracted from '{url}'."
            )
        except requests.exceptions.Timeout:
            return self.fail_response(
                f"Request timed out when fetching '{url}' (timeout: {_REQUEST_TIMEOUT}s)."
            )
        except requests.exceptions.ConnectionError as e:
            return self.fail_response(
                f"Connection error when fetching '{url}': {e}"
            )
        except Exception as e:
            return self.fail_response(f"Failed to scrape '{url}': {e}")

    @staticmethod
    def _fetch_and_clean(url: str, max_chars: int) -> Optional[str]:
        """Synchronous helper: fetch URL and strip boilerplate HTML."""
        response = requests.get(url, headers=_HEADERS, timeout=_REQUEST_TIMEOUT)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove non-content elements
        for tag in soup(["script", "style", "header", "footer", "nav", "aside", "form"]):
            tag.decompose()

        # Prefer article / main content blocks when available
        main = soup.find("main") or soup.find("article") or soup.find(id="content")
        target = main if main else soup.body or soup

        text = target.get_text(separator="\n", strip=True)
        # Collapse excessive blank lines
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        cleaned = "\n".join(lines)
        return cleaned[:max_chars] if cleaned else None
