"""run_jobpilot.py — Entry point for the JobPilot job application assistant.

Usage examples:

    # Interactive mode (prompts for JD and resume):
    python run_jobpilot.py

    # Non-interactive with a JSON config file:
    python run_jobpilot.py --input input.json

    # Provide JD text and resume directly via CLI flags:
    python run_jobpilot.py --jd "We are looking for a Python engineer..." \
                           --resume "John Doe, 3 years Python experience..." \
                           --company "Acme Corp" \
                           --company-url "https://acme.com"

    # Provide a JD URL and a resume file:
    python run_jobpilot.py --jd-url "https://company.com/jobs/123" \
                           --resume-path "/path/to/resume.pdf"

The generated report is saved to workspace/jobpilot/jobpilot_report_<timestamp>.md
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path when running directly
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.flow.jobpilot_flow import JobPilotFlow
from app.logger import logger
from app.tool.jobpilot.md_exporter import MarkdownExporterTool


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="JobPilot — AI-powered job application assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i",
        metavar="FILE",
        help="Path to a JSON file with keys: jd_text, jd_url, resume_text, resume_path, "
             "company_name, company_url",
    )
    parser.add_argument("--jd", metavar="TEXT", help="Job description as raw text")
    parser.add_argument(
        "--jd-url", metavar="URL", help="URL of the job posting (fetched automatically)"
    )
    parser.add_argument("--resume", metavar="TEXT", help="Resume content as raw text")
    parser.add_argument(
        "--resume-path",
        metavar="PATH",
        help="Path to resume file (.pdf or .docx)",
    )
    parser.add_argument("--company", metavar="NAME", help="Company name")
    parser.add_argument("--company-url", metavar="URL", help="Company website URL")
    parser.add_argument(
        "--output", "-o",
        metavar="FILE",
        help="Output Markdown file path (default: workspace/jobpilot/jobpilot_report_<ts>.md)",
    )
    return parser


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------

def _build_context(args: argparse.Namespace) -> dict:
    """Assemble the context dict from CLI args or interactive prompts."""
    ctx: dict = {}

    # Load from JSON file if provided
    if args.input:
        with open(args.input, encoding="utf-8") as f:
            ctx = json.load(f)
        logger.info(f"Loaded input context from: {args.input}")
        return ctx

    # CLI flags
    if args.jd:
        ctx["jd_text"] = args.jd
    if args.jd_url:
        ctx["jd_url"] = args.jd_url
    if args.resume:
        ctx["resume_text"] = args.resume
    if args.resume_path:
        ctx["resume_path"] = args.resume_path
    if args.company:
        ctx["company_name"] = args.company
    if args.company_url:
        ctx["company_url"] = args.company_url

    # Interactive mode if nothing was provided
    if not ctx:
        ctx = _interactive_prompts()

    return ctx


def _interactive_prompts() -> dict:
    """Collect inputs interactively when no CLI flags are given."""
    print("\n" + "=" * 60)
    print("  JobPilot — Interactive Mode")
    print("=" * 60)
    print("Press Enter to skip optional fields.\n")

    ctx: dict = {}

    # JD
    print("📋 Job Description")
    jd_url = input("   JD URL (optional): ").strip()
    if jd_url:
        ctx["jd_url"] = jd_url
    print("   Paste JD text below. Enter a blank line when done:")
    jd_lines = _read_multiline("   ")
    if jd_lines:
        ctx["jd_text"] = jd_lines

    if not ctx.get("jd_url") and not ctx.get("jd_text"):
        print("❌ Either a JD URL or JD text is required. Exiting.")
        sys.exit(1)

    # Resume
    print("\n📄 Resume")
    resume_path = input("   Resume file path (.pdf/.docx, optional): ").strip()
    if resume_path:
        ctx["resume_path"] = resume_path
    if not resume_path:
        print("   Paste resume text below. Enter a blank line when done:")
        resume_lines = _read_multiline("   ")
        if resume_lines:
            ctx["resume_text"] = resume_lines

    # Company
    print("\n🏢 Company (optional)")
    company = input("   Company name: ").strip()
    if company:
        ctx["company_name"] = company
    company_url = input("   Company URL: ").strip()
    if company_url:
        ctx["company_url"] = company_url

    return ctx


def _read_multiline(prefix: str = "") -> str:
    """Read multiple lines until an empty line is entered."""
    lines = []
    while True:
        try:
            line = input(prefix)
        except EOFError:
            break
        if line == "":
            break
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_jobpilot(ctx: dict, output_path: str | None = None) -> str:
    """Execute the JobPilot pipeline and save the report.

    Args:
        ctx: Context dict with jd_text/jd_url, resume_text/resume_path, etc.
        output_path: Optional explicit output file path.

    Returns:
        The final Markdown report string.
    """
    flow = JobPilotFlow.create()

    logger.info("🚀 Starting JobPilot pipeline...")
    start = time.time()

    report = await asyncio.wait_for(
        flow.execute(json.dumps(ctx)),
        timeout=3600,  # 60-minute safety cap
    )

    elapsed = time.time() - start
    logger.info(f"⏱️  Pipeline finished in {elapsed:.1f}s")

    # Save the report
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(report, encoding="utf-8")
        logger.info(f"📝 Report saved to: {output_path}")
    else:
        exporter = MarkdownExporterTool()
        save_result = await exporter.execute(
            content=report,
            filename=f"jobpilot_report_{int(time.time())}",
        )
        logger.info(f"📝 {save_result}")

    return report


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        ctx = _build_context(args)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ Failed to load input: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        report = asyncio.run(run_jobpilot(ctx, output_path=args.output if args else None))
        print("\n" + "=" * 60)
        print("✅ JobPilot pipeline complete!")
        print("=" * 60)
        print("\nREPORT PREVIEW (first 2000 chars):\n")
        print(report[:2000])
        if len(report) > 2000:
            print("\n... [truncated — see saved report file for full output]")
    except asyncio.TimeoutError:
        print("❌ Pipeline timed out after 60 minutes.", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Cancelled by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
