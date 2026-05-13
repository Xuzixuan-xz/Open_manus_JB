from app.flow.base import BaseFlow


class JobPilotFlow(BaseFlow):
    """Deterministic multi-agent workflow for job/internship applications."""

    async def execute(self, input_text: str) -> str:
        coordinator = self._require_agent("coordinator")
        jd_agent = self._require_agent("jd_analysis")
        company_agent = self._require_agent("company_research")
        resume_agent = self._require_agent("resume_optimization")
        interview_agent = self._require_agent("interview")
        review_agent = self._require_agent("review")
        report_agent = self._require_agent("report")

        coordinator_plan = await coordinator.run(
            f"""
User request:
{input_text}

Create a concise job-application execution brief for downstream specialist agents.
"""
        )

        jd_output = await jd_agent.run(
            f"""
User request:
{input_text}

Coordinator brief:
{coordinator_plan}
"""
        )

        company_output = await company_agent.run(
            f"""
User request:
{input_text}

Coordinator brief:
{coordinator_plan}
"""
        )

        resume_output = await resume_agent.run(
            f"""
User request:
{input_text}

Coordinator brief:
{coordinator_plan}

JD analysis:
{jd_output}
"""
        )

        interview_output = await interview_agent.run(
            f"""
User request:
{input_text}

Coordinator brief:
{coordinator_plan}

JD analysis:
{jd_output}

Resume optimization:
{resume_output}
"""
        )

        review_output = await review_agent.run(
            f"""
Review this draft package for consistency, realism, and quality:

[Coordinator Brief]
{coordinator_plan}

[JD Analysis]
{jd_output}

[Company Research]
{company_output}

[Resume Optimization]
{resume_output}

[Interview Preparation]
{interview_output}
"""
        )

        report = await report_agent.run(
            f"""
Create the final structured JobPilot report from these materials:

[User Request]
{input_text}

[Coordinator Brief]
{coordinator_plan}

[JD Analysis]
{jd_output}

[Company Research]
{company_output}

[Resume Optimization]
{resume_output}

[Interview Preparation]
{interview_output}

[Review Findings]
{review_output}
"""
        )

        return report

    def _require_agent(self, key: str):
        agent = self.get_agent(key)
        if agent is None:
            raise ValueError(f"JobPilotFlow requires agent '{key}'")
        return agent
