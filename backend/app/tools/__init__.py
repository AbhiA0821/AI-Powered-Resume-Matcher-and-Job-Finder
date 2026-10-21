"""
Agent tools package for HireAgent assistant and CrewAI integration.
"""
from app.tools.schemas import (
    GetStudentProfileInput,
    SearchLiveJobsInput,
    GetJobDetailsInput,
    GetJobRequirementsInput,
    CalculateSkillGapInput,
    SearchCareerKnowledgeInput,
    ToolContext
)
from app.tools.registry import (
    ToolDefinition,
    ToolRegistry,
    tool_registry
)

__all__ = [
    "GetStudentProfileInput",
    "SearchLiveJobsInput",
    "GetJobDetailsInput",
    "GetJobRequirementsInput",
    "CalculateSkillGapInput",
    "SearchCareerKnowledgeInput",
    "ToolContext",
    "ToolDefinition",
    "ToolRegistry",
    "tool_registry"
]
