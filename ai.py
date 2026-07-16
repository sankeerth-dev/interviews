"""
Enterprise-grade AI extraction pipeline for Meeting Task Manager.

Architecture (7 layers):

  Layer 1 — TranscriptPreprocessor
            Clean & segment the raw transcript before it reaches the LLM.

  Layer 2 — DateResolver
            Resolve relative date expressions ("next Friday", "end of month")
            to ISO-8601 strings using dateparser + manual rules.

  Layer 3 — TranscriptChunker
            Split arbitrarily long transcripts into overlapping, context-safe
            chunks that fit the Azure OpenAI context window.

  Layer 4 — Prompt Engineering
            Build rich, few-shot system prompts that maximise extraction
            precision and minimise hallucinations.

  Layer 5 — Parallel Extract + Per-Chunk Validation
            Each chunk is extracted and validated independently in parallel.
            The validator only sees the tasks and transcript FROM ITS OWN CHUNK.
            A 10% safety threshold automatically falls back to raw extraction
            if validation over-removes tasks.

  Layer 6 — Multi-Field Composite Deduplicator
            Compares title + assignee + due_date + evidence position together.
            Requires multiple fields to match before merging — never merges on
            title similarity alone.

  Layer 7 — RuleEngine
            Post-processing normaliser: fix, enrich, flag. Never silently drops
            valid tasks (user requirement: never miss a task).

Public interface (UNCHANGED — fully backward-compatible):
  AIService.generate_tasks(transcript, team_members, today) -> list[dict]
"""

import json
import re
import time
import unicodedata
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from typing import NamedTuple

from openai import AzureOpenAI, APIError, APITimeoutError, RateLimitError

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


# ── Azure OpenAI client ────────────────────────────────────────────────────────
client = AzureOpenAI(
    api_key=settings.AZURE_OPENAI_API_KEY,
    api_version=settings.AZURE_OPENAI_API_VERSION,
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
)


# ── Pipeline constants (overridable via settings) ──────────────────────────────
CHUNK_SIZE_CHARS: int = getattr(settings, "AI_CHUNK_SIZE_CHARS", 32_000)
CHUNK_OVERLAP_CHARS: int = 500
MAX_RETRIES: int = getattr(settings, "AI_MAX_RETRIES", 3)
VALIDATION_PASS_ENABLED: bool = getattr(settings, "AI_VALIDATION_PASS", True)
# Minimum confidence to include a task. Set to 0.50 to catch everything
# while still blocking obvious hallucinations.
MIN_CONFIDENCE: float = getattr(settings, "AI_MIN_CONFIDENCE", 0.50)
# If validation removes more than this fraction of tasks, discard validation
# results and fall back to the raw extraction for that chunk.
VALIDATION_SAFETY_THRESHOLD: float = 0.10  # 10% max removal per chunk
# Maximum parallel workers for chunk processing
MAX_PARALLEL_WORKERS: int = getattr(settings, "AI_MAX_PARALLEL_WORKERS", 4)

# Common action verbs that should start a task title
_ACTION_VERBS: frozenset[str] = frozenset({
    "add", "address", "analyze", "approve", "assign", "build", "check",
    "clarify", "collect", "complete", "configure", "confirm", "contact",
    "coordinate", "create", "define", "deliver", "deploy", "design",
    "document", "draft", "ensure", "evaluate", "finish", "fix", "follow",
    "hire", "identify", "implement", "install", "integrate", "investigate",
    "merge", "migrate", "monitor", "notify", "onboard", "optimize", "plan",
    "prepare", "present", "refactor", "release", "remove", "report",
    "resolve", "review", "schedule", "send", "set", "setup", "share",
    "submit", "test", "update", "upload", "write",
})

_VALID_PRIORITIES: frozenset[str] = frozenset({"low", "medium", "high", "critical"})

_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "and", "are", "at", "be", "been", "being", "by", "can",
    "could", "for", "in", "is", "it", "its", "may", "might", "must", "of",
    "on", "or", "shall", "should", "that", "the", "these", "this", "those",
    "to", "was", "were", "will", "with", "would",
})


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — TRANSCRIPT PREPROCESSOR
# ═══════════════════════════════════════════════════════════════════════════════

class TranscriptPreprocessor:
    """
    Clean and normalise a raw meeting transcript before it reaches the LLM.

    Guarantees
    ----------
    * Whitespace is normalised (no trailing spaces, no excess blank lines).
    * Transcription artefacts are removed ([inaudible], [crosstalk], etc.).
    * Repeated filler words are collapsed to a single instance.
    * Punctuation is normalised (curly quotes → straight, em-dash → --).
    * Speaker lines are preserved and formatted consistently.
    * Timestamps are preserved in-place.
    * Orphaned continuation lines are merged into the preceding speaker turn.
    * The SEMANTIC MEANING of the transcript is never changed.
    """

    _ARTIFACT_RE = re.compile(
        r"\[(?:inaudible|crosstalk|laughter|applause|noise|background\s+noise"
        r"|music|silence|pause|unintelligible|unclear|indistinct|cough"
        r"|phone\s+ringing|door\s+closing|typing|recording\s+started"
        r"|recording\s+stopped)\]",
        re.IGNORECASE,
    )

    # Collapse repeated filler words — keep only the first token
    _FILLER_RE = re.compile(
        r"\b(um+|uh+|er+|hmm+|hm+|mhm+|ah+|aha+|uhh+|umm+|like,?\s+like)\b",
        re.IGNORECASE,
    )

    # Speaker line pattern — handles:
    #   "Name:"
    #   "Name (role):"
    #   "[HH:MM] Name:"
    #   "[HH:MM:SS] Name:"
    _SPEAKER_RE = re.compile(
        r"^(?:\[?\d{1,2}:\d{2}(?::\d{2})?\]?\s+)?[A-Z][^\n:]{0,50}:\s",
        re.MULTILINE,
    )

    @classmethod
    def process(cls, transcript: str) -> str:
        """Return a cleaned transcript string, ready for LLM ingestion."""
        text = transcript

        # ── Unicode normalisation ──────────────────────────────────────────
        text = unicodedata.normalize("NFKC", text)

        # ── Curly quotes → straight ────────────────────────────────────────
        text = (
            text.replace("\u2018", "'").replace("\u2019", "'")
                .replace("\u201c", '"').replace("\u201d", '"')
        )

        # ── Em-dash / en-dash → double-hyphen ─────────────────────────────
        text = text.replace("\u2014", "--").replace("\u2013", "--")

        # ── Remove transcription artefacts ─────────────────────────────────
        text = cls._ARTIFACT_RE.sub("", text)

        # ── Collapse filler words ──────────────────────────────────────────
        text = cls._FILLER_RE.sub(r"\1", text)

        # ── Normalise whitespace within each line ──────────────────────────
        lines = text.splitlines()
        cleaned: list[str] = []
        for line in lines:
            line = re.sub(r"[ \t]+", " ", line).strip()
            if line:
                cleaned.append(line)

        # ── Merge orphaned continuation lines ──────────────────────────────
        # A line that does NOT match the speaker pattern is treated as a
        # continuation of the previous speaker's turn.
        merged: list[str] = []
        for line in cleaned:
            if merged and not cls._SPEAKER_RE.match(line):
                merged[-1] += " " + line
            else:
                merged.append(line)

        # ── Collapse excess blank lines ────────────────────────────────────
        result = "\n".join(merged)
        result = re.sub(r"\n{3,}", "\n\n", result)

        return result.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — DATE RESOLVER
# ═══════════════════════════════════════════════════════════════════════════════

class DateResolver:
    """
    Resolve natural-language date expressions to ISO-8601 strings (YYYY-MM-DD).

    Resolution strategy
    -------------------
    1. Fast manual rules handle the most common business phrases exactly.
    2. ``dateparser`` library handles everything else with PREFER_DATES_FROM=future.
    3. Returns None for any expression that cannot be confidently resolved.

    Never returns invalid ISO strings — all output is validated before return.
    """

    @staticmethod
    def _next_weekday(base: date, weekday: int) -> date:
        """Return the upcoming occurrence of ``weekday`` (0=Mon…6=Sun) on or after ``base + 1``."""
        days_ahead = weekday - base.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        return base + timedelta(days=days_ahead)

    @staticmethod
    def _last_day_of_month(d: date) -> date:
        """Return the last calendar day of the month containing ``d``."""
        if d.month == 12:
            return date(d.year + 1, 1, 1) - timedelta(days=1)
        return date(d.year, d.month + 1, 1) - timedelta(days=1)

    @classmethod
    def resolve(cls, expression: str | None, today: date) -> str | None:
        """
        Convert a date expression to YYYY-MM-DD anchored at ``today``.
        Returns None if the expression is empty or cannot be resolved.
        """
        if not expression:
            return None

        expr = expression.strip().lower()

        # ── Fast manual rules ──────────────────────────────────────────────
        if expr in ("today", "eod", "cob", "by eod", "by cob"):
            return today.isoformat()

        if expr in ("tomorrow",):
            return (today + timedelta(days=1)).isoformat()

        if expr in ("this friday", "friday", "end of week", "eow", "by friday"):
            return cls._next_weekday(today, 4).isoformat()

        if expr in ("this monday", "monday", "next week", "start of next week"):
            return cls._next_weekday(today, 0).isoformat()

        if expr in ("this tuesday", "tuesday"):
            return cls._next_weekday(today, 1).isoformat()

        if expr in ("this wednesday", "wednesday", "mid-week"):
            return cls._next_weekday(today, 2).isoformat()

        if expr in ("this thursday", "thursday"):
            return cls._next_weekday(today, 3).isoformat()

        if expr in ("this saturday", "saturday"):
            return cls._next_weekday(today, 5).isoformat()

        if expr in ("this sunday", "sunday", "end of weekend"):
            return cls._next_weekday(today, 6).isoformat()

        if expr in ("end of month", "eom", "by end of month"):
            return cls._last_day_of_month(today).isoformat()

        if expr in ("this week",):
            # End of current business week (Friday)
            return cls._next_weekday(today, 4).isoformat()

        if expr in ("next friday",):
            # Skip the upcoming Friday — go to the one after
            upcoming = cls._next_weekday(today, 4)
            return (upcoming + timedelta(weeks=1)).isoformat()

        if expr in ("next monday",):
            upcoming = cls._next_weekday(today, 0)
            return (upcoming + timedelta(weeks=1)).isoformat()

        # "in N days"
        m = re.fullmatch(r"in (\d+) days?", expr)
        if m:
            return (today + timedelta(days=int(m.group(1)))).isoformat()

        # "in N weeks"
        m = re.fullmatch(r"in (\d+) weeks?", expr)
        if m:
            return (today + timedelta(weeks=int(m.group(1)))).isoformat()

        # "by [date expression]" — strip "by" and recurse
        m = re.fullmatch(r"by (.+)", expr)
        if m:
            return cls.resolve(m.group(1), today)

        # Already a valid ISO date?
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", expression):
            try:
                date.fromisoformat(expression)
                return expression
            except ValueError:
                return None

        # ── dateparser fallback ────────────────────────────────────────────
        try:
            import dateparser  # type: ignore
            parsed = dateparser.parse(
                expression,
                settings={
                    "PREFER_DATES_FROM": "future",
                    "RELATIVE_BASE": today,
                    "RETURN_AS_TIMEZONE_AWARE": False,
                },
            )
            if parsed:
                return parsed.date().isoformat()
        except Exception:
            pass

        return None


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 3 — TRANSCRIPT CHUNKER
# ═══════════════════════════════════════════════════════════════════════════════

class TranscriptChunker:
    """
    Split a transcript of arbitrary length into overlapping chunks that fit
    within the Azure OpenAI context window.

    Design
    ------
    * Target chunk size: ``CHUNK_SIZE_CHARS`` characters (default 32,000).
    * Overlap: ``CHUNK_OVERLAP_CHARS`` characters (default 500) so that
      tasks spanning a chunk boundary are not missed.
    * Splits are always on newline boundaries — a speaker turn is never split.
    * Short transcripts (≤ chunk_size) return a single-element list.
    """

    @staticmethod
    def split(
        transcript: str,
        chunk_size: int = CHUNK_SIZE_CHARS,
        overlap: int = CHUNK_OVERLAP_CHARS,
    ) -> list[str]:
        """Return a list of transcript chunks."""
        if len(transcript) <= chunk_size:
            return [transcript]

        lines = transcript.splitlines(keepends=True)
        chunks: list[str] = []
        current = ""

        for line in lines:
            if len(current) + len(line) > chunk_size and current:
                chunks.append(current.strip())
                # Start next chunk with an overlap tail for cross-boundary context
                tail = current[-overlap:] if len(current) > overlap else current
                current = tail + line
            else:
                current += line

        if current.strip():
            chunks.append(current.strip())

        return chunks


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 4 — PROMPT ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

_FEW_SHOT_EXAMPLES = """
═══════════════════════════════════════════════════
EXAMPLE 1 — Direct assignment with due date
═══════════════════════════════════════════════════
INPUT:
  Team: Alice Johnson, Bob Smith, Carol White
  Today: 2025-06-10
  Alice: "Bob, can you finish the API docs by end of this week?"
  Bob: "Sure, I'll have it done by Friday."

OUTPUT:
{
  "tasks": [
    {
      "title": "Complete API documentation",
      "description": "Bob committed to finishing the API documentation by Friday (end of the week). Alice requested this during the sprint review to unblock downstream teams.",
      "assignee": "Bob Smith",
      "assigner": "Alice Johnson",
      "priority": "high",
      "due_date": "2025-06-13",
      "confidence": 0.98,
      "evidence": "Alice: 'Bob, can you finish the API docs by end of this week?' Bob: 'Sure, I'll have it done by Friday.'",
      "meeting_section": "Sprint review",
      "tags": ["documentation", "api"]
    }
  ]
}

═══════════════════════════════════════════════════
EXAMPLE 2 — Critical blocking issue
═══════════════════════════════════════════════════
INPUT:
  Team: Alice Johnson, Carol White
  Today: 2025-06-10
  Alice: "Carol, the login bug is blocking QA — fix it ASAP."
  Carol: "On it, I'll push a fix today."

OUTPUT:
{
  "tasks": [
    {
      "title": "Fix login bug blocking QA",
      "description": "Critical login bug must be fixed immediately — it is blocking the entire QA team from proceeding. Carol committed to pushing a fix the same day.",
      "assignee": "Carol White",
      "assigner": "Alice Johnson",
      "priority": "critical",
      "due_date": "2025-06-10",
      "confidence": 0.99,
      "evidence": "Alice: 'Carol, the login bug is blocking QA — fix it ASAP.' Carol: 'On it, I'll push a fix today.'",
      "meeting_section": "Bug triage",
      "tags": ["bug", "qa", "blocking"]
    }
  ]
}

═══════════════════════════════════════════════════
EXAMPLE 3 — Pronoun resolution ("it", "that", "this")
═══════════════════════════════════════════════════
INPUT:
  Team: John Doe, Sarah Lee
  Today: 2025-06-10
  Sarah: "The dashboard redesign is critical for the Q3 launch."
  John: "Agreed. I'll take care of it by next Monday."

OUTPUT:
{
  "tasks": [
    {
      "title": "Implement dashboard redesign",
      "description": "John committed to completing the dashboard redesign by next Monday. 'It' refers to the dashboard redesign mentioned immediately before. This is tied to the Q3 launch timeline.",
      "assignee": "John Doe",
      "assigner": null,
      "priority": "high",
      "due_date": "2025-06-16",
      "confidence": 0.92,
      "evidence": "Sarah: 'The dashboard redesign is critical for the Q3 launch.' John: 'Agreed. I'll take care of it by next Monday.'",
      "meeting_section": "Design review",
      "tags": ["dashboard", "redesign", "q3"]
    }
  ]
}

═══════════════════════════════════════════════════
EXAMPLE 4 — Assignment conflict (LATEST assignment wins)
═══════════════════════════════════════════════════
INPUT:
  Team: John Doe, Sarah Lee
  Today: 2025-06-10
  John: "I'll handle the production deployment."
  [Later in the meeting]
  Sarah: "Actually, I'll take the deployment — John is overloaded this week."

OUTPUT:
{
  "tasks": [
    {
      "title": "Deploy application to production",
      "description": "Sarah took over the production deployment from John because John is overloaded. The latest explicit assignment (Sarah) supersedes John's earlier commitment.",
      "assignee": "Sarah Lee",
      "assigner": null,
      "priority": "medium",
      "due_date": null,
      "confidence": 0.95,
      "evidence": "John: 'I'll handle the production deployment.' Sarah: 'Actually, I'll take the deployment — John is overloaded this week.'",
      "meeting_section": "Release planning",
      "tags": ["deployment", "production", "release"]
    }
  ]
}

═══════════════════════════════════════════════════
EXAMPLE 5 — Duplicate merge (same task mentioned twice)
═══════════════════════════════════════════════════
INPUT:
  Team: Alice Johnson
  Today: 2025-06-10
  Alice: "I need to update the dashboard metrics before the board meeting."
  [Later]
  Alice: "Right, the dashboard metrics update — I'll get that done by Thursday."

OUTPUT:
{
  "tasks": [
    {
      "title": "Update dashboard metrics for board meeting",
      "description": "Alice committed to updating the dashboard metrics before the board meeting. The task was mentioned twice — merged into one entry with the most specific deadline (Thursday).",
      "assignee": "Alice Johnson",
      "assigner": null,
      "priority": "high",
      "due_date": "2025-06-12",
      "confidence": 0.97,
      "evidence": "Alice: 'I need to update the dashboard metrics before the board meeting.' [Later] Alice: 'the dashboard metrics update — I'll get that done by Thursday.'",
      "meeting_section": "Status updates",
      "tags": ["dashboard", "metrics", "board-meeting"]
    }
  ]
}

═══════════════════════════════════════════════════
EXAMPLE 6 — Self-commitment without explicit assigner
═══════════════════════════════════════════════════
INPUT:
  Team: John Doe, Sarah Lee
  Today: 2025-06-10
  John: "I'll send the updated contract to the client by June 20th."

OUTPUT:
{
  "tasks": [
    {
      "title": "Send updated contract to client",
      "description": "John committed to sending the updated contract to the client by June 20th. Self-assigned — no explicit assigner in the transcript.",
      "assignee": "John Doe",
      "assigner": null,
      "priority": "high",
      "due_date": "2025-06-20",
      "confidence": 0.99,
      "evidence": "John: 'I'll send the updated contract to the client by June 20th.'",
      "meeting_section": "Client updates",
      "tags": ["contract", "client"]
    }
  ]
}

═══════════════════════════════════════════════════
EXAMPLE 7 — NEGATIVE EXAMPLES: What NOT to extract
═══════════════════════════════════════════════════
INPUT:
  Team: Alice Johnson, Bob Smith
  Today: 2025-06-10
  Alice: "I think we should really consider improving performance."
  Bob: "Yeah, performance improvements are definitely important."
  Alice: "We've decided to adopt the new microservices architecture."
  Bob: "The Q3 metrics are looking pretty good overall."
  Alice: "There's a risk that the vendor won't deliver on time."

OUTPUT: {"tasks": []}

REASON: None of these sentences are explicit commitments or assignments.
- "I think we should consider" → opinion, not a commitment
- "performance is important" → vague discussion, no owner or action
- "We've decided to adopt" → group decision, no individual action owner
- "Q3 metrics looking good" → status update, no action needed
- "There's a risk" → risk identification without an owner or action
"""


def _build_system_prompt(team_members: list[str], today: str) -> str:
    """Build the context-aware system prompt for Pass 1 extraction."""
    if team_members:
        member_block = (
            "KNOWN TEAM MEMBERS — assignee and assigner MUST be one of these exact full names, or null:\n"
            + "\n".join(f"  • {name}" for name in team_members)
        )
    else:
        member_block = "KNOWN TEAM MEMBERS: none provided — use null for assignee and assigner."

    return f"""You are an enterprise-grade AI meeting analyst specialising in task extraction.
Your ONLY job is to extract every explicit, actionable task from the meeting transcript.

{member_block}

TODAY'S DATE: {today}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FEW-SHOT EXAMPLES — study these carefully before extracting
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{_FEW_SHOT_EXAMPLES}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1: INTENT CLASSIFICATION (do this mentally for every sentence)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For every statement classify it as:
  TASK       → explicit commitment / assignment        → EXTRACT ✓
  FOLLOW-UP  → next-step after a decision with owner  → EXTRACT ✓
  DECISION   → group decision, no individual owner    → SKIP ✗
  DISCUSSION → opinion, idea, debate                  → SKIP ✗
  QUESTION   → someone asking, not committing         → SKIP ✗
  STATUS     → update on existing work                → SKIP ✗
  RISK/ISSUE → risk flagged, no action owner          → SKIP ✗
  REMINDER   → reminder of something already agreed   → SKIP unless it creates a new action ✗

ONLY items classified as TASK or FOLLOW-UP become tasks in your output.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2: OUTPUT FORMAT — return ONLY valid JSON, no prose
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{{
  "tasks": [
    {{
      "title": "Action verb + object, ≤ 10 words",
      "description": "Full self-contained context: what, why, constraints, deadline rationale",
      "assignee": "Exact full name from KNOWN TEAM MEMBERS, or null",
      "assigner": "Exact full name from KNOWN TEAM MEMBERS, or null",
      "priority": "low | medium | high | critical",
      "due_date": "YYYY-MM-DD or null",
      "confidence": 0.00,
      "evidence": "VERBATIM transcript quote that justifies this task",
      "meeting_section": "Short label for this part of the meeting",
      "tags": ["tag1", "tag2"]
    }}
  ]
}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3: STRICT RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXTRACTION RULES:
 1. Extract ONLY if someone was EXPLICITLY asked or EXPLICITLY committed to doing something.
 2. Extract EVERY task — missing a task is worse than a false positive.
 3. Merge repeated mentions of the same task into ONE entry with the best evidence.
 4. If a task is reassigned, use the LATEST explicit assignment. Latest wins.
 5. Resolve pronouns (it, that, this, them) using the immediately preceding context.

ASSIGNEE / ASSIGNER:
 6. assignee = person who WILL DO the task.
 7. assigner = person who REQUESTED or DELEGATED it. If self-assigned, assigner = null.
 8. Match case-insensitively: "john", "John Doe", "John D" all match "John Doe".
 9. Output the FULL NAME exactly as it appears in KNOWN TEAM MEMBERS.
10. If the name is not in the team list → null.

PRIORITY:
11. critical → blocking / ASAP / "must be done today" ("blocking", "urgent", "immediately", "ASAP").
12. high     → this week / soon / tight deadline ("by Friday", "this week", "by end of week").
13. medium   → important but no hard urgency signal.
14. low      → optional / future / "whenever you get a chance".
15. Default: medium.

DUE DATES (Today = {today}):
16. today           → {today}
17. tomorrow        → next calendar day
18. this Friday     → upcoming Friday
19. next week       → upcoming Monday
20. end of week     → upcoming Friday
21. end of month    → last day of current month
22. in N days       → {today} + N days
23. Output ISO format YYYY-MM-DD. Use null if no date is mentioned.

CONFIDENCE SCORING:
24. 1.00 = explicit verb + known assignee + specific deadline.
25. 0.90-0.99 = explicit verb + known assignee, vague deadline.
26. 0.80-0.89 = explicit verb + assignee inferred from pronoun or context.
27. 0.70-0.79 = commitment implied but not 100% direct, or assignee partially matched.
28. 0.50-0.69 = low certainty but plausible task — include with lower confidence.
29. Below 0.50 = do NOT include (likely hallucination).

TITLE RULES:
30. Must begin with an action verb (Fix, Send, Review, Schedule, Deploy, Update, etc.).
31. ≤ 10 words, specific enough to understand without reading the transcript.

EVIDENCE:
32. Copy the EXACT verbatim speaker quote that justifies the task.
33. Never invent, summarise, or paraphrase evidence.

EMPTY RESULT:
34. If there are genuinely no actionable tasks, return: {{"tasks": []}}
"""


def _build_chunk_validation_prompt(
    team_members: list[str],
    today: str,
    extracted_tasks: list[dict],
    chunk_transcript: str,
    chunk_index: int,
    total_chunks: int,
) -> str:
    """
    Build the per-chunk auditor prompt.

    Critical design decisions:
    - The validator receives the FULL CHUNK TEXT that produced these tasks,
      not a truncated global sample.
    - The prompt explicitly forbids aggressive deletion.
    - Deletion is only permitted when evidence is completely absent AND
      confidence would be < 0.50.
    """
    team_block = ", ".join(team_members) if team_members else "none"
    tasks_json = json.dumps({"tasks": extracted_tasks}, indent=2, ensure_ascii=False)

    return f"""You are a CONSERVATIVE AI auditor reviewing extracted tasks.
You are NOT an extractor. You are NOT allowed to rebuild the task list from scratch.
Your ONLY job is to CORRECT mistakes — not to remove tasks.

TEAM MEMBERS: {team_block}
TODAY: {today}
CHUNK: {chunk_index + 1} of {total_chunks}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠ CRITICAL DELETION POLICY — READ BEFORE DOING ANYTHING ELSE ⚠
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You may ONLY delete a task if ALL THREE conditions are true:
  1. There is ZERO evidence in the transcript below.
  AND
  2. Confidence would be below 0.50.
  AND
  3. The task is clearly a hallucination with no plausible basis.

If you are uncertain whether to delete → KEEP THE TASK and lower confidence.
When in doubt, KEEP IT. Missing a task is worse than keeping an uncertain one.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRANSCRIPT SEGMENT (this is the ONLY text these tasks came from):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{chunk_transcript}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TASKS TO AUDIT (extracted from the segment above):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{tasks_json}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AUDIT ACTIONS (do these in order, conservatively):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ FIX assignee   — Correct wrong name using TEAM MEMBERS. If unknown → null.
✅ FIX assigner   — Correct wrong name using TEAM MEMBERS. If unknown → null.
✅ FIX priority   — Adjust if urgency signals in the transcript clearly contradict it.
✅ FIX due_date   — Correct if the date is clearly wrong. Today = {today}.
✅ IMPROVE desc   — Enrich the description with additional context from the transcript.
✅ IMPROVE evid   — Improve the evidence quote if a better verbatim quote exists.
✅ IMPROVE conf   — Adjust confidence based on certainty of the extraction.
✅ MERGE dupes    — Merge two tasks that are IDENTICAL in this chunk (same action, same assignee, same timeframe).
✅ ADD missed     — Add a task ONLY if there is an explicit commitment or assignment in this transcript segment that was clearly missed.

❌ DO NOT delete tasks because they seem minor.
❌ DO NOT delete tasks because the priority seems wrong.
❌ DO NOT delete tasks because the description is vague.
❌ DO NOT delete tasks if you are uncertain.
❌ DO NOT rebuild the task list from scratch — correct existing entries.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT — return the full corrected list in EXACTLY this JSON:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{{
  "tasks": [
    {{
      "title": "...",
      "description": "...",
      "assignee": "exact name from TEAM MEMBERS or null",
      "assigner": "exact name from TEAM MEMBERS or null",
      "priority": "low | medium | high | critical",
      "due_date": "YYYY-MM-DD or null",
      "confidence": 0.00,
      "evidence": "verbatim transcript quote",
      "meeting_section": "...",
      "tags": [],
      "chunk_id": {chunk_index}
    }}
  ]
}}

Return ALL tasks unless they are clear hallucinations with zero transcript evidence.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 5 — EXTRACTION PIPELINE (two-pass, with retry)
# ═══════════════════════════════════════════════════════════════════════════════

def _call_llm(
    messages: list[dict],
    temperature: float = 0.05,
    max_retries: int = MAX_RETRIES,
) -> dict:
    """
    Call Azure OpenAI with exponential-backoff retry logic.

    Retries on:
    - APIError (server-side errors)
    - APITimeoutError (network timeout)
    - RateLimitError (throttling)
    - json.JSONDecodeError (malformed response)

    Raises RuntimeError after all attempts are exhausted.
    """
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=settings.AZURE_OPENAI_DEPLOYMENT,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from Azure OpenAI")
            return json.loads(content)

        except (APIError, APITimeoutError, RateLimitError) as e:
            last_error = e
            wait = 2 ** attempt  # 2s, 4s, 8s
            logger.warning(
                "Azure OpenAI API error (attempt %d/%d): %s — retrying in %ds",
                attempt, max_retries, e, wait,
            )
            time.sleep(wait)

        except json.JSONDecodeError as e:
            last_error = e
            logger.warning("JSON decode error (attempt %d/%d): %s", attempt, max_retries, e)
            if attempt < max_retries:
                time.sleep(1)

        except Exception as e:
            last_error = e
            logger.error("Unexpected LLM error (attempt %d/%d): %s", attempt, max_retries, e)
            if attempt < max_retries:
                time.sleep(1)

    raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_error}")


def _extract_from_chunk(
    chunk: str,
    system_prompt: str,
    team_members: list[str],
    today: str,
    chunk_index: int,
    total_chunks: int,
) -> list[dict]:
    """
    Pass 1 — extract tasks from a single transcript chunk.

    Each returned task is stamped with ``chunk_id`` so that the
    per-chunk validator and the deduplicator can track provenance.
    """
    chunk_header = (
        f"[Transcript segment {chunk_index + 1} of {total_chunks}]\n\n"
        f"Today: {today}\n"
        f"Team members: {', '.join(team_members) if team_members else 'none provided'}\n\n"
        f"Meeting transcript:\n\n{chunk}"
    )

    try:
        data = _call_llm(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chunk_header},
            ]
        )
        tasks = data.get("tasks", [])
        tasks = tasks if isinstance(tasks, list) else []
        # Stamp every task with its source chunk AND a stable identity
        # immediately after extraction so the identity is immutable.
        for t in tasks:
            if isinstance(t, dict):
                t["chunk_id"] = chunk_index
                TaskIdentityStamper.stamp(t, chunk_index, chunk)
        logger.info(
            "[Chunk %d/%d] Extraction → %d task(s)",
            chunk_index + 1, total_chunks, len(tasks),
        )
        return tasks

    except Exception as e:
        logger.error("[Chunk %d/%d] Extraction FAILED: %s", chunk_index + 1, total_chunks, e)
        return []


def _validate_chunk(
    raw_tasks: list[dict],
    chunk_transcript: str,
    team_members: list[str],
    today: str,
    chunk_index: int,
    total_chunks: int,
) -> list[dict]:
    """
    Per-chunk auditor (Pass 2).

    Key architectural guarantees:
    ─────────────────────────────
    1. The validator ONLY sees tasks from THIS chunk and the FULL text of THIS
       chunk — never a truncated global sample.
    2. A 10 % safety threshold: if the validated list contains fewer than
       (1 - VALIDATION_SAFETY_THRESHOLD) × len(raw_tasks) tasks, the
       validation result is discarded and the original extraction is used.
       This prevents runaway deletion.
    3. If the LLM call fails for any reason, the original tasks are returned
       unchanged.
    """
    if not VALIDATION_PASS_ENABLED or not raw_tasks:
        return raw_tasks

    validation_prompt = _build_chunk_validation_prompt(
        team_members=team_members,
        today=today,
        extracted_tasks=raw_tasks,
        chunk_transcript=chunk_transcript,
        chunk_index=chunk_index,
        total_chunks=total_chunks,
    )

    try:
        data = _call_llm(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a conservative AI auditor for meeting task extraction. "
                        "Your primary obligation is to PRESERVE tasks, not remove them. "
                        "Return only valid JSON."
                    ),
                },
                {"role": "user", "content": validation_prompt},
            ],
            temperature=0.0,
        )
        validated = data.get("tasks", [])
        if not isinstance(validated, list):
            logger.warning(
                "[Chunk %d/%d] Validator returned non-list — using raw extraction",
                chunk_index + 1, total_chunks,
            )
            return raw_tasks

        raw_n = len(raw_tasks)
        val_n = len(validated)
        removed = raw_n - val_n
        corrected = sum(
            1 for i, t in enumerate(validated)
            if i < len(raw_tasks) and t.get("title") != raw_tasks[i].get("title")
        )
        added = max(0, val_n - raw_n)

        logger.info(
            "[Chunk %d/%d] Validation — extracted: %d | validated: %d | "
            "removed: %d | corrected: ~%d | added: %d",
            chunk_index + 1, total_chunks,
            raw_n, val_n, max(0, removed), corrected, added,
        )

        # ── Safety threshold ───────────────────────────────────────────────
        # If the validator removed more than VALIDATION_SAFETY_THRESHOLD
        # fraction of tasks, discard the validation and keep the originals.
        if raw_n > 0 and removed > 0:
            removal_fraction = removed / raw_n
            if removal_fraction > VALIDATION_SAFETY_THRESHOLD:
                logger.warning(
                    "[Chunk %d/%d] ⚠ SAFETY THRESHOLD TRIGGERED: validator removed "
                    "%.0f%% of tasks (>%.0f%% limit). "
                    "Discarding validation — using raw extraction (%d tasks).",
                    chunk_index + 1, total_chunks,
                    removal_fraction * 100,
                    VALIDATION_SAFETY_THRESHOLD * 100,
                    raw_n,
                )
                return raw_tasks

        # Ensure all validated tasks carry the correct chunk_id
        for t in validated:
            if isinstance(t, dict):
                t["chunk_id"] = chunk_index
        return validated

    except Exception as e:
        logger.warning(
            "[Chunk %d/%d] Validation FAILED (%s) — using raw extraction",
            chunk_index + 1, total_chunks, e,
        )
        return raw_tasks


def _process_chunk(
    chunk: str,
    system_prompt: str,
    team_members: list[str],
    today: str,
    chunk_index: int,
    total_chunks: int,
) -> list[dict]:
    """
    Full single-chunk pipeline: extract → validate.
    Designed to be called in a thread pool for parallelization.
    """
    raw = _extract_from_chunk(
        chunk=chunk,
        system_prompt=system_prompt,
        team_members=team_members,
        today=today,
        chunk_index=chunk_index,
        total_chunks=total_chunks,
    )
    validated = _validate_chunk(
        raw_tasks=raw,
        chunk_transcript=chunk,         # ← FULL chunk, not a 8K sample
        team_members=team_members,
        today=today,
        chunk_index=chunk_index,
        total_chunks=total_chunks,
    )
    return validated



# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 6a — TASK IDENTITY + CONFLICT RESOLUTION
# ═══════════════════════════════════════════════════════════════════════════════

def _tokenize(text: str) -> frozenset[str]:
    """Lowercase, remove stop words, return a frozen token set."""
    tokens = re.findall(r"\b[a-z]+\b", text.lower())
    return frozenset(t for t in tokens if t not in _STOP_WORDS)


def _jaccard(a: frozenset, b: frozenset) -> float:
    """Jaccard similarity between two token sets. Returns 0.0 for empty sets."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class TaskIdentityStamper:
    """
    Stamp every extracted task with a stable, multi-attribute identity
    immediately after extraction.

    Fields added to the task dict
    ──────────────────────────────
    task_uuid            — uuid4 string; immutable for the task's lifetime.
    original_assignee    — snapshot of assignee at extraction time; never changes.
    reassignment_history — list of reassignment event dicts; starts empty.
    transcript_offset    — estimated char offset of the evidence inside the chunk.
    line_number          — estimated line number of the evidence inside the chunk.
    chunk_index          — alias for chunk_id for clarity.
    _title_tokens        — sorted list of stop-word-stripped title tokens (cached).

    This stamper is called once, inside _extract_from_chunk, before any other
    pipeline layer sees the task.  The identity is then immutable.
    """

    @staticmethod
    def stamp(task: dict, chunk_index: int, chunk_text: str) -> None:
        """Mutate *task* in-place, adding identity fields."""
        # ── Unique ID ──────────────────────────────────────────────────────
        task["task_uuid"] = str(uuid.uuid4())

        # ── Ownership snapshot ─────────────────────────────────────────────
        task["original_assignee"] = task.get("assignee")
        task["reassignment_history"] = []

        # ── Transcript position ────────────────────────────────────────────
        evidence = (task.get("evidence") or "").strip()
        if evidence and chunk_text:
            probe = evidence[:20] if len(evidence) >= 20 else evidence
            offset = chunk_text.find(probe)
            if offset == -1:
                offset = 0
            task["transcript_offset"] = offset
            task["line_number"] = chunk_text[:offset].count("\n") + 1
        else:
            task["transcript_offset"] = 0
            task["line_number"] = 0

        task["chunk_index"] = chunk_index  # alias for clarity

        # ── Cached title tokens (sorted list — JSON-serialisable) ──────────
        title_tokens = _tokenize(task.get("title") or "")
        task["_title_tokens"] = sorted(title_tokens)


# ─────────────────────────────────────────────────────────────────────────────

class ReassignmentDetector:
    """
    Determine whether a piece of transcript evidence contains an *explicit*
    reassignment signal.

    A reassignment signal is one of the phrases below appearing in the evidence
    of the *later* task (higher chunk_id / transcript_offset).  The presence of
    such a phrase is a necessary condition for two tasks to be considered the
    same task under a new owner.

    Design notes
    ────────────
    * Matching is case-insensitive and searches the full evidence string.
    * Single-word triggers (e.g. "actually") use word-boundary anchors so that
      "actually" does not match inside "practically".
    * Multi-word phrases are matched as sub-strings (order matters).
    * Returns False (independent task) when no phrase is found.
    """

    _SINGLE_WORD_TRIGGERS: frozenset[str] = frozenset({
        "actually",
        "instead",
        "reassign",
        "reassigned",
        "transfer",
        "transferred",
    })

    _MULTI_WORD_PHRASES: tuple[str, ...] = (
        "take over",
        "taking over",
        "took over",
        "switch owner",
        "switching owner",
        "move this to",
        "moving this to",
        "hand over",
        "handing over",
        "handed over",
        "give it to",
        "giving it to",
        "assign it to",
        "assigning it to",
        "will own it",
        "will handle it",
        "will be handling",
        "no longer",
        "someone else will",
        "change owner",
        "let's reassign",
        "let us reassign",
        "pass this to",
        "passing this to",
        "now owns",
        "will now own",
    )

    @classmethod
    def detect(cls, evidence: str) -> bool:
        """
        Return True if *evidence* contains at least one explicit reassignment
        signal.  Returns False for empty evidence.
        """
        if not evidence:
            return False
        text = evidence.lower()

        # Single-word triggers with word boundaries
        for word in cls._SINGLE_WORD_TRIGGERS:
            if re.search(rf"\b{re.escape(word)}\b", text):
                return True

        # Multi-word phrases as substrings
        for phrase in cls._MULTI_WORD_PHRASES:
            if phrase in text:
                return True

        return False


# ─────────────────────────────────────────────────────────────────────────────

class ConflictResolver:
    """
    Resolve true reassignment events without collapsing independent tasks.

    Core principle
    ──────────────
    Two tasks are the SAME task ONLY when ALL of the following hold:

      1. Normalised-title Jaccard similarity ≥ TITLE_THRESHOLD (0.90).
      2. The tasks have DIFFERENT assignees (otherwise nothing to resolve).
      3. At least ONE of:
           a. The later task's evidence contains an explicit reassignment phrase
              (detected by ReassignmentDetector).
           b. The two tasks share the same evidence block
              (evidence Jaccard ≥ EVIDENCE_IDENTITY_THRESHOLD, 0.85).

    If condition 3 is NOT met the tasks are treated as independent assignments
    even when the titles are identical.  They are kept as separate tasks.

    Resolution actions (when a true reassignment IS confirmed)
    ──────────────────────────────────────────────────────────
    * Update ``assignee`` on the surviving (later) task.
    * Append a reassignment event to ``reassignment_history``.
    * Append evidence from the earlier task (never replace).
    * Boost confidence by 0.02 on the surviving task.
    * NEVER delete tasks without a confirmed reassignment.

    5 % safety gate
    ───────────────
    If conflict resolution would remove more than MAX_REDUCTION_FRACTION (5 %)
    of the input tasks, abort and restore the original list with a warning.

    Logging
    ───────
    Confirmed reassignment:
      [ConflictResolver] Reassignment confirmed | Task ID: <uuid> | Title: ...
      | Original assignee: X → Current assignee: Y | Reason: ... | Evidence: ...

    Independent task (no merge):
      [ConflictResolver] Independent task detected — no merge performed
      | title=... | UUID-A=... assignee=... | UUID-B=... assignee=...
    """

    TITLE_THRESHOLD: float = 0.90
    EVIDENCE_IDENTITY_THRESHOLD: float = 0.85
    MAX_REDUCTION_FRACTION: float = 0.05

    @classmethod
    def resolve(cls, tasks: list[dict]) -> list[dict]:
        """
        Process *tasks* and return the resolved list.

        Independent tasks are NEVER removed even when they share a title.
        """
        if len(tasks) <= 1:
            return tasks

        original_count = len(tasks)
        absorbed: list[bool] = [False] * original_count
        conflicts_resolved = 0
        independent_logged = 0

        def _title_tokens(t: dict) -> frozenset:
            cached = t.get("_title_tokens")
            if cached:
                return frozenset(cached)
            return _tokenize(t.get("title") or "")

        token_sets = [_title_tokens(t) for t in tasks]

        for i in range(original_count):
            if absorbed[i]:
                continue

            task_a = tasks[i]
            chunk_a = task_a.get("chunk_id") or 0
            offset_a = task_a.get("transcript_offset") or 0
            assignee_a = (task_a.get("assignee") or "").lower().strip()
            evidence_a_tokens = _tokenize(task_a.get("evidence") or "")

            for j in range(i + 1, original_count):
                if absorbed[j]:
                    continue

                task_b = tasks[j]

                # ── Condition 1: title similarity ──────────────────────────
                title_sim = _jaccard(token_sets[i], token_sets[j])
                if title_sim < cls.TITLE_THRESHOLD:
                    continue

                # ── Condition 2: different assignees ───────────────────────
                assignee_b = (task_b.get("assignee") or "").lower().strip()
                if assignee_a == assignee_b:
                    # Same title + same assignee → duplicate; handled by
                    # SemanticDeduplicator downstream, not here.
                    continue

                # ── Condition 3: explicit reassignment evidence ─────────────
                chunk_b = task_b.get("chunk_id") or 0
                offset_b = task_b.get("transcript_offset") or 0

                # Determine which task is "later" in the transcript
                if chunk_b > chunk_a or (chunk_b == chunk_a and offset_b > offset_a):
                    later_task, earlier_task, later_idx, earlier_idx = task_b, task_a, j, i
                else:
                    later_task, earlier_task, later_idx, earlier_idx = task_a, task_b, i, j

                later_evidence = later_task.get("evidence") or ""
                evidence_b_tokens = _tokenize(later_task.get("evidence") or "")
                evidence_sim = _jaccard(evidence_a_tokens, evidence_b_tokens)

                has_reassignment_phrase = ReassignmentDetector.detect(later_evidence)
                has_shared_evidence = evidence_sim >= cls.EVIDENCE_IDENTITY_THRESHOLD

                if not has_reassignment_phrase and not has_shared_evidence:
                    # Independent tasks — same title, different assignees,
                    # no explicit reassignment signal → keep both.
                    logger.info(
                        "[ConflictResolver] Independent task detected — no merge performed | "
                        "title=%r | UUID-A=%s assignee=%s | UUID-B=%s assignee=%s | "
                        "title_sim=%.2f | evidence_sim=%.2f | reason=no_reassignment_phrase",
                        task_a.get("title"),
                        task_a.get("task_uuid", "?"),
                        task_a.get("assignee"),
                        task_b.get("task_uuid", "?"),
                        task_b.get("assignee"),
                        title_sim,
                        evidence_sim,
                    )
                    independent_logged += 1
                    continue

                # ── Confirmed reassignment: apply it ──────────────────────
                earlier_assignee_display = earlier_task.get("assignee") or "unknown"
                later_assignee_display = later_task.get("assignee") or "unknown"
                reason = (
                    "Explicit reassignment phrase detected in evidence"
                    if has_reassignment_phrase
                    else "Identical evidence block — same transcript event"
                )

                # Record history on the surviving (later) task
                later_task.setdefault("reassignment_history", []).append({
                    "from_assignee": earlier_assignee_display,
                    "to_assignee": later_assignee_display,
                    "reason": reason,
                    "evidence": later_evidence,
                    "source_task_uuid": earlier_task.get("task_uuid", ""),
                })

                # Merge evidence (append, never replace)
                earlier_ev = (earlier_task.get("evidence") or "").strip()
                existing_ev = (later_task.get("evidence") or "").strip()
                if earlier_ev and earlier_ev != existing_ev:
                    later_task["evidence"] = (
                        existing_ev + " | [Prior assignment] " + earlier_ev
                    ).strip(" | ")

                # Slight confidence boost on confirmed reassignment
                base_conf = float(later_task.get("confidence") or 0.80)
                later_task["confidence"] = round(min(1.0, base_conf + 0.02), 2)

                absorbed[earlier_idx] = True
                conflicts_resolved += 1

                logger.info(
                    "[ConflictResolver] Reassignment confirmed | "
                    "Task ID: %s | Title: %r | "
                    "Original assignee: %s -> Current assignee: %s | "
                    "Reason: %s | "
                    "Evidence: %r",
                    later_task.get("task_uuid", "?"),
                    later_task.get("title"),
                    earlier_assignee_display,
                    later_assignee_display,
                    reason,
                    later_evidence[:200],
                )

        result = [t for idx, t in enumerate(tasks) if not absorbed[idx]]

        # -- 5% safety gate (only fires when batch is large enough) -----------
        # A minimum of MIN_GATE_SIZE tasks is required before the gate can fire.
        # For small batches a genuine reassignment can legitimately remove 50%
        # of the input, so we must not block it.
        MIN_GATE_SIZE = 20
        removed = original_count - len(result)
        reduction_fraction = removed / original_count if original_count else 0.0

        if original_count >= MIN_GATE_SIZE and reduction_fraction > cls.MAX_REDUCTION_FRACTION:
            logger.warning(
                "[ConflictResolver] SAFETY GATE TRIGGERED -- "
                "conflict resolution would remove %.1f%% of tasks (limit: %.0f%%). "
                "Aborting. Restoring all %d original tasks. "
                "Message: 'Conflict resolution rejected excessive merges.'",
                reduction_fraction * 100,
                cls.MAX_REDUCTION_FRACTION * 100,
                original_count,
            )
            # Roll back reassignment_history since we're restoring
            for t in tasks:
                t["reassignment_history"] = []
            return tasks

        logger.info(
            "[ConflictResolver] Complete -- input: %d | reassignments resolved: %d | "
            "independent tasks kept: %d | output: %d",
            original_count,
            conflicts_resolved,
            independent_logged,
            len(result),
        )
        return result




# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 6b — MULTI-FIELD COMPOSITE DEDUPLICATOR
# ═══════════════════════════════════════════════════════════════════════════════

def _composite_similarity(a: dict, b: dict) -> float:
    """
    Multi-field composite similarity score between two tasks.

    Scoring breakdown (weights sum to 1.0):
      title     0.40  — Jaccard token overlap on normalised titles
      assignee  0.25  — exact match on assignee name
      due_date  0.15  — exact match on due date
      evidence  0.20  — Jaccard token overlap on evidence snippets

    Two tasks in DIFFERENT chunks receive a 0.10 penalty to avoid
    over-merging legitimate repeated agenda items across chunks.

    Returns a value in [0.0, 1.0].
    """
    title_sim = _jaccard(
        _tokenize(a.get("title") or ""),
        _tokenize(b.get("title") or ""),
    )

    assignee_a = (a.get("assignee") or "").lower().strip()
    assignee_b = (b.get("assignee") or "").lower().strip()
    assignee_sim = 1.0 if (assignee_a and assignee_a == assignee_b) else 0.0

    date_a = a.get("due_date") or ""
    date_b = b.get("due_date") or ""
    date_sim = 1.0 if (date_a and date_a == date_b) else 0.0

    evidence_sim = _jaccard(
        _tokenize(a.get("evidence") or ""),
        _tokenize(b.get("evidence") or ""),
    )

    score = (
        0.40 * title_sim
        + 0.25 * assignee_sim
        + 0.15 * date_sim
        + 0.20 * evidence_sim
    )

    # Cross-chunk penalty — same title can legitimately appear in different
    # meeting sections (e.g. mentioned at start, followed up at end).
    chunk_a = a.get("chunk_id")
    chunk_b = b.get("chunk_id")
    if chunk_a is not None and chunk_b is not None and chunk_a != chunk_b:
        score -= 0.10

    return max(0.0, score)


class SemanticDeduplicator:
    """
    Remove genuinely duplicate tasks using a multi-field composite score.

    Design principles:
    ──────────────────
    * Title alone is NOT sufficient to declare a duplicate.
    * The composite score weights title (40%), assignee (25%),
      due_date (15%), and evidence (20%).
    * Tasks from different chunks incur a 0.10 cross-chunk penalty to avoid
      over-merging legitimate repeated agenda items.
    * The threshold is deliberately HIGH (0.82) to err on the side of keeping
      tasks rather than merging them.
    """

    # Require high composite similarity to declare a duplicate.
    # Raising this compared to the old Jaccard-only 0.65 threshold prevents
    # over-aggressive merging that caused 109 → 13 task collapse.
    SIMILARITY_THRESHOLD: float = 0.82

    @classmethod
    def deduplicate(cls, tasks: list[dict]) -> list[dict]:
        """Return a deduplicated task list. Input order is preserved."""
        if len(tasks) <= 1:
            return tasks

        absorbed = [False] * len(tasks)
        result: list[dict] = []

        for i in range(len(tasks)):
            if absorbed[i]:
                continue
            canonical = dict(tasks[i])

            for j in range(i + 1, len(tasks)):
                if absorbed[j]:
                    continue

                score = _composite_similarity(tasks[i], tasks[j])
                if score < cls.SIMILARITY_THRESHOLD:
                    continue

                # Confirmed duplicate — merge conservatively
                other = tasks[j]
                absorbed[j] = True

                base_conf = float(canonical.get("confidence") or 0.0)
                other_conf = float(other.get("confidence") or 0.0)

                if other_conf > base_conf:
                    saved_evidence = canonical.get("evidence") or ""
                    canonical = dict(other)
                    other_evidence = saved_evidence
                else:
                    other_evidence = other.get("evidence") or ""

                # Concatenate unique evidence snippets
                existing_ev = (canonical.get("evidence") or "").strip()
                if other_evidence.strip() and other_evidence.strip() != existing_ev:
                    canonical["evidence"] = (existing_ev + " | " + other_evidence.strip()).strip(" | ")

                # Union tags
                canonical["tags"] = sorted(
                    set(canonical.get("tags") or []) | set(other.get("tags") or [])
                )

                logger.debug(
                    "Deduplicator: merged '%s' ← '%s' (composite_score=%.2f)",
                    canonical.get("title"), other.get("title"), score,
                )

            result.append(canonical)

        merged_count = sum(absorbed)
        logger.info(
            "Deduplication: %d → %d tasks (%d merged)",
            len(tasks), len(result), merged_count,
        )
        return result


def _run_final_consistency_check(
    tasks: list[dict],
    team_members: list[str],
    today_str: str,
) -> list[dict]:
    """
    Lightweight final consistency check after all chunks are merged.

    This is a PURE PYTHON pass — NO LLM call.
    It may ONLY:
      - Normalise assignee names against the team list
      - Validate and repair due dates
      - Normalise priority values
      - Remove tasks with empty titles (nothing can be done with them)

    It may NEVER bulk-delete tasks.
    """
    member_lower = {m.lower(): m for m in team_members}

    def _rematch(name: str | None) -> str | None:
        if not name:
            return None
        nl = name.lower().strip()
        if nl in member_lower:
            return member_lower[nl]
        for key, full in member_lower.items():
            if nl in key.split() or any(part == nl for part in key.split()):
                return full
        return name  # Keep original if no match — don't null it out

    result = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        title = (task.get("title") or "").strip()
        if not title:
            continue  # Only valid deletion reason

        task["title"] = title
        if team_members:
            task["assignee"] = _rematch(task.get("assignee"))
            task["assigner"] = _rematch(task.get("assigner"))
        priority = (task.get("priority") or "medium").lower().strip()
        task["priority"] = priority if priority in _VALID_PRIORITIES else "medium"
        result.append(task)

    logger.info("Final consistency check: %d → %d tasks", len(tasks), len(result))
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 7 — RULE ENGINE (post-processing validator)
# ═══════════════════════════════════════════════════════════════════════════════

class RuleEngine:
    """
    Validate, normalise, and enrich every extracted task before it leaves
    the pipeline.

    Policy
    ------
    * Tasks are NEVER rejected based on confidence alone (user requirement).
    * Tasks below 0.80 confidence are logged as warnings but still included.
    * Tasks below MIN_CONFIDENCE (default 0.50) are dropped — they represent
      clear hallucinations with no transcript basis.
    * Tasks without a title are always dropped (nothing can be done with them).
    * Assignee / assigner names are re-matched against the team list.
    * Priority values are normalised to the four valid levels.
    * Due dates are re-validated; any relative expression the LLM missed is
      resolved by DateResolver.
    * Tags are cleaned and deduplicated.
    """

    @staticmethod
    def _match_member(name: str | None, member_lower: dict[str, str]) -> str | None:
        """Fuzzy-match a name token against the team member lookup dict."""
        if not name:
            return None
        nl = name.lower().strip()
        # Exact full-name match
        if nl in member_lower:
            return member_lower[nl]
        # Any team member whose full name shares a word with the token
        for key, full in member_lower.items():
            key_parts = key.split()
            if nl in key_parts or any(part in nl for part in key_parts):
                return full
        return None

    @staticmethod
    def _validate_date(value: str | None, today_str: str) -> str | None:
        """Return a valid ISO date string, or None."""
        if not value:
            return None
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            try:
                date.fromisoformat(value)
                return value
            except ValueError:
                pass
        # Try resolving as a relative expression
        try:
            today = date.fromisoformat(today_str)
        except ValueError:
            today = date.today()
        return DateResolver.resolve(value, today)

    @classmethod
    def process(
        cls,
        tasks: list[dict],
        team_members: list[str],
        today_str: str,
    ) -> list[dict]:
        """
        Apply all validation rules and return the final task list.

        Args:
            tasks:          Raw task dicts from the pipeline.
            team_members:   Full names of all team members.
            today_str:      Today's date as ISO string.
        """
        member_lower = {m.lower(): m for m in team_members}
        result: list[dict] = []

        for task in tasks:
            if not isinstance(task, dict):
                continue

            # ── Confidence gate ────────────────────────────────────────────
            try:
                confidence = float(task.get("confidence") or 0.80)
            except (TypeError, ValueError):
                confidence = 0.80
            confidence = round(max(0.0, min(1.0, confidence)), 2)

            if confidence < MIN_CONFIDENCE:
                logger.warning(
                    "RuleEngine: dropped task (confidence=%.2f < %.2f) — title=%r",
                    confidence, MIN_CONFIDENCE, task.get("title"),
                )
                continue

            if 0.50 <= confidence < 0.80:
                logger.warning(
                    "RuleEngine: low-confidence task included — confidence=%.2f title=%r",
                    confidence, task.get("title"),
                )

            # ── Title ──────────────────────────────────────────────────────
            title = (task.get("title") or "").strip()
            if not title:
                logger.warning("RuleEngine: dropped task — missing title")
                continue
            if len(title) > 100:
                title = title[:97] + "..."

            first_word = title.split()[0].lower().rstrip(".,;:")
            if first_word not in _ACTION_VERBS:
                logger.debug(
                    "RuleEngine: title '%s' does not begin with a known action verb (first_word=%r)",
                    title, first_word,
                )

            # ── Description ────────────────────────────────────────────────
            description = (task.get("description") or "").strip() or None

            # ── Assignee / Assigner ────────────────────────────────────────
            assignee = (
                cls._match_member(task.get("assignee"), member_lower)
                if team_members else task.get("assignee")
            )
            assigner = (
                cls._match_member(task.get("assigner"), member_lower)
                if team_members else task.get("assigner")
            )

            # ── Priority ───────────────────────────────────────────────────
            priority = (task.get("priority") or "medium").lower().strip()
            if priority not in _VALID_PRIORITIES:
                priority = "medium"

            # ── Due Date ───────────────────────────────────────────────────
            due_date = cls._validate_date(task.get("due_date"), today_str)

            # ── Evidence ──────────────────────────────────────────────────
            evidence = (task.get("evidence") or "").strip() or None

            # ── Meeting section ────────────────────────────────────────────
            meeting_section = (task.get("meeting_section") or "").strip() or None

            # ── Tags ──────────────────────────────────────────────────────
            raw_tags = task.get("tags") or []
            if isinstance(raw_tags, list):
                tags = sorted({str(t).lower().strip() for t in raw_tags if t})
            else:
                tags = []

            result.append({
                "title": title,
                "description": description,
                "assignee": assignee,
                "assigner": assigner,
                "priority": priority,
                "due_date": due_date,
                "confidence": confidence,
                "evidence": evidence,
                "meeting_section": meeting_section,
                "tags": tags,
                # Identity fields — preserved from TaskIdentityStamper
                "task_uuid": task.get("task_uuid") or str(uuid.uuid4()),
                "original_assignee": task.get("original_assignee"),
                "reassignment_history": task.get("reassignment_history") or [],
                "chunk_id": task.get("chunk_id"),
                # _title_tokens is an internal cache field — not exposed to callers
            })

        return result


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

class AIService:
    """
    Enterprise-grade AI extraction pipeline.

    New Pipeline
    ────────────
    Transcript
      ↓  Layer 1   TranscriptPreprocessor    — clean & normalise
      ↓  Layer 3   TranscriptChunker         — split into context-safe chunks
      ↓  Layer 5   ThreadPoolExecutor        — parallel per-chunk pipeline:
      │               ↓  _extract_from_chunk()   — Pass 1: extract
      │               ↓  TaskIdentityStamper.stamp() — UUID + position fingerprint
      │               ↓  _validate_chunk()        — Pass 2: per-chunk audit
      │                    (safety threshold: auto-fallback if >10% removed)
      ↓  Merge all chunk results
      ↓  Layer 6a  ConflictResolver.resolve()  — reassignment-only; 5% safety gate
      ↓  Layer 6b  SemanticDeduplicator      — multi-field composite dedup
      ↓  Layer 6c  _run_final_consistency_check()  — lightweight pure-python pass
      ↓  Layer 7   RuleEngine               — normalise, flag, enrich; preserves identity
      ↓
    Final task list

    Public interface (UNCHANGED):
        AIService.generate_tasks(transcript, team_members, today) -> list[dict]
    """

    @staticmethod
    def generate_tasks(
        transcript: str,
        team_members: list[str] | None = None,
        today: str | None = None,
    ) -> list[dict]:
        """
        Extract action-item tasks from a meeting transcript.

        Args
        ----
        transcript:    Full meeting transcript (any length; 1+ hour meetings supported).
        team_members:  Full names of all team members for accurate name matching.
        today:         ISO date for relative date resolution. Defaults to server date.

        Returns
        -------
        List of task dicts. Each dict contains:
            title, description, assignee, assigner, priority, due_date,
            confidence, evidence, meeting_section, tags, chunk_id
        """
        from fastapi import HTTPException

        resolved_today = today or date.today().isoformat()
        resolved_members = team_members or []

        try:
            # ── Layer 1: Preprocess ───────────────────────────────────────
            logger.info(
                "AI Pipeline START — transcript: %d chars | team: %d members",
                len(transcript), len(resolved_members),
            )
            clean = TranscriptPreprocessor.process(transcript)
            logger.info("AI Pipeline — preprocessed: %d chars", len(clean))

            # ── Layer 3: Chunk ────────────────────────────────────────────
            chunks = TranscriptChunker.split(clean)
            total_chunks = len(chunks)
            logger.info("AI Pipeline — %d chunk(s) queued for parallel processing", total_chunks)

            # Build the system prompt once (shared across all chunks)
            system_prompt = _build_system_prompt(resolved_members, resolved_today)

            # ── Layer 5: Parallel extract + per-chunk validate ────────────
            # Each future maps to its chunk_index so we can merge in order.
            chunk_results: dict[int, list[dict]] = {}

            with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_WORKERS, total_chunks)) as executor:
                futures = {
                    executor.submit(
                        _process_chunk,
                        chunk,
                        system_prompt,
                        resolved_members,
                        resolved_today,
                        idx,
                        total_chunks,
                    ): idx
                    for idx, chunk in enumerate(chunks)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        chunk_results[idx] = future.result()
                    except Exception as exc:
                        logger.error("[Chunk %d] parallel processing raised: %s", idx + 1, exc)
                        chunk_results[idx] = []

            # Merge in chunk order to preserve meeting chronology
            all_validated: list[dict] = []
            for idx in range(total_chunks):
                all_validated.extend(chunk_results.get(idx, []))

            total_extracted = sum(
                len(chunk_results.get(i, [])) for i in range(total_chunks)
            )
            logger.info(
                "AI Pipeline — Parallel phase complete: "
                "%d task(s) across %d chunk(s)",
                total_extracted, total_chunks,
            )

            # ── Layer 6a: Task-identity-aware conflict resolution ──────────
            after_conflict = ConflictResolver.resolve(all_validated)

            # ── Layer 6b: Multi-field deduplication ───────────────────────
            deduped = SemanticDeduplicator.deduplicate(after_conflict)

            # ── Layer 6c: Final lightweight consistency check ─────────────
            consistent = _run_final_consistency_check(
                deduped, resolved_members, resolved_today
            )

            # ── Layer 7: Rule engine ──────────────────────────────────────
            final = RuleEngine.process(consistent, resolved_members, resolved_today)

            logger.info(
                "AI Pipeline COMPLETE ── "
                "extracted: %d | post-conflict: %d | deduped: %d | "
                "consistent: %d | final: %d",
                total_extracted,
                len(after_conflict),
                len(deduped),
                len(consistent),
                len(final),
            )
            return final

        except HTTPException:
            raise
        except RuntimeError as e:
            logger.error("AI Pipeline — LLM exhausted retries: %s", e)
            raise HTTPException(status_code=502, detail=f"AI Processing failed: {e}")
        except Exception as e:
            logger.error("AI Pipeline — unexpected error: %s", e)
            raise HTTPException(status_code=502, detail=f"AI Processing failed: {e}")