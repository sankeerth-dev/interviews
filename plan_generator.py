"""Interview Plan Generator.

Runs ONCE before the greeting. Analyzes resume + job description to produce:
  - A seniority estimate (junior/mid/senior/staff_principal)
  - An ordered list of TopicPlan objects (what to cover, in what order, with what depth)
  - A flag indicating whether a coding assessment is needed

Why this exists (vs the old reactive approach):
  Old system: LLM invents a question every turn with no blueprint — leads to random
  topic jumps, missed coverage, and uniform depth regardless of candidate level.

  New system: We build a structured plan upfront. The question_node picks from the plan.
  The depth engine (in control_nodes) decides when a topic is "done" based on seniority
  rules, so a Staff Engineer gets pushed much harder than a junior candidate.
"""

import logging
import json
import uuid
from typing import Optional

from src.services.orchestrator.llm_helpers import LLMHelper
from src.services.orchestrator.context_builders import build_resume_context, build_job_context, build_skills_context
from src.services.orchestrator.constants import (
    SENIORITY_JUNIOR, SENIORITY_MID, SENIORITY_SENIOR, SENIORITY_STAFF,
    SENIORITY_LEVELS, DEPTH_RULES,
    TOPIC_BACKGROUND, TOPIC_TECHNICAL, TOPIC_BEHAVIORAL,
    TOPIC_SITUATIONAL, TOPIC_PROJECT, TOPIC_CODING, TOPIC_CATEGORIES,
    COVERAGE_PENDING, COVERAGE_IN_PROGRESS, PRIORITY_MUST_ASK, PRIORITY_SHOULD_ASK, PRIORITY_NICE_TO_HAVE,
    STYLE_TECHNICAL_HEAVY, STYLE_BEHAVIORAL_HEAVY, STYLE_BALANCED,
    TEMPERATURE_ANALYTICAL, TEMPERATURE_BALANCED,
    PLAN_MIN_TOPICS, PLAN_MAX_TOPICS, PLAN_MIN_TARGET_TURNS, PLAN_MAX_TARGET_TURNS,
    QUESTION_BANK_MIN_SIZE, QUESTION_BANK_MAX_SIZE,
    QUESTION_BANK_MIN_SIZE_SHORT, QUESTION_BANK_MAX_SIZE_SHORT,
    QUESTION_BANK_MIN_SIZE_CODING, QUESTION_BANK_MAX_SIZE_CODING,
    TOPIC_CODING_MAX_ITERATIONS,
    DIFFICULTY_MEDIUM, DIFFICULTY_TOPIC_INSTRUCTIONS,
    INTERVIEW_MODE_JD_AND_RESUME, INTERVIEW_MODE_RESUME_ONLY, INTERVIEW_MODE_JD_ONLY,
    INTERVIEW_MODE_SKILLS_ONLY, INTERVIEW_MODES,
    USER_INTERVIEW_TECHNICAL, USER_INTERVIEW_BEHAVIORAL_TECHNICAL, USER_INTERVIEW_HR,
    HR_SYSTEM_PROMPT,
    HR_STAGE_OPENING, HR_STAGE_BACKGROUND, HR_STAGE_BEHAVIORAL,
    HR_STAGE_CULTURE, HR_STAGE_LOGISTICS, HR_STAGE_CLOSE,
)

logger = logging.getLogger(__name__)


_BANK_FALLBACK_BY_CATEGORY: dict[str, list[str]] = {
    TOPIC_TECHNICAL: [
        "What trade-offs did you evaluate when choosing this approach?",
        "How did you validate correctness, reliability, and performance for this work?",
        "What would you optimize first if this system needed to handle 10× the current load?",
        "What failure modes did you anticipate and how did you defend against them?",
    ],
    TOPIC_BEHAVIORAL: [
        "Can you walk me through what happened next and what you learned from that experience?",
        "How did others on the team respond, and how did you adapt your approach?",
        "What would you do differently if you faced a similar situation again?",
        "How did you communicate your perspective while staying respectful of others?",
    ],
    TOPIC_SITUATIONAL: [
        "What factors would you weigh before deciding on a course of action?",
        "How would you balance competing priorities or stakeholder expectations?",
        "What risks would you flag early, and how would you mitigate them?",
        "How would you follow up to ensure the outcome was fair and sustainable?",
    ],
    TOPIC_BACKGROUND: [
        "What drew you to this path, and what keeps you motivated in your work?",
        "How do you prefer to collaborate with managers and teammates?",
        "What kind of feedback helps you grow the most?",
    ],
    TOPIC_PROJECT: [
        "What was the hardest technical constraint and how did you work around it?",
        "Walk me through the key architectural decision — what alternatives did you consider?",
        "How did you handle data consistency or reliability guarantees in this system?",
        "If you rebuilt this system from scratch today, what would you design differently?",
    ],
    TOPIC_CODING: [
        "What is the time and space complexity of your solution? Is there a more efficient approach?",
        "How does your solution handle edge cases — empty input, duplicates, overflow, no valid answer?",
        "Go ahead and implement this in the code editor. Walk me through your thinking as you code.",
        "Now that you've written it, walk me through the implementation. Any differences from your original plan?",
        "Can you optimize it further — in time, space, or both? What trade-offs would that involve?",
        "How would you test this solution? What test cases would you write to validate correctness?",
    ],
}


def _detect_interview_mode(has_resume: bool, has_jd: bool, has_skills: bool) -> str:
    """Derive the interview mode from what context is available.

    Called once at plan time. The mode drives which prompt variant is used for
    seniority estimation and topic generation, ensuring the LLM never hallucinates
    missing context.

    Priority: skills_only > jd_only > resume_only
    If multiple contexts are present, precedence follows the order above.
    When neither is present, falls back to resume_only (generic experience questions).
    """
    if has_skills:
        return INTERVIEW_MODE_SKILLS_ONLY
    if has_resume and has_jd:
        return INTERVIEW_MODE_JD_AND_RESUME
    if has_resume:
        return INTERVIEW_MODE_RESUME_ONLY
    if has_jd:
        return INTERVIEW_MODE_JD_ONLY
    return INTERVIEW_MODE_RESUME_ONLY  # no context at all — ask generic technical questions


def _normalize_question_bank(raw_bank: object, category: str = "") -> tuple[int, list[str]]:
    """Normalize question_bank entries: trim, remove empties, dedupe, cap size."""
    raw_count = len(raw_bank) if isinstance(raw_bank, list) else 0
    if not isinstance(raw_bank, list):
        return raw_count, []

    cap = QUESTION_BANK_MAX_SIZE_CODING if category == TOPIC_CODING else QUESTION_BANK_MAX_SIZE
    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw_bank:
        if not isinstance(item, str):
            continue
        cleaned = " ".join(item.strip().split())
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(cleaned)
        if len(normalized) >= cap:
            break

    return raw_count, normalized


def _required_bank_size(max_iterations: int, category: str = "") -> int:
    """Return required minimum bank size based on topic depth profile."""
    if category == TOPIC_CODING:
        return QUESTION_BANK_MIN_SIZE_CODING
    return QUESTION_BANK_MIN_SIZE_SHORT if max_iterations == 1 else QUESTION_BANK_MIN_SIZE


def _bank_max_size(category: str = "") -> int:
    """Return the bank size cap for a given topic category."""
    return QUESTION_BANK_MAX_SIZE_CODING if category == TOPIC_CODING else QUESTION_BANK_MAX_SIZE


def _ensure_minimum_question_bank(
    question_bank: list[str],
    category: str,
    topic: str,
    initial_question: str,
    max_iterations: int,
) -> tuple[list[str], bool]:
    """Backfill question bank deterministically if below required minimum."""
    required_min = _required_bank_size(max_iterations, category)
    bank_max = _bank_max_size(category)
    if len(question_bank) >= required_min:
        return question_bank, False

    fallback_templates = _BANK_FALLBACK_BY_CATEGORY.get(category, _BANK_FALLBACK_BY_CATEGORY[TOPIC_TECHNICAL])
    topic_phrase = topic.strip() or "this topic"
    question_seed = initial_question.strip() or f"Can you walk me through your experience with {topic_phrase}?"

    expanded = list(question_bank)
    seen = {q.casefold() for q in expanded}

    candidate_fillers = [
        f"You mentioned {topic_phrase}. {question_seed}",
        *fallback_templates,
        f"What is one practical example from your work that best demonstrates your approach to {topic_phrase}?",
    ]

    for filler in candidate_fillers:
        cleaned = " ".join(filler.strip().split())
        key = cleaned.casefold()
        if not cleaned or key in seen:
            continue
        expanded.append(cleaned)
        seen.add(key)
        if len(expanded) >= required_min or len(expanded) >= bank_max:
            break

    return expanded[:bank_max], len(expanded) > len(question_bank)


def _sanitize_topic_source_for_mode(source: str, interview_mode: str) -> tuple[str, str | None]:
    """Enforce mode-compatible topic source and return (sanitized_source, repair_reason)."""
    normalized = (source or "").strip()
    repair_reason = None

    if interview_mode == INTERVIEW_MODE_RESUME_ONLY and normalized == "jd_requirement":
        repair_reason = "resume_only_disallowed_jd_requirement"
        return "resume_project", repair_reason

    if interview_mode == INTERVIEW_MODE_JD_ONLY and normalized in {"resume_claim", "resume_project"}:
        repair_reason = "jd_only_disallowed_resume_source"
        if normalized == "resume_claim":
            return "jd_requirement", repair_reason
        return "standard_technical", repair_reason

    if interview_mode == INTERVIEW_MODE_SKILLS_ONLY and normalized in {"resume_claim", "resume_project", "jd_requirement"}:
        repair_reason = "skills_only_disallowed_source"
        return "standard_technical", repair_reason

    return normalized or "standard_technical", repair_reason


def _ensure_coding_topic_in_plan(
    topics: list[dict],
    seniority: str,
    coding_language: str | None,
) -> list[dict]:
    """Inject a coding topic when the UI chose 'Technical' so question_node can open the sandbox.

    Without a TOPIC_CODING row, interviews stay verbal-only unless sandbox_guidance fires.
    """
    if any(t.get("category") == TOPIC_CODING for t in topics):
        return topics

    d = DEPTH_RULES[seniority]
    initial_question = (
        "Let's move to a coding exercise. Before you open the code editor, "
        "walk me through how you'd approach a typical algorithm problem — "
        "for example, finding all pairs in an array that sum to a target value."
    )
    _, bank = _normalize_question_bank(
        [
            "What's the time and space complexity of the approach you described? Can we do better than O(n²)?",
            "What edge cases should your solution handle — empty input, duplicates, no valid pairs?",
            "Go ahead and implement that in the code editor. Think out loud as you write.",
            "Walk me through your implementation — does it match what you described? Any changes you made?",
            "Can you optimize it further? What would you trade off if memory was the primary constraint?",
        ],
        TOPIC_CODING,
    )
    bank, _ = _ensure_minimum_question_bank(
        bank,
        TOPIC_CODING,
        "Coding exercise — algorithms",
        initial_question,
        max_iterations=TOPIC_CODING_MAX_ITERATIONS,
    )
    topic = {
        "id": str(uuid.uuid4()),
        "topic": "Coding exercise — algorithms & data structures",
        "category": TOPIC_CODING,
        "priority": PRIORITY_SHOULD_ASK,
        "source": "standard_technical",
        "resume_claim": None,
        "evidence_anchor": "Session: technical interview (product) — coding topic for sandbox",
        "initial_question": initial_question,
        "question_bank": bank,
        "max_iterations": TOPIC_CODING_MAX_ITERATIONS,
        "min_quality_to_advance": float(d["min_quality_to_advance"]),
        "requires_code": True,
        "coverage_status": COVERAGE_PENDING,
        "iterations_done": 0,
        "last_quality_score": None,
    }
    insert_at = min(2, len(topics))
    return topics[:insert_at] + [topic] + topics[insert_at:]


async def _parse_jd_structured(raw_jd: str, llm_helper: LLMHelper) -> dict:
    """Parse raw JD text into a compact structured dict with one LLM call.

    Returns a dict with keys: role_title, company, tech_stack, years_required,
    key_responsibilities, must_have_skills, nice_to_have_skills, seniority_signal.
    Returns {} on failure so caller falls back to raw JD truncation.
    """
    prompt = f"""Parse this job description into a compact JSON structure.

JOB DESCRIPTION:
{raw_jd[:3000]}

Return ONLY valid JSON (no markdown, no extra text):
{{
  "role_title": "exact job title from the JD",
  "company": "company name if mentioned, else empty string",
  "tech_stack": ["specific technologies, frameworks, tools — no soft skills"],
  "years_required": <integer or null if not specified>,
  "key_responsibilities": ["2-3 most important responsibilities, max 8 words each"],
  "must_have_skills": ["explicitly required skills only, max 6 items"],
  "nice_to_have_skills": ["preferred or bonus skills, max 4 items"],
  "seniority_signal": "junior|mid|senior|staff_principal"
}}

Rules:
- tech_stack: explicit technologies only (Python, FastAPI, Redis, Kubernetes — not 'communication skills')
- must_have_skills: only items explicitly stated as required or must-have
- seniority_signal: derive from job title and years_required field
- each list item max 8 words"""

    try:
        result_json = await llm_helper.call_llm_json(
            system_prompt=(
                "Parse job descriptions into compact JSON. "
                "Extract only what is explicitly stated — never invent requirements. "
                "Return only valid JSON."
            ),
            user_prompt=prompt,
            temperature=TEMPERATURE_ANALYTICAL,
        )
        data = json.loads(result_json)
        if not isinstance(data, dict) or "role_title" not in data:
            raise ValueError(
                f"Unexpected structure: {list(data.keys()) if isinstance(data, dict) else type(data)}"
            )
        logger.info(
            "[JD] Structured parse: role='%s' company='%s' tech=%d must_have=%d",
            data.get("role_title", "?"),
            data.get("company", ""),
            len(data.get("tech_stack", [])),
            len(data.get("must_have_skills", [])),
        )
        return data
    except Exception as e:
        logger.warning("[JD] Structured parsing failed (%s: %s) — using raw JD", type(e).__name__, e)
        return {}


def _format_structured_jd(jd_struct: dict) -> str:
    """Serialize a structured JD dict to a compact LLM-readable string (~80–120 tokens).

    Used inside generate_interview_plan() to replace the raw 600-char job_context
    with a denser, structured representation before passing to seniority estimation
    and topic generation.
    """
    if not jd_struct:
        return ""
    lines = []
    role = jd_struct.get("role_title", "")
    company = jd_struct.get("company", "")
    seniority = jd_struct.get("seniority_signal", "")
    years = jd_struct.get("years_required")

    header = role
    if company:
        header += f" at {company}"
    if seniority:
        header += f" ({seniority} level)"
    if years:
        header += f" | {years}+ years required"
    if header:
        lines.append(f"Role: {header}")
    if jd_struct.get("tech_stack"):
        lines.append(f"Tech: {', '.join(jd_struct['tech_stack'])}")
    if jd_struct.get("must_have_skills"):
        lines.append(f"Must-have: {', '.join(jd_struct['must_have_skills'])}")
    if jd_struct.get("nice_to_have_skills"):
        lines.append(f"Preferred: {', '.join(jd_struct['nice_to_have_skills'])}")
    if jd_struct.get("key_responsibilities"):
        lines.append(f"Responsibilities: {'; '.join(jd_struct['key_responsibilities'])}")
    return "\n".join(lines)


async def generate_interview_plan(state: dict, llm_helper: LLMHelper) -> dict:
    """Generate a structured interview plan from resume / job description / skills.

    Returns a dict (InterviewPlan) with:
        topics: list[TopicPlan]
        seniority_level: str
        expected_depth: str
        requires_coding: bool
        coding_language: str | None
        target_turns: int
        interview_style: str
        interview_mode: str
    """
    resume_context = build_resume_context(state)
    job_context = build_job_context(state)
    skills_context = build_skills_context(state)

    # ── Diagnostic: log what context builders returned ────────────────────────
    _has_resume = bool(resume_context and resume_context.strip()
                       and resume_context != "No resume details available."
                       and resume_context != "No resume context available.")
    _has_job = bool(job_context and job_context.strip()
                    and job_context != "Job Requirements:\n")
    _has_skills = bool(skills_context and skills_context.strip())

    forced = (state.get("forced_interview_mode") or "").strip().lower()
    if forced in INTERVIEW_MODES:
        interview_mode = forced
    else:
        interview_mode = _detect_interview_mode(_has_resume, _has_job, _has_skills)

    logger.info(
        "[PLAN] generate_interview_plan: interview=%s — "
        "interview_mode=%s (forced=%s) resume=%s jd=%s skills=%s user_type=%s",
        state.get("interview_id", "?"),
        interview_mode, bool(forced in INTERVIEW_MODES), _has_resume, _has_job, _has_skills,
        state.get("user_interview_type") or "—",
    )
    if not _has_resume and not _has_job and not _has_skills:
        logger.warning(
            "[PLAN] generate_interview_plan: interview=%s — "
            "no context available (no resume, no JD, no skills). Falling back to generic topics.",
            state.get("interview_id", "?"),
        )
    # ─────────────────────────────────────────────────────────────────────────

    difficulty_mode = state.get("difficulty_mode")
    interview_type = state.get("interview_type")
    skills = state.get("skills") or []
    difficulty = difficulty_mode or DIFFICULTY_MEDIUM

    # ── JD structured parsing (one-time) ────────────────────────────────────
    # Convert raw JD text (~600 chars / ~150 tok) → compact structured JSON
    # (~80-120 tok). The compact form is used by ALL downstream LLM calls this
    # session, including seniority estimation, topic generation, and action nodes.
    # One extra LLM call at plan time saves ~60-100 tokens on every subsequent call.
    job_description_structured: dict = {}
    raw_jd = state.get("job_description") or ""
    if raw_jd and interview_mode in (INTERVIEW_MODE_JD_AND_RESUME, INTERVIEW_MODE_JD_ONLY):
        job_description_structured = await _parse_jd_structured(raw_jd, llm_helper)
    if job_description_structured:
        job_context = "Job Requirements:\n" + _format_structured_jd(job_description_structured) + "\n\n"
    # ─────────────────────────────────────────────────────────────────────────

    # Seniority estimation only needs title + years-of-experience signals —
    # not the full resume narrative. Compact context saves ~4,000-8,000 tokens
    # per session (this call + _generate_topics both receive full context otherwise).
    _seniority_resume = resume_context[:500] if resume_context else ""
    _seniority_jd = job_context[:300] if job_context else ""
    seniority = await _estimate_seniority(
        _seniority_resume, _seniority_jd, skills_context, llm_helper, interview_mode)

    user_ut = (state.get("user_interview_type") or "").strip().lower()
    interview_type = (interview_type or user_ut).strip().lower()
    effective_type = interview_type or user_ut

    # Topic generation uses full context so questions are properly personalised
    topics = await _generate_topics(
        resume_context, job_context, skills_context, seniority, llm_helper,
        difficulty, interview_mode, interview_type=effective_type,
        difficulty_mode=difficulty_mode, skills=skills,
    )

    # Skill coverage guarantee: for skills_only mode, ensure every requested skill
    # has at least one topic. The LLM may omit a skill or rename it — this patch
    # adds a synthetic topic for any skill not found in the generated plan.
    if interview_mode == INTERVIEW_MODE_SKILLS_ONLY:
        user_skills_raw: list = state.get("user_skills") or []
        skills_list = [s.strip() for s in user_skills_raw if s.strip()]
        if skills_list:
            covered_skills: set[str] = set()
            for t in topics:
                topic_text = (t.get("topic", "") + " " + t.get("initial_question", "")).lower()
                for skill in skills_list:
                    if skill.lower() in topic_text:
                        covered_skills.add(skill.lower())

            _d = DEPTH_RULES.get(seniority, DEPTH_RULES[SENIORITY_MID])
            is_hr = effective_type == USER_INTERVIEW_HR
            for skill in skills_list:
                if skill.lower() not in covered_skills:
                    if is_hr:
                        missing_topic = {
                            "id": str(uuid.uuid4()),
                            "topic": f"{skill} — collaboration and ownership",
                            "category": TOPIC_BEHAVIORAL,
                            "priority": PRIORITY_SHOULD_ASK,
                            "source": "standard_behavioral",
                            "evidence_anchor": f"User-requested skill (HR lens): {skill}",
                            "initial_question": (
                                f"Tell me about a time you had to work with others while using {skill}. "
                                f"What was your role and how did you handle challenges?"
                            ),
                            "question_bank": [
                                f"How did you communicate progress or blockers to stakeholders while working on {skill}?",
                                f"What would you do differently if you faced a similar situation involving {skill}?",
                                f"How did working with {skill} help you grow as a teammate or professional?",
                            ],
                            "max_iterations": _d["behavioral_max_iterations"],
                            "min_quality_to_advance": _d["min_quality_to_advance"],
                            "requires_code": False,
                            "coverage_status": COVERAGE_PENDING,
                            "iterations_done": 0,
                            "last_quality_score": None,
                            "_topic_source": "skill_coverage_patch",
                        }
                    else:
                        missing_topic = {
                            "id": str(uuid.uuid4()),
                            "topic": f"{skill} — depth assessment",
                            "category": TOPIC_TECHNICAL,
                            "priority": PRIORITY_SHOULD_ASK,
                            "source": "standard_technical",
                            "evidence_anchor": f"User-requested skill: {skill}",
                            "initial_question": (
                                f"Walk me through your experience with {skill}. "
                                f"What have you built or worked on using it?"
                            ),
                            "question_bank": [
                                f"What are the key trade-offs or limitations you have encountered with {skill}?",
                                f"How does {skill} compare to alternatives you have used or considered?",
                                f"Give me a concrete example of a real problem you solved specifically using {skill}.",
                            ],
                            "max_iterations": _d["technical_max_iterations"],
                            "min_quality_to_advance": _d["min_quality_to_advance"],
                            "requires_code": False,
                            "coverage_status": COVERAGE_PENDING,
                            "iterations_done": 0,
                            "last_quality_score": None,
                            "_topic_source": "skill_coverage_patch",
                        }
                    # Insert before any coding topic so it lands in the body of the interview
                    coding_idx = next(
                        (i for i, t in enumerate(topics) if t.get("category") == TOPIC_CODING),
                        len(topics),
                    )
                    topics.insert(coding_idx, missing_topic)
                    logger.info("[PLAN] Skill coverage patch: added topic for missing skill '%s'", skill)

    # Derive plan-level metadata
    requires_coding = any(t["category"] == TOPIC_CODING for t in topics)
    coding_language = _detect_primary_language(resume_context, job_context)

    depth_rules = DEPTH_RULES[seniority]

    technical_count = sum(
        1 for t in topics
        if t["category"] in [TOPIC_TECHNICAL, TOPIC_CODING, TOPIC_PROJECT]
    )
    behavioral_count = sum(
        1 for t in topics
        if t["category"] in [TOPIC_BEHAVIORAL, TOPIC_SITUATIONAL]
    )
    if technical_count > behavioral_count * 1.5:
        interview_style = STYLE_TECHNICAL_HEAVY
    elif behavioral_count > technical_count * 1.5:
        interview_style = STYLE_BEHAVIORAL_HEAVY
    else:
        interview_style = STYLE_BALANCED

    # Product wizard: explicit interview type overrides inferred style
    if effective_type == USER_INTERVIEW_TECHNICAL:
        interview_style = STYLE_TECHNICAL_HEAVY
        topics = _ensure_coding_topic_in_plan(topics, seniority, coding_language)
        requires_coding = any(t["category"] == TOPIC_CODING for t in topics)
    elif effective_type == USER_INTERVIEW_HR:
        interview_style = STYLE_BEHAVIORAL_HEAVY
        # Strip any coding topics that may have slipped through — HR interviews never open the sandbox
        topics = [t for t in topics if t.get("category") != TOPIC_CODING]
        requires_coding = False
    elif effective_type == USER_INTERVIEW_BEHAVIORAL_TECHNICAL:
        interview_style = STYLE_BALANCED

    # Estimate total turns: each topic takes max_iterations turns on average
    must_ask_topics = [t for t in topics if t["priority"] <= PRIORITY_SHOULD_ASK]
    target_turns = sum(t["max_iterations"] for t in must_ask_topics)

    total_questions = state.get("total_questions")
    dynamic_questions = state.get("dynamic_questions", True)
    duration_minutes = state.get("duration_minutes")

    if total_questions and not dynamic_questions and len(topics) > total_questions:
        topics = topics[:total_questions]
        must_ask_topics = [t for t in topics if t["priority"] <= PRIORITY_SHOULD_ASK]
        target_turns = sum(t["max_iterations"] for t in must_ask_topics)

    if duration_minutes:
        target_turns = max(PLAN_MIN_TARGET_TURNS, min(int(duration_minutes / 3), PLAN_MAX_TARGET_TURNS))
    elif total_questions and not dynamic_questions:
        target_turns = max(PLAN_MIN_TARGET_TURNS, min(total_questions * 2, PLAN_MAX_TARGET_TURNS))
    else:
        target_turns = max(PLAN_MIN_TARGET_TURNS, min(target_turns, PLAN_MAX_TARGET_TURNS))

    plan = {
        "topics": topics,
        "seniority_level": seniority,
        "expected_depth": depth_rules["expected_depth"],
        "requires_coding": requires_coding,
        "coding_language": coding_language,
        "target_turns": target_turns,
        "interview_style": interview_style,
        "interview_mode": interview_mode,
        "plan_source": "llm",
        # Persisted with the plan so all action nodes can use the compact form.
        "job_description_structured": job_description_structured,
    }

    logger.info(
        "Interview plan generated: mode=%s seniority=%s topics=%d "
        "requires_coding=%s style=%s target_turns=%d",
        interview_mode, seniority, len(topics), requires_coding, interview_style, target_turns,
    )
    return plan


def generate_fast_fallback_plan(state: dict) -> dict:
    """Build a deterministic low-latency fallback plan.

    Used when full plan generation exceeds a strict latency budget on first turn.
    """
    resume_context = build_resume_context(state)
    job_context = build_job_context(state)

    _has_resume = bool(resume_context and resume_context.strip()
                       and resume_context != "No resume details available."
                       and resume_context != "No resume context available.")
    _has_job = bool(job_context and job_context.strip()
                    and job_context != "Job Requirements:\n")
    skills_context = build_skills_context(state)
    _has_skills = bool(skills_context and skills_context.strip())
    interview_mode = _detect_interview_mode(_has_resume, _has_job, _has_skills)

    seniority = SENIORITY_MID
    depth_rules = DEPTH_RULES[seniority]
    user_type = (
        (state.get("user_interview_type") or state.get("interview_type") or "")
        .strip()
        .lower()
    )
    topics = _fallback_hr_topics(seniority) if user_type == USER_INTERVIEW_HR else _fallback_topics(seniority)

    technical_count = sum(
        1 for t in topics
        if t["category"] in [TOPIC_TECHNICAL, TOPIC_CODING, TOPIC_PROJECT]
    )
    behavioral_count = sum(
        1 for t in topics
        if t["category"] in [TOPIC_BEHAVIORAL, TOPIC_SITUATIONAL]
    )
    if technical_count > behavioral_count * 1.5:
        interview_style = STYLE_TECHNICAL_HEAVY
    elif behavioral_count > technical_count * 1.5:
        interview_style = STYLE_BEHAVIORAL_HEAVY
    else:
        interview_style = STYLE_BALANCED

    if user_type == USER_INTERVIEW_HR:
        interview_style = STYLE_BEHAVIORAL_HEAVY
        topics = [t for t in topics if t.get("category") != TOPIC_CODING]
        requires_coding = False
    else:
        requires_coding = any(t["category"] == TOPIC_CODING for t in topics)
    coding_language = _detect_primary_language(resume_context, job_context)
    target_turns = max(PLAN_MIN_TARGET_TURNS, min(sum(t["max_iterations"] for t in topics), PLAN_MAX_TARGET_TURNS))

    return {
        "topics": topics,
        "seniority_level": seniority,
        "expected_depth": depth_rules["expected_depth"],
        "requires_coding": requires_coding,
        "coding_language": coding_language,
        "target_turns": target_turns,
        "interview_style": interview_style,
        "interview_mode": interview_mode,
        "plan_source": "fallback",  # deterministic fallback — not LLM-generated
    }


async def _estimate_seniority(
    resume_context: str,
    job_context: str,
    skills_context: str,
    llm_helper: LLMHelper,
    interview_mode: str = INTERVIEW_MODE_JD_AND_RESUME,
) -> str:
    """Estimate candidate seniority level.

    Uses only the context that is actually available (controlled by interview_mode)
    so the LLM never hallucinates a missing resume or JD.
    For skills_only mode, defaults to mid (no experience signals available).
    """
    if interview_mode == INTERVIEW_MODE_SKILLS_ONLY:
        # No resume or JD — cannot estimate seniority. Default to mid.
        # The difficulty mode selected by the user adjusts depth independently.
        logger.info("Seniority estimation: defaulting to mid (skills_only mode — no resume/JD)")
        return SENIORITY_MID

    if interview_mode == INTERVIEW_MODE_RESUME_ONLY:
        context_block = f"""RESUME:
{resume_context}

JOB DESCRIPTION: Not provided. Base your estimate entirely on the resume signals above.
Do NOT invent or assume a seniority title from a job description that does not exist."""

        evidence_note = (
            "Evidence to weigh (resume only — no JD available):\n"
            "1. Years of experience explicitly stated\n"
            "2. Progression of job titles\n"
            "3. Complexity and scope of described projects\n"
            "4. Leadership or mentorship signals\n"
            "Default to 'mid' if the resume is absent or ambiguous."
        )

    elif interview_mode == INTERVIEW_MODE_JD_ONLY:
        context_block = f"""RESUME: Not provided. Base your estimate on the job description signals below.
Do NOT invent resume details.

JOB DESCRIPTION:
{job_context}"""

        evidence_note = (
            "Evidence to weigh (JD only — no resume available):\n"
            "1. Required years of experience stated in the JD\n"
            "2. Seniority signals in the job title (senior, staff, principal, lead, etc.)\n"
            "3. Scope and complexity of responsibilities described\n"
            "Default to 'mid' if the JD is absent or ambiguous."
        )

    else:  # jd_and_resume — full context
        context_block = f"""RESUME:
{resume_context}

JOB DESCRIPTION:
{job_context}"""

        evidence_note = (
            "Evidence to weigh (in order of importance):\n"
            "1. Years of experience explicitly stated\n"
            "2. Progression of job titles\n"
            "3. Complexity and scope of described projects\n"
            "4. JD's required years of experience (if stated)\n"
            "5. Leadership or mentorship signals"
        )

    prompt = f"""You are assessing a Software Engineering / IT industry candidate. Estimate their engineering seniority level.

{context_block}

Seniority definitions for Software Engineering roles (IC = Individual Contributor):
- "junior": 0-2 years of professional SE experience. Titles: intern, associate engineer, junior developer, graduate SWE, SDE-I, IC1/IC2.
  Signals: limited independent scope, guided by senior engineers, basic CS fundamentals applied.
- "mid": 2-5 years. Titles: software engineer, developer, SDE-II, IC3.
  Signals: owns feature delivery end-to-end, familiar with production systems, some cross-team coordination.
- "senior": 5-9 years. Titles: senior software engineer, tech lead, SDE-III, IC4, engineering lead.
  Signals: drives architectural decisions for a team, mentors juniors, handles ambiguous technical problems independently.
- "staff_principal": 9+ years. Titles: staff engineer, principal engineer, distinguished engineer, architect, engineering manager, IC5/IC6, director of engineering.
  Signals: cross-org technical influence, defines technical strategy, owns large-scope systems, evaluates build-vs-buy.

{evidence_note}

Return JSON only: {{"seniority": "junior|mid|senior|staff_principal", "reasoning": "1-sentence justification citing specific evidence"}}"""

    try:
        result_json = await llm_helper.call_llm_json(
            system_prompt=(
                "You are an expert at evaluating Software Engineering seniority in the IT industry. "
                "Be evidence-based and precise. Use IC-level and title signals where available. "
                "Never hallucinate context that was not provided. "
                "Return only valid JSON."
            ),
            user_prompt=prompt,
            temperature=TEMPERATURE_ANALYTICAL,
        )
        result = json.loads(result_json)
        seniority = result.get("seniority", SENIORITY_MID)
        if seniority not in SENIORITY_LEVELS:
            seniority = SENIORITY_MID
        logger.info(
            "Seniority estimated: %s (mode=%s) — %s",
            seniority, interview_mode, result.get("reasoning", ""),
        )
        return seniority
    except Exception as e:
        logger.warning(f"Seniority estimation failed ({e}), defaulting to mid")
        return SENIORITY_MID


def _hr_skills_block(skills_context: str) -> str:
    """Optional skills section for HR prompts when skills are provided alongside resume/JD."""
    if not skills_context or not skills_context.strip():
        return ""
    return (
        f"{skills_context}\n\n"
        "Use these skills as HR dimensions: probe teamwork, communication, ownership, "
        "conflict, and growth in contexts where the candidate applied these competencies — "
        "not as technical deep-dives.\n\n"
    )


def _build_hr_topic_prompt(
    resume_context: str,
    job_context: str,
    skills_context: str,
    seniority: str,
    depth_rules: dict,
    difficulty: str,
    difficulty_instruction: str,
    interview_mode: str,
) -> str:
    """Build HR/behavioral interview topic prompt aligned with the Priya 6-stage structure.

    The LLM receives the full Priya HR_SYSTEM_PROMPT structure as context and generates
    stage-aligned topics. The actual question wording is picked and phrased by the LLM
    at question/followup generation time using HR_SYSTEM_PROMPT — nothing is hardcoded
    in the topic objects themselves.

    Stage mapping:
      Stage 1  Opening            → category: background,  priority: 1, max_iter: 1
      Stage 2  Background/Motiv.  → category: background,  priority: 1, max_iter: 2
      Stage 3  Behavioral (STAR)  → category: behavioral,  priority: 1, max_iter: 3
      Stage 4  Culture & Fit      → category: situational, priority: 1, max_iter: 1
      Stage 5  Logistics          → category: background,  priority: 1, max_iter: 1
      Stage 6  Close              → category: background,  priority: 1, max_iter: 1
    """
    beh_iter = depth_rules.get("behavioral_max_iterations", 2)
    min_quality = depth_rules["min_quality_to_advance"]

    hr_difficulty_note = {
        "easy": "Keep scenarios simple and relatable. Focus on fundamental interpersonal situations.",
        "medium": "Balance straightforward and nuanced scenarios. Probe for self-awareness.",
        "hard": "Use complex, multi-stakeholder scenarios. Probe for judgment under ambiguity and conflict resolution.",
    }.get(difficulty, "Balance straightforward and nuanced scenarios.")

    # Candidate context block — only include what is actually available
    skills_prefix = _hr_skills_block(skills_context)
    if interview_mode == INTERVIEW_MODE_SKILLS_ONLY:
        context_block = (
            f"{skills_context}\n\n"
            "RESUME: Not provided.\nJOB DESCRIPTION: Not provided."
        )
        personalization_note = (
            "Since no resume or JD is available, ground personalization in the listed skills above. "
            "Frame each skill through an HR lens, for example collaboration, communication, "
            "handling pressure, or learning agility."
        )
    elif interview_mode == INTERVIEW_MODE_RESUME_ONLY:
        context_block = (
            f"{skills_prefix}"
            f"RESUME:\n{resume_context}\n\n"
            "JOB DESCRIPTION: Not provided."
        )
        personalization_note = (
            "Anchor 2 to 3 topics to real projects or roles in the resume using a soft-skills lens "
            "(teamwork, conflict, accountability, influence). "
            "Personalize background and situational topics to the candidate's actual career trajectory."
        )
    elif interview_mode == INTERVIEW_MODE_JD_ONLY:
        context_block = (
            f"{skills_prefix}"
            f"RESUME: Not provided.\n\nJOB DESCRIPTION:\n{job_context}"
        )
        personalization_note = (
            "Align topic scenarios with the values and soft-skill requirements stated in the JD. "
            "Where skills are listed, map them to behavioral competency angles."
        )
    else:  # jd_and_resume (most common)
        context_block = (
            f"{skills_prefix}"
            f"RESUME:\n{resume_context}\n\n"
            f"JOB DESCRIPTION:\n{job_context}"
        )
        personalization_note = (
            "Use the JD to understand what soft skills and values this role requires. "
            "Use the resume to personalize each topic to the candidate's real experiences. "
            "Where skills are listed, shape questions around the candidate's demonstrated competencies."
        )

    # The Priya 6-stage structure is given to the LLM so it generates stage-aligned topics.
    # The LLM picks question wording from the stage banks at question-generation time (HR_SYSTEM_PROMPT).
    priya_stage_reference = f"""
━━━ PRIYA HR INTERVIEW STRUCTURE (generate topics aligned to these 6 stages) ━━━

You are creating the interview plan for Priya, an HR Business Partner who follows a structured
6-stage HR screening interview. Generate topics that map to the stages below IN ORDER.
The Priya system prompt (used during the actual interview) provides the full question banks.
Your job here is to create the stage structure so the orchestrator can route correctly.

STAGE 1, OPENING (1 topic, category=background, max_iterations=1):
  Opening topic where the candidate introduces themselves and motivations.
  Suggested opening angle: personal background and motivation for this opportunity.

STAGE 2, BACKGROUND AND MOTIVATION (1 topic, category=background, max_iterations=2):
  Explore the candidate's recent role, what prompted the search, and ideal next role.
  Personalize using resume context where available.

STAGE 3, BEHAVIORAL QUESTIONS (1 topic, category=behavioral, max_iterations={beh_iter}):
  STAR-format behavioral probing. Pick 3 to 4 dimensions from:
  tight deadlines, disagreement with a teammate or manager, proudest project,
  receiving critical feedback, rapid learning, cross-functional collaboration,
  project failure and recovery.
  Personalize to the candidate's background where possible.

STAGE 4, CULTURE AND FIT (1 topic, category=situational, max_iterations=1):
  Explore working style, preferred team environment, and workload management.

STAGE 5, LOGISTICS AND EXPECTATIONS (1 topic, category=background, max_iterations=1):
  Salary expectations, notice period, and other ongoing interviews.
  All 3 logistics questions must be covered in this single topic.

STAGE 6, CLOSE (1 topic, category=background, max_iterations=1):
  Candidate Q&A and warm closing. The interviewer asks if the candidate has questions
  (answers up to 2 generically), then closes with a warm 3 to 5 business day timeline.

DIFFICULTY: {difficulty.upper()} — {hr_difficulty_note}
CANDIDATE SENIORITY: {seniority}

━━━ TOPIC QUALITY RULES ━━━
- Generate exactly 6 topics, one per stage, in stage order.
- Each topic must have a distinct focus. No duplicates.
- initial_question: a brief, open-ended anchor that signals the stage theme.
  The actual wording will be naturally rephrased by Priya at interview time.
- question_bank: 2 to 3 follow-up probe angles for the stage (not full sentences,
  just thematic anchors the LLM will use to generate natural questions).
- NEVER invent resume details or JD requirements not present in the context.
- For Stage 3 behavioral topics: anchor to real experiences from the resume if available.

━━━ OUTPUT FORMAT ━━━

Return a JSON object with a single "topics" key:
{{
  "topics": [
    {{
      "topic": "descriptive stage topic name",
      "category": "background|behavioral|situational",
      "priority": 1,
      "source": "standard_behavioral",
      "evidence_anchor": "HR Priya Stage N or brief resume/JD pointer",
      "initial_question": "open-ended stage-appropriate opening anchor",
      "question_bank": ["probe angle 1", "probe angle 2"],
      "max_iterations": 1,
      "min_quality_to_advance": {min_quality},
      "requires_code": false
    }}
  ]
}}"""

    return (
        "You are an experienced HR interviewer designing a structured behavioral interview plan\n"
        "for Priya, a Senior HR Business Partner who follows the 6-stage Priya screening structure.\n"
        "This is NOT a technical interview. Focus on soft skills, culture fit, and growth mindset.\n\n"
        f"CANDIDATE CONTEXT:\n{context_block}\n\n"
        f"PERSONALIZATION GUIDANCE:\n{personalization_note}\n\n"
        f"{priya_stage_reference}"
    )


def _build_topic_prompt(
    resume_context: str,
    job_context: str,
    skills_context: str,
    seniority: str,
    depth_rules: dict,
    difficulty: str,
    interview_mode: str,
    interview_type: str = "",
) -> str:
    """Build the _generate_topics LLM prompt for the given interview mode and type.

    interview_type:
      - "technical" (default): technical deep-dives, project probes, coding assessment.
      - "hr": behavioral STAR, situational judgment, culture-fit, leadership — no coding.
      - "behavioral_technical" / "": balanced (technical default, no forced coding).

    Each mode uses a dedicated context block so the LLM never sees a missing JD or resume
    and hallucinates placeholder details.
    """
    difficulty_instruction = DIFFICULTY_TOPIC_INSTRUCTIONS.get(
        difficulty, DIFFICULTY_TOPIC_INSTRUCTIONS[DIFFICULTY_MEDIUM]
    )

    # ── HR interview type: completely different prompt ────────────────────────
    if interview_type == USER_INTERVIEW_HR:
        return _build_hr_topic_prompt(
            resume_context, job_context, skills_context,
            seniority, depth_rules, difficulty, difficulty_instruction, interview_mode,
        )

    # ── Shared rules (schema + constraints, appended to every mode prompt) ────
    shared_rules = f"""━━━ TOPIC CATEGORIES (use ONLY these three) ━━━

  "technical"  — Deep-dive into a specific technology, framework, concept, or SE principle.
                 Examples: "Redis cache eviction strategies and TTL design",
                           "PostgreSQL EXPLAIN ANALYZE and index selection",
                           "Async task queues: Celery vs. async workers trade-offs"
                 max_iterations = {depth_rules["technical_max_iterations"]}

  "project"    — Deep-dive into a specific system or project the candidate built or owned.
                 Probe architecture, scale, key decisions, failure modes, and lessons learned.
                 Examples: "Payment gateway integration and idempotency handling",
                           "Event-driven order processing pipeline on AWS SQS"
                 max_iterations = {depth_rules["technical_max_iterations"]}

  "coding"     — A concrete algorithmic or data structure problem solved live in the editor.
                 Must be a specific, well-scoped problem (not "write any sorting algorithm").
                 Calibrate complexity to seniority:
                   junior/mid   → arrays, hashmaps, strings, linked lists, binary search
                   senior       → graphs, trees, sliding window, two-pointer, DP
                   staff/principal → system-design-in-code, concurrency, advanced DP
                 max_iterations = {TOPIC_CODING_MAX_ITERATIONS}  (coding topics need room for the full arc)

  DO NOT use: "background", "behavioral", "situational" — these are excluded from SE technical interviews.

━━━ ITERATION + QUALITY RULES ━━━

  Quality threshold to advance topic: {depth_rules["min_quality_to_advance"]}
  Difficulty: {difficulty.upper()} — {difficulty_instruction}

━━━ TOPIC QUALITY RULES ━━━

  - Be SPECIFIC and TECHNICAL. Bad: "Python experience". Good: "Python asyncio event loop and task cancellation".
  - initial_question: open-ended, invites the candidate to explain their approach BEFORE coding.
  - Coding topics: question_bank MUST contain {QUESTION_BANK_MIN_SIZE_CODING}–{QUESTION_BANK_MAX_SIZE_CODING} questions that cover the FULL arc in order:
      1. Complexity probe ("What's the time/space complexity? Can we do better?")
      2. Edge case discussion ("How does your solution handle empty input, duplicates, no answer?")
      3. Invitation to code ("Go ahead and implement it in the code editor. Think out loud as you go.")
      4. Post-submission walkthrough ("Walk me through your implementation — any differences from your plan?")
      5. Optimization follow-up ("Can you optimize further? What trade-offs does that involve?")
    max_iterations must be {TOPIC_CODING_MAX_ITERATIONS}. Do NOT ask to code in initial_question.
  - All other topics: question_bank contains {QUESTION_BANK_MIN_SIZE}–{QUESTION_BANK_MAX_SIZE} DISTINCT sub-questions covering
    genuinely different technical angles. Each must stand alone as a self-contained probe.
  - EVIDENCE-ONLY RULE: ground every topic in explicit text from the provided context.
    NEVER invent technologies, projects, metrics, or requirements absent from the context.
    If context is sparse, use a generic-but-valid SE/IT topic instead of fabricating specifics.

━━━ OUTPUT FORMAT ━━━

Return a JSON object with a single "topics" key containing the array:
{{
  "topics": [
    {{
      "topic": "specific descriptive topic name",
      "category": "technical|project|coding",
      "priority": 1|2|3,
      "source": "resume_project|jd_requirement|standard_technical",
      "evidence_anchor": "short quote or pointer to where in the context this topic comes from",
      "initial_question": "opening question",
      "question_bank": ["sub-question 1", "sub-question 2", "sub-question 3"],
      "max_iterations": 1|2|3,
      "min_quality_to_advance": 0.0-1.0,
      "requires_code": true|false
    }}
  ]
}}"""

    header = (
        f"CANDIDATE SENIORITY : {seniority}\n"
        f"EXPECTED DEPTH      : {depth_rules['expected_depth']}\n"
        f"PROBE STYLE         : {depth_rules['probe_style']}"
    )

    # ── Mode: jd_and_resume ───────────────────────────────────────────────────
    if interview_mode == INTERVIEW_MODE_JD_AND_RESUME:
        topic_guidance = (
            f"Generate {PLAN_MIN_TOPICS}–{PLAN_MAX_TOPICS} topics.\n\n"
            f"HIERARCHY — follow this strictly:\n"
            f"  1. JOB DESCRIPTION is the PRIMARY driver. Every topic must map to a skill, technology,\n"
            f"     or capability that the role actually requires. Start from the JD and ask:\n"
            f"     'What do I need to assess to know this person can do this job?'\n"
            f"  2. RESUME is the SECONDARY source. Use it to:\n"
            f"     (a) Personalize technical questions — if JD needs distributed systems and the resume\n"
            f"         shows Kafka work, probe Kafka specifically rather than asking generically.\n"
            f"     (b) Select a relevant coding problem — match the candidate's stack to the JD.\n"
            f"  3. If the JD requires a skill the resume doesn't mention, still include that topic —\n"
            f"     probe adjacent knowledge or first-principles understanding of that domain.\n\n"
            f"Topic mix guidelines:\n"
            f"  • Technical deep-dives (2–4 topics, priority 1–2): one per major JD requirement.\n"
            f"  • Project/system deep-dives (1–2 topics): drill into a system from the resume most\n"
            f"    relevant to the JD.\n"
            f"  • Coding assessment — ALWAYS include exactly 1 coding topic (category=\"coding\",\n"
            f"    requires_code=true). Problem must reflect the JD's tech stack and seniority level.\n\n"
            f"For every topic include an evidence_anchor citing which JD requirement or resume text drove it."
        )

        return (
            "You are a senior Software Engineering interviewer at a top tech company conducting a technical interview.\n"
            "Your job is to design a rigorous interview plan that assesses whether this candidate can do THIS specific job.\n"
            "The Job Description tells you WHAT to assess. The resume tells you HOW to personalize each assessment.\n\n"
            f"JOB DESCRIPTION (primary — what the role requires):\n{job_context}\n\n"
            f"RESUME (secondary — personalization):\n{resume_context}\n\n"
            f"{header}\n\n"
            f"{topic_guidance}\n\n"
            f"{shared_rules}"
        )

    # ── Mode: resume_only ─────────────────────────────────────────────────────
    if interview_mode == INTERVIEW_MODE_RESUME_ONLY:
        topic_guidance = (
            f"Generate {PLAN_MIN_TOPICS}–{PLAN_MAX_TOPICS} topics anchored ENTIRELY to what is in the resume. "
            f"You decide the best mix — intelligently weighted to reveal genuine engineering depth.\n\n"
            f"Guidelines (not a rigid formula):\n"
            f"  • Project/system deep-dives — 1–3 topics drilling into systems the candidate built.\n"
            f"    Pick the most architecturally interesting or highest-scale projects from the resume.\n"
            f"    Probe decisions, trade-offs, failure handling, and what they'd do differently today.\n"
            f"  • Technical deep-dives — 1–3 topics on specific technologies claimed in the resume.\n"
            f"    Anchor each to an explicit technology, library, or concept mentioned in the resume.\n"
            f"  • Coding assessment — ALWAYS include exactly 1 coding topic (category=\"coding\",\n"
            f"    requires_code=true). Choose a problem appropriate to the candidate's tech stack\n"
            f"    (e.g. Python data structures if the resume shows Python) and their seniority.\n"
            f"\nIMPORTANT: No job description exists. Do NOT invent role requirements.\n"
            f"Every topic must reference something explicitly in the resume.\n"
            f"For each topic, include an evidence_anchor citing the specific resume evidence."
        )

        return (
            "You are a senior Software Engineering interviewer conducting a resume-based technical interview.\n"
            "No job description is available — design a rigorous plan based solely on the candidate's resume.\n\n"
            f"RESUME:\n{resume_context}\n\n"
            "JOB DESCRIPTION: Not provided. Do NOT invent role requirements.\n\n"
            f"{header}\n\n"
            f"{topic_guidance}\n\n"
            f"{shared_rules}"
        )

    # ── Mode: skills_only ────────────────────────────────────────────────────
    if interview_mode == INTERVIEW_MODE_SKILLS_ONLY:
        topic_guidance = (
            f"Generate {PLAN_MIN_TOPICS}–{PLAN_MAX_TOPICS} topics that systematically test each skill "
            f"in the provided list. Every topic must be anchored to one of these skills.\n\n"
            f"Guidelines:\n"
            f"  • Technical deep-dives — one topic per listed skill (priority 1–2).\n"
            f"    Go deep: probe implementation internals, edge cases, trade-offs, and failure modes.\n"
            f"    Do NOT ask surface-level questions — the candidate explicitly listed these skills.\n"
            f"  • Cross-skill topics — 1–2 topics that combine multiple listed skills in a realistic\n"
            f"    scenario (e.g. 'Python + Redis: async cache invalidation strategy').\n"
            f"  • Coding assessment — ALWAYS include exactly 1 coding topic (category=\"coding\",\n"
            f"    requires_code=true). Choose a problem that exercises one or more of the listed skills\n"
            f"    at the appropriate seniority level.\n"
            f"\nIMPORTANT: No resume or job description exists. Source must be \"standard_technical\" for all topics.\n"
            f"Every topic must test one of the explicitly listed skills — do not invent new skill areas."
        )

        return (
            "You are a senior Software Engineering interviewer conducting a skills-based technical interview.\n"
            "The candidate has provided a list of skills they want to be assessed on.\n"
            "Design a rigorous interview plan that tests each skill in depth.\n\n"
            f"{skills_context}\n\n"
            "RESUME: Not provided.\n"
            "JOB DESCRIPTION: Not provided.\n\n"
            f"{header}\n\n"
            f"{topic_guidance}\n\n"
            f"{shared_rules}"
        )

    # ── Mode: jd_only ─────────────────────────────────────────────────────────
    # (also the default/fallback for unrecognised modes)
    topic_guidance = (
        f"Generate {PLAN_MIN_TOPICS}–{PLAN_MAX_TOPICS} topics anchored ENTIRELY to the job description. "
        f"You decide the best mix — intelligently weighted to evaluate candidates for this specific SE role.\n\n"
        f"Guidelines (not a rigid formula):\n"
        f"  • Technical deep-dives — 3–5 topics on the most important technical skills required by the JD.\n"
        f"    source=\"jd_requirement\". Go deep: implementation details, architecture, trade-offs, failure modes.\n"
        f"    Prioritize skills that are hardest to fake and most critical to the role.\n"
        f"  • Project/system topics — 1–2 topics as open-ended system scenarios grounded in the JD's domain.\n"
        f"    e.g. if JD requires distributed systems experience, probe design of a specific component type.\n"
        f"  • Coding assessment — ALWAYS include exactly 1 coding topic (category=\"coding\",\n"
        f"    requires_code=true). Pick a problem representative of the role's daily engineering work,\n"
        f"    calibrated to the seniority level specified in the JD.\n"
        f"\nIMPORTANT: No candidate resume exists. Do NOT use source=\"resume_claim\" or \"resume_project\".\n"
        f"Every topic must be grounded in the job description. Include an evidence_anchor per topic."
    )

    return (
        "You are a senior Software Engineering interviewer conducting a JD-driven technical interview.\n"
        "No candidate resume is available — design a rigorous plan based solely on the job requirements.\n\n"
        "RESUME: Not provided. Do NOT invent candidate details.\n\n"
        f"JOB DESCRIPTION:\n{job_context}\n\n"
        f"{header}\n\n"
        f"{topic_guidance}\n\n"
        f"{shared_rules}"
    )


async def _generate_topics(
    resume_context: str,
    job_context: str,
    skills_context: str,
    seniority: str,
    llm_helper: LLMHelper,
    difficulty: str = DIFFICULTY_MEDIUM,
    interview_mode: str = INTERVIEW_MODE_JD_AND_RESUME,
    interview_type: str = "",
    difficulty_mode: str | None = None,
    skills: list[str] | None = None,
) -> list[dict]:
    """Generate an ordered, depth-calibrated topic plan for the interview."""
    # Must be set before any branch that references it (avoids UnboundLocalError in try/parse paths).
    is_hr = interview_type == USER_INTERVIEW_HR

    depth_rules = DEPTH_RULES[seniority]
    prompt = _build_topic_prompt(
        resume_context, job_context, skills_context, seniority, depth_rules,
        difficulty, interview_mode, interview_type,
    )
    # Inject session-level preference hints into the planning prompt.
    difficulty_note = ""
    if difficulty_mode:
        if is_hr:
            difficulty_note = (
                f"\nDIFFICULTY: {difficulty_mode.upper()} "
                "— calibrate scenario complexity and the depth of follow-up probes accordingly."
            )
        else:
            difficulty_note = (
                f"\nCODING DIFFICULTY REQUESTED BY CANDIDATE: {difficulty_mode.upper()} "
                "— calibrate coding and technical questions to this level."
            )

    skills_note = ""
    if skills:
        if is_hr:
            skills_note = (
                f"\nSKILLS TO FOCUS ON: {', '.join(skills)} "
                "— weave these into HR/behavioral topics (teamwork, communication, ownership, "
                "conflict resolution, growth) rather than technical drills."
            )
        else:
            skills_note = (
                f"\nSKILLS TO FOCUS ON: {', '.join(skills)} "
                "— prioritize these skills in technical coverage."
            )

    type_note = ""
    if interview_type == "hr":
        type_note = (
            "\nINTERVIEW TYPE: HR / behavioral — skip coding assessment and "
            "focus on behavioral/situational depth."
        )
    elif interview_type == "technical":
        type_note = (
            "\nINTERVIEW TYPE: Technical — include a coding assessment when appropriate "
            "and emphasize deep technical topics."
        )

    if difficulty_note or skills_note or type_note:
        prompt = f"{prompt}\n{difficulty_note}{skills_note}{type_note}"

    try:
        system_prompt = (
            "You are an experienced HR interviewer creating a structured behavioral interview plan. "
            "All topics must probe soft skills, interpersonal effectiveness, culture fit, and growth mindset. "
            "Do NOT include any technical or coding topics. "
            "Return ONLY valid JSON with a 'topics' key, no markdown, no extra text."
            if is_hr else
            "You are an expert Software Engineering interviewer creating a structured technical interview plan "
            "for the IT industry. All topics must probe SE/IT competencies: programming, system design, "
            "databases, cloud, algorithms, APIs, DevOps, or engineering processes. "
            "Return ONLY valid JSON with a 'topics' key, no markdown, no extra text."
        )
        result_json = await llm_helper.call_llm_json(
            system_prompt=system_prompt,
            user_prompt=prompt,
            temperature=TEMPERATURE_BALANCED,
        )

        raw = json.loads(result_json)
        # Unwrap {"topics": [...]} or any other dict wrapper the model may use.
        if isinstance(raw, dict):
            # Try known keys first, then scan for the first list value.
            raw = (
                raw.get("topics")
                or raw.get("topic_plan")
                or raw.get("interview_plan")
                or raw.get("plan")
                or raw.get("result")
                or next((v for v in raw.values() if isinstance(v, list)), [])
            )
        logger.info("[PLAN] _generate_topics: raw topic count from LLM = %d", len(raw) if isinstance(raw, list) else 0)

        # Category remap rules differ by interview type.
        # HR interviews: allow behavioral/situational/background; remap technical → behavioral.
        # Technical interviews (default): remap behavioral → project, situational → technical.
        if is_hr:
            _EXCLUDED_CATEGORY_REMAP = {
                TOPIC_TECHNICAL: TOPIC_BEHAVIORAL,  # technical question → behavioral probe
                TOPIC_CODING:    TOPIC_SITUATIONAL, # coding topic → situational judgment
            }
        else:
            _EXCLUDED_CATEGORY_REMAP = {
                TOPIC_BACKGROUND:  TOPIC_TECHNICAL,   # career warmup → technical probe
                TOPIC_BEHAVIORAL:  TOPIC_PROJECT,      # STAR question → project deep-dive
                TOPIC_SITUATIONAL: TOPIC_TECHNICAL,    # hypothetical → technical scenario
            }

        topics: list[dict] = []
        for item in raw:
            repair_actions: list[str] = []
            _default_category = TOPIC_BEHAVIORAL if is_hr else TOPIC_TECHNICAL
            category = item.get("category", _default_category)
            if category not in TOPIC_CATEGORIES:
                category = _default_category
                repair_actions.append(f"invalid_category_defaulted_{_default_category}")
            elif category in _EXCLUDED_CATEGORY_REMAP:
                remapped = _EXCLUDED_CATEGORY_REMAP[category]
                repair_actions.append(f"excluded_category_{category}_remapped_to_{remapped}")
                category = remapped

            raw_category = item.get("category", TOPIC_TECHNICAL)
            _iter_cap = TOPIC_CODING_MAX_ITERATIONS if raw_category == TOPIC_CODING else 5
            max_iterations = min(int(item.get("max_iterations", 2)), _iter_cap)
            topic_name = str(item.get("topic", "General background"))
            initial_question = str(item.get("initial_question", ""))
            evidence_anchor = str(item.get("evidence_anchor", "")).strip()

            raw_bank_count, question_bank = _normalize_question_bank(item.get("question_bank", []), category)
            cleaned_bank_count = len(question_bank)
            question_bank, bank_was_autofilled = _ensure_minimum_question_bank(
                question_bank=question_bank,
                category=category,
                topic=topic_name,
                initial_question=initial_question,
                max_iterations=max_iterations,
            )
            required_min_bank = _required_bank_size(max_iterations)
            logger.info(
                "Topic bank normalized: topic='%s' category=%s max_iter=%d raw=%d clean=%d final=%d required_min=%d autofilled=%s",
                topic_name,
                category,
                max_iterations,
                raw_bank_count,
                cleaned_bank_count,
                len(question_bank),
                required_min_bank,
                bank_was_autofilled,
            )

            raw_source = str(item.get("source", "standard_technical"))
            source, source_repair = _sanitize_topic_source_for_mode(raw_source, interview_mode)
            if source_repair:
                repair_actions.append(source_repair)

            # Enforce mode-safe source fallback for unknown or excluded values.
            if is_hr:
                allowed_sources = {"resume_project", "standard_behavioral"}
                _default_source = "standard_behavioral"
            else:
                allowed_sources = {"resume_project", "jd_requirement", "standard_technical"}
                _default_source = "standard_technical"
            if source not in allowed_sources:
                repair_actions.append(f"unknown_or_excluded_source_defaulted_{_default_source}")
                source = _default_source

            # Evidence anchor is optional, but we record missing anchors for telemetry.
            if not evidence_anchor:
                repair_actions.append("missing_evidence_anchor")

            topic: dict = {
                "id": str(uuid.uuid4()),
                "topic": topic_name,
                "category": category,
                "priority": int(item.get("priority", PRIORITY_SHOULD_ASK)),
                "source": source,
                "evidence_anchor": evidence_anchor or None,
                "initial_question": initial_question,
                "question_bank": question_bank,
                "max_iterations": max_iterations,
                "min_quality_to_advance": float(
                    max(min(item.get("min_quality_to_advance", depth_rules["min_quality_to_advance"]), 1.0), 0.0)
                ),
                "requires_code": bool(item.get("requires_code", False)),
                # Runtime tracking fields
                "coverage_status": COVERAGE_PENDING,
                "iterations_done": 0,
                "last_quality_score": None,
                "_topic_source": "llm",  # generated by LLM
            }
            topics.append(topic)

            logger.info(
                "Topic accepted: topic='%s' mode=%s source=%s repairs=%s evidence_anchor=%s",
                topic_name,
                interview_mode,
                source,
                repair_actions or ["none"],
                bool(evidence_anchor),
            )

        # Sort by priority first, then by category order within each priority group.
        # Category order creates a natural interview arc:
        #   background → behavioral → technical → project → situational → coding
        # Coding always last so the candidate warms up before live code assessment.
        _CATEGORY_SORT_ORDER = {
            TOPIC_BACKGROUND:  0,
            TOPIC_BEHAVIORAL:  1,
            TOPIC_TECHNICAL:   2,
            TOPIC_PROJECT:     3,
            TOPIC_SITUATIONAL: 4,
            TOPIC_CODING:      5,
        }
        topics.sort(key=lambda x: (
            x["priority"],
            _CATEGORY_SORT_ORDER.get(x.get("category", TOPIC_TECHNICAL), 3),
        ))

        if topics:
            return topics

    except Exception as e:
        logger.warning("Topic generation failed (%s: %s), using fallback topics", type(e).__name__, e, exc_info=True)

    if is_hr:
        return _fallback_hr_topics(seniority)
    return _fallback_topics(seniority)


def _fallback_hr_topics(seniority: str) -> list[dict]:
    """Minimal Priya-structured HR fallback topic list when LLM generation fails.

    Returns 6 stage-skeleton topics aligned with the Priya 6-stage HR interview structure.
    The topics contain only stage names, categories, and brief thematic anchors —
    actual question wording is generated by the LLM at interview time using HR_SYSTEM_PROMPT.
    This avoids hardcoded questions while still providing the orchestrator with the
    stage order it needs to route correctly.
    """
    d = DEPTH_RULES[seniority]
    beh_iter = d.get("behavioral_max_iterations", 2)
    q = d["min_quality_to_advance"]

    def _stage_topic(
        topic: str,
        category: str,
        stage_id: str,
        initial_question: str,
        bank: list,
        max_iter: int,
    ) -> dict:
        """Helper: build a stage-skeleton topic dict."""
        return {
            "id": str(uuid.uuid4()),
            "topic": topic,
            "category": category,
            "priority": PRIORITY_MUST_ASK,
            "source": "standard_behavioral",
            "evidence_anchor": f"HR Priya {stage_id}",
            "resume_claim": None,
            "initial_question": initial_question,
            "question_bank": bank,
            "max_iterations": max_iter,
            "min_quality_to_advance": q,
            "requires_code": False,
            "coverage_status": COVERAGE_PENDING,
            "iterations_done": 0,
            "last_quality_score": None,
            "_topic_source": "fallback",
            "_hr_stage": stage_id,
        }

    return [
        # Stage 1 — Opening
        _stage_topic(
            topic="Opening — candidate introduction and motivation",
            category=TOPIC_BACKGROUND,
            stage_id=HR_STAGE_OPENING,
            initial_question="Tell me about yourself and what drew you to this opportunity.",
            bank=[
                "Background and career path summary",
                "Motivation for pursuing this role specifically",
            ],
            max_iter=1,
        ),
        # Stage 2 — Background & Motivation
        _stage_topic(
            topic="Background and motivation — recent role and ideal next step",
            category=TOPIC_BACKGROUND,
            stage_id=HR_STAGE_BACKGROUND,
            initial_question="Walk me through your most recent role and the kind of work you were doing day-to-day.",
            bank=[
                "What prompted the job search",
                "Ideal next role in terms of team, scope, and growth",
            ],
            max_iter=2,
        ),
        # Stage 3 — Behavioral (STAR)
        _stage_topic(
            topic="Behavioral questions — STAR format past experiences",
            category=TOPIC_BEHAVIORAL,
            stage_id=HR_STAGE_BEHAVIORAL,
            initial_question="Tell me about a time you had to work under a tight deadline. How did you prioritize?",
            bank=[
                "Disagreement with a teammate or manager and resolution",
                "Project most proud of and specific contribution",
                "Critical feedback received and response",
                "Rapid learning to complete a task",
                "Cross-functional collaboration experience",
                "Project that did not go as planned and recovery",
            ],
            max_iter=beh_iter,
        ),
        # Stage 4 — Culture & Fit
        _stage_topic(
            topic="Culture and fit — working style and environment",
            category=TOPIC_SITUATIONAL,
            stage_id=HR_STAGE_CULTURE,
            initial_question="How would your current teammates describe your working style?",
            bank=[
                "Team environment that brings out best work",
                "Managing workload when multiple priorities compete",
            ],
            max_iter=1,
        ),
        # Stage 5 — Logistics
        _stage_topic(
            topic="Logistics and expectations — salary, availability, and process",
            category=TOPIC_BACKGROUND,
            stage_id=HR_STAGE_LOGISTICS,
            initial_question="What are your salary expectations for this role?",
            bank=[
                "Current notice period or earliest joining date",
                "Other ongoing interviews and current stage in those processes",
            ],
            max_iter=1,
        ),
        # Stage 6 — Close
        _stage_topic(
            topic="Candidate questions and warm close",
            category=TOPIC_BACKGROUND,
            stage_id=HR_STAGE_CLOSE,
            initial_question="That covers everything on my end. Do you have any questions about the role or the company?",
            bank=[
                "Answer candidate questions generically, up to 2",
                "Warm closing with 3 to 5 business day timeline",
            ],
            max_iter=1,
        ),
    ]


def _fallback_topics(seniority: str) -> list[dict]:
    """Minimal fallback topic list when LLM generation fails.

    Topics are anchored to Software Engineering / IT industry domain.
    Covers: SE background, backend system design, engineering behavioral, and coding assessment.
    """
    d = DEPTH_RULES[seniority]
    return [
        {
            "id": str(uuid.uuid4()),
            "topic": "Software Engineering background and technical focus",
            "category": TOPIC_TECHNICAL,
            "priority": PRIORITY_MUST_ASK,
            "source": "standard_technical",
            "evidence_anchor": None,
            "resume_claim": None,
            "initial_question": "Walk me through your engineering background — what types of systems have you built and what's your primary tech stack?",
            "question_bank": [
                "What kind of backend or infrastructure problems are you most drawn to solving and why?",
                "How do you stay current with evolving technologies in the software engineering space?",
                "What's the largest production system you've worked on in terms of scale or team size?",
            ],
            "max_iterations": 1,
            "min_quality_to_advance": d["min_quality_to_advance"],
            "requires_code": False,
            "coverage_status": COVERAGE_PENDING,
            "iterations_done": 0,
            "last_quality_score": None,
            "_topic_source": "fallback",
        },
        {
            "id": str(uuid.uuid4()),
            "topic": "Most technically challenging system or project",
            "category": TOPIC_PROJECT,
            "priority": PRIORITY_MUST_ASK,
            "source": "standard_technical",
            "evidence_anchor": None,
            "resume_claim": None,
            "initial_question": "Walk me through the most technically challenging system or project you've worked on — what made it hard and how did you approach it?",
            "question_bank": [
                "What were the key architectural decisions and what trade-offs did you weigh?",
                "How did you handle scalability, reliability, or performance constraints?",
                "What went wrong during development or in production, and how did you debug and resolve it?",
                "If you were rebuilding this system today, what would you design differently?",
            ],
            "max_iterations": d["technical_max_iterations"],
            "min_quality_to_advance": d["min_quality_to_advance"],
            "requires_code": False,
            "coverage_status": COVERAGE_PENDING,
            "iterations_done": 0,
            "last_quality_score": None,
            "_topic_source": "fallback",
        },
        {
            "id": str(uuid.uuid4()),
            "topic": "Handling a production incident or critical technical failure",
            "category": TOPIC_PROJECT,
            "priority": PRIORITY_MUST_ASK,
            "source": "standard_technical",
            "evidence_anchor": None,
            "resume_claim": None,
            "initial_question": "Tell me about a significant production incident or critical technical failure you dealt with — what happened and how did you handle it?",
            "question_bank": [
                "How did you diagnose and isolate the root cause under time pressure?",
                "What immediate actions did you take and how did you communicate with stakeholders?",
                "What process or engineering changes did you make afterward to prevent recurrence?",
            ],
            "max_iterations": d["technical_max_iterations"],
            "min_quality_to_advance": d["min_quality_to_advance"],
            "requires_code": False,
            "coverage_status": COVERAGE_PENDING,
            "iterations_done": 0,
            "last_quality_score": None,
            "_topic_source": "fallback",
        },
        {
            "id": str(uuid.uuid4()),
            "topic": "Coding: Arrays and hash maps — find pairs summing to a target",
            "category": TOPIC_CODING,
            "priority": PRIORITY_MUST_ASK,
            "source": "standard_technical",
            "evidence_anchor": None,
            "resume_claim": None,
            "initial_question": "Let's do a coding exercise. Given an array of integers and a target value, return all unique pairs that sum to the target. Before you open the code editor, walk me through how you'd approach this problem.",
            "question_bank": [
                "Good. What's the time and space complexity of the approach you described? And is there a way to get better than O(n²)?",
                "What edge cases should your solution handle — empty array, duplicates, negative numbers, no pairs found?",
                "Great thinking. Go ahead and implement that in the code editor — take your time, think it through as you go.",
                "Now that you've coded it, walk me through your implementation. Does it match the approach you described earlier?",
                "Can you optimize this further? What would you change if memory usage was the main constraint?",
            ],
            "max_iterations": TOPIC_CODING_MAX_ITERATIONS,
            "min_quality_to_advance": d["min_quality_to_advance"],
            "requires_code": True,
            "coverage_status": COVERAGE_PENDING,
            "iterations_done": 0,
            "last_quality_score": None,
            "_topic_source": "fallback",
        },
    ]


def _detect_primary_language(resume_context: str, job_context: str) -> Optional[str]:
    """Detect the primary programming language from resume and job description text."""
    text = (resume_context + " " + job_context).lower()

    # Only languages supported by the sandbox executor
    language_signals: dict[str, list[str]] = {
        "python": ["python", "django", "flask", "fastapi", "pandas", "pytorch", "tensorflow"],
        "javascript": ["javascript", "typescript", "node.js", "nodejs", "react", "vue", "angular", "next.js"],
        "java": ["java", "spring", "maven", "gradle", "kotlin", "jvm"],
        "cpp": ["c++", "cpp", "c plus plus", "stl", "boost"],
    }

    scores: dict[str, int] = {}
    for lang, signals in language_signals.items():
        scores[lang] = sum(text.count(s) for s in signals)

    if not any(scores.values()):
        return None

    return max(scores, key=lambda k: scores[k])


def get_next_pending_topic(plan: dict) -> Optional[dict]:
    """Return the next topic that hasn't been adequately covered yet."""
    topics = plan.get("topics", [])
    for topic in topics:
        if topic.get("coverage_status") in (COVERAGE_PENDING, COVERAGE_IN_PROGRESS):
            return topic
    return None


def get_topic_by_id(plan: dict, topic_id: str) -> Optional[dict]:
    """Find a topic in the plan by its ID."""
    for topic in plan.get("topics", []):
        if topic.get("id") == topic_id:
            return topic
    return None


def update_topic_in_plan(plan: dict, topic_id: str, updates: dict) -> dict:
    """Return a new plan dict with the specified topic updated (no mutation)."""
    new_topics = []
    for topic in plan.get("topics", []):
        if topic.get("id") == topic_id:
            new_topics.append({**topic, **updates})
        else:
            new_topics.append(topic)
    return {**plan, "topics": new_topics}