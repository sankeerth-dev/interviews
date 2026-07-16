"""Action nodes for interview orchestrator.

This module contains all action nodes: greeting, question, followup,
sandbox_guidance, code_review, evaluation, and closing.
"""

import logging
import json
import re
import uuid
import time
from typing import TYPE_CHECKING
from datetime import datetime
from openai import AsyncOpenAI

from src.services.execution.sandbox_service import SandboxService, Language as SandboxLanguage
from src.services.analysis.code_metrics import get_code_metrics
from src.services.orchestrator.types import InterviewState, QuestionRecord, QuestionGeneration
from src.services.orchestrator.context_builders import (
    build_resume_context, build_conversation_context, build_job_context,
    build_covered_topics_line,
)
from src.services.orchestrator.constants import (
    COMMON_SYSTEM_PROMPT, HR_SYSTEM_PROMPT, TERMINATION_MESSAGE,
    SANDBOX_POLL_INTERVAL_SECONDS, SANDBOX_STUCK_THRESHOLD_SECONDS,
    TEMPERATURE_CREATIVE, TEMPERATURE_BALANCED, TEMPERATURE_ANALYTICAL, TEMPERATURE_QUESTION,
    DEFAULT_MODEL,
    DEPTH_RULES, SENIORITY_MID, COVERAGE_IN_PROGRESS, COVERAGE_ADEQUATE,
    DIFFICULTY_MEDIUM, DIFFICULTY_PROBE_STYLE_SUFFIX, DIFFICULTY_QUESTION_FRAMING,
    TOPIC_CODING,
    USER_INTERVIEW_HR,
)
from src.services.orchestrator.plan_generator import (
    get_next_pending_topic, get_topic_by_id, update_topic_in_plan
)
logger = logging.getLogger(__name__)


def _is_idk_response(text: str) -> bool:
    """Return True if the candidate's response signals they don't know the answer.

    Matches common "I don't know" phrasings broadly — not just exact strings.
    Used to trigger the IDK offer flow instead of blindly rephrasing the same question.
    """
    if not text:
        return False
    t = text.lower().strip()
    idk_phrases = [
        "i don't know", "i do not know", "dont know", "don't know",
        "not sure", "no idea", "i have no idea", "i'm not sure",
        "i am not sure", "i have no clue", "no clue",
        "i can't answer", "i cannot answer",
        "i'm lost", "i am lost", "i'm stuck", "i am stuck",
        "i don't recall", "i don't remember",
        "i have no knowledge",
    ]
    return any(phrase in t for phrase in idk_phrases)


def _validate_code_syntax(code: str, language: str) -> str | None:
    """Return an error message if the code has obvious syntax problems, else None.

    Only validates languages where static syntax checking is cheap and reliable.
    Returns None (no error) for unsupported languages so execution can proceed.
    """
    if not code or not code.strip():
        return "No code submitted — please write your solution and try again."
    if len(code.strip()) < 15:
        return "Submission is too short to be a valid solution. Please write your full implementation."

    if language in ("python", "py"):
        import ast as _ast
        try:
            _ast.parse(code)
        except SyntaxError as e:
            return f"Python syntax error on line {e.lineno}: {e.msg}"

    return None


def _get_starter_template(language: str) -> str:
    """Return a minimal language-appropriate starter template for the code sandbox."""
    if language in ("javascript", "js"):
        return "// Write your solution here\n\nfunction solution(input) {\n  // TODO: implement your solution\n}\n"
    if language in ("java",):
        return "// Write your solution here\n\npublic class Solution {\n    public static void main(String[] args) {\n        // TODO: implement your solution\n    }\n}\n"
    if language in ("cpp", "c++"):
        return "// Write your solution here\n#include <iostream>\nusing namespace std;\n\nint main() {\n    // TODO: implement your solution\n    return 0;\n}\n"
    # Default: Python
    return "# Write your solution here\n\ndef solution(input_data):\n    # TODO: implement your solution\n    pass\n"


class ActionNodeMixin:
    """Mixin containing all action node methods."""

    def _derive_tone_adjustment(self, state: InterviewState) -> str:
        """Get effective tone adjustment from state, defaulting to balanced."""
        tone = state.get("tone_adjustment")
        if tone in {"warm", "balanced", "rigorous"}:
            return tone
        return "balanced"

    def _tone_instruction(self, tone: str) -> str:
        """Map effective tone to prompt-safe behavioral guidance."""
        if tone == "warm":
            return "Use a warm, encouraging, and collaborative tone while staying technically grounded."
        if tone == "rigorous":
            return "Use a professional but rigorous screening tone. Be concise and probing."
        return "Use a balanced professional tone: supportive but appropriately evaluative."

    def _is_interview_scope_question(self, text: str) -> bool:
        """Return True for candidate questions related to role/company/interview."""
        q = (text or "").lower()
        if not q:
            return False
        scope_terms = [
            "role", "position", "team", "company", "culture", "interview", "round",
            "process", "tech stack", "technology", "remote", "onsite", "timeline",
            "responsibilities", "expectation", "hiring", "job", "project",
        ]
        return any(term in q for term in scope_terms)

    _STOP_WORDS: frozenset[str] = frozenset({
        "the", "a", "an", "is", "are", "was", "were", "how", "what", "why",
        "when", "where", "who", "which", "can", "could", "would", "should",
        "do", "does", "did", "have", "has", "had", "will", "i", "you", "your",
        "my", "me", "we", "us", "our", "they", "them", "their", "this", "that",
        "these", "those", "in", "on", "at", "to", "for", "with", "about", "by",
        "from", "up", "down", "it", "its", "be", "been", "being", "and", "or",
        "but", "if", "of", "as", "into", "through", "during", "before", "after",
        "tell", "me", "give", "walk", "describe", "explain", "please",
    })

    def _keywords(self, text: str) -> frozenset[str]:
        """Extract meaningful keywords from question text."""
        return frozenset(
            w for w in text.lower().split()
            if len(w) > 3 and w.strip("?.,:;!") not in self._STOP_WORDS
        )

    def _is_duplicate_question(
        self,
        question_text: str,
        state: InterviewState,
        similarity_threshold: float = 0.62,
    ) -> bool:
        """Return True if the question is too similar to a recently asked question.

        Checks both exact text match (full history) and keyword Jaccard similarity
        against the last 8 questions (to catch paraphrased repeats).
        """
        questions_asked = state.get("questions_asked", [])
        normalized = question_text.lower().strip()

        # Exact match — full history
        for q in questions_asked:
            if q["text"].lower().strip() == normalized:
                return True

        # Keyword overlap — last 8 questions only (avoids false positives over long sessions)
        new_kw = self._keywords(question_text)
        if not new_kw:
            return False

        for q in questions_asked[-8:]:
            prev_kw = self._keywords(q["text"])
            if not prev_kw:
                continue
            intersection = new_kw & prev_kw
            union = new_kw | prev_kw
            if union and len(intersection) / len(union) >= similarity_threshold:
                logger.debug(
                    "Duplicate question suppressed (%.0f%% overlap): '%s' ≈ '%s'",
                    100 * len(intersection) / len(union),
                    question_text[:60],
                    q["text"][:60],
                )
                return True

        return False

    async def greeting_node(self, state: InterviewState) -> InterviewState:
        """Generate personalized greeting.

        Returns partial state update. Prevents duplicate greetings on reconnects.
        """
        # Check if greeting was already shown (state-based check)
        last_node = state.get("last_node", "")
        conv_history = state.get("conversation_history", [])

        # If greeting node was already executed, return existing greeting from history
        if last_node == "greeting":
            for msg in reversed(conv_history):
                if msg.get("role") == "assistant" and msg.get("content"):
                    return {
                        "last_node": "greeting",
                        "phase": "intro",
                        "next_message": msg.get("content"),
                    }

        resume_context = build_resume_context(state)
        job_context = build_job_context(state)
        candidate_name = state.get("candidate_name")
        is_hr_interview = (
            (state.get("user_interview_type") or "").strip().lower() == USER_INTERVIEW_HR
        )

        if is_hr_interview:
            # For HR interviews, always use the Priya persona — no LLM call needed.
            # The interviewer name is Priya, as defined in HR_SYSTEM_PROMPT.
            interviewer_name = "Priya"
            company_name = "our company"
            interviewer_role = "Senior HR Business Partner"
        else:
            # Generate interviewer persona from job description (original behaviour)
            persona_prompt = f"""Based on this job description, create a realistic interviewer persona.

            Job Description:
            {job_context if job_context else "No specific job description provided."}

            Generate:
            1. A female name for the interviewer (e.g., Sarah, Emma, Maya, Priya, etc.)
            2. The company name (extract from job description or use a realistic tech company name)
            3. The interviewer's role/title (e.g., Senior Engineering Manager, Tech Lead, etc.)

            Return ONLY a JSON object with: "name", "company", "role"
            Example: {{"name": "Sarah", "company": "TechCorp", "role": "Senior Engineering Manager"}}"""

            try:
                persona_json = await self.llm_helper.call_llm_json(
                    system_prompt="You are generating a realistic interviewer persona. Return only valid JSON.",
                    user_prompt=persona_prompt,
                    temperature=TEMPERATURE_BALANCED,
                )
                persona = json.loads(persona_json)
                interviewer_name = persona.get("name", "Sarah")
                company_name = persona.get("company", "our company")
                interviewer_role = persona.get("role", "Engineering Manager")
            except Exception as e:
                logger.warning(f"Failed to generate persona, using defaults: {e}")
                interviewer_name = "Sarah"
                company_name = "our company"
                interviewer_role = "Engineering Manager"

        name_context = ""
        if candidate_name:
            first_name = candidate_name.split()[0] if candidate_name else None
            if first_name:
                name_context = f"\nCandidate's name: {first_name}\nYou can refer to them by their first name ({first_name}) to make it more personal and friendly."

        # ── Build prompt based on interview type ──────────────────────────────
        if is_hr_interview:
            # HR: Priya persona greeting that ENDS with Stage 1 question.
            # Keeping the opening question inside the greeting avoids a double-ask
            # from question_node. The Opening topic is marked covered below.
            first_name = (candidate_name or "").split()[0] if candidate_name else None
            name_part = first_name if first_name else "there"
            prompt = f"""You are Priya, a Senior HR Business Partner. Generate the opening of an HR screening interview.

Candidate's name: {name_part}
Resume context (use only 1 brief personal detail if helpful): {resume_context[:400] if resume_context else "Not provided"}

Your message will be spoken aloud. Follow this EXACT structure — no deviation:
1. Greet the candidate: "Hi {name_part}, I'm Priya, Senior HR Business Partner here."
2. One warm sentence mentioning the company or the role (keep it vague since you don't have specifics).
3. One sentence acknowledging something brief and genuine from their background if available, OR a warm welcome line.
4. End your message EXACTLY with this sentence, word for word: "Tell me about yourself and what drew you to this opportunity."

Rules:
- Do NOT ask "How are you feeling?" or any other question before the Stage 1 question.
- Do NOT add anything after "Tell me about yourself and what drew you to this opportunity."
- Keep the total message to 3 to 4 sentences. Spoken, warm, natural."""
        else:
            prompt = f"""Generate a personalized greeting for the interview. You are {interviewer_name}, a {interviewer_role} at {company_name}.
        {name_context}
        Resume Context:
        {resume_context}

        Your greeting will be spoken aloud.
        - Introduce yourself: "Hi, I'm {interviewer_name}. I'm a {interviewer_role} at {company_name}."
        - Welcome them warmly and personally
        - Reference something brief from their resume if relevant
        - Keep it conversational and natural, not formal."""

        try:
            _greeting_system = (
                HR_SYSTEM_PROMPT
                if is_hr_interview
                else (
                    COMMON_SYSTEM_PROMPT
                    + f" You are {interviewer_name}, a {interviewer_role} at {company_name}."
                    " Welcome the candidate genuinely and personally. Be warm and conversational."
                )
            )
            greeting = await self.llm_helper.call_llm_creative(
                system_prompt=_greeting_system,
                user_prompt=prompt,
            )

            state_updates: dict = {
                "last_node": "greeting",
                "phase": "intro",
                "next_message": greeting,
            }

            # For HR interviews, Stage 1 (Opening) is delivered inside the greeting.
            # Mark the first plan topic as covered so question_node begins at Stage 2.
            if is_hr_interview:
                plan = state.get("interview_plan") or {}
                topics = plan.get("topics", [])
                if topics:
                    opening_id = topics[0].get("id")
                    if opening_id:
                        updated_plan = update_topic_in_plan(plan, opening_id, {
                            "coverage_status": COVERAGE_ADEQUATE,
                            "iterations_done": 1,
                        })
                        state_updates["interview_plan"] = updated_plan

            return state_updates

        except Exception:
            default_greeting = (
                "Hi there, I'm Priya, Senior HR Business Partner. "
                "Thanks for joining today. Tell me about yourself and what drew you to this opportunity."
                if is_hr_interview
                else "Hello! Welcome to your interview. I'm looking forward to learning more about your background."
            )
            return {
                "last_node": "greeting",
                "phase": "intro",
                "next_message": default_greeting,
            }

    async def question_node(self, state: InterviewState) -> InterviewState:
        """Generate the next question, driven by the interview plan when available.

        Plan-driven path:
          1. Get next pending topic from plan.
          2. Use topic's initial_question as the anchor.
          3. LLM crafts a natural, conversational version (not verbatim).
          4. Mark topic as in_progress and set current_topic_id.

        Reactive fallback (no plan):
          LLM generates freely based on resume + conversation (old behaviour).
        """
        resume_context = build_resume_context(state)
        job_context = build_job_context(state)
        candidate_name = state.get("candidate_name")
        name_note = (
            f"\nCandidate's first name: {candidate_name.split()[0]}"
            if candidate_name else ""
        )

        # ── Plan-driven path ──────────────────────────────────────────────────
        plan = state.get("interview_plan")
        next_topic = get_next_pending_topic(plan) if plan else None

        if next_topic:
            seniority = state.get("seniority_level") or plan.get("seniority_level", SENIORITY_MID)
            depth_rules = DEPTH_RULES.get(seniority, DEPTH_RULES[SENIORITY_MID])
            expected_depth = depth_rules["expected_depth"]
            difficulty = state.get("difficulty_mode") or DIFFICULTY_MEDIUM
            difficulty_suffix = DIFFICULTY_PROBE_STYLE_SUFFIX.get(difficulty, "")
            _framing = DIFFICULTY_QUESTION_FRAMING.get(difficulty, "")
            question_framing_line = f"\nQUESTION FRAMING: {_framing}" if _framing else ""
            tone_adjustment = self._derive_tone_adjustment(state)
            tone_instruction = self._tone_instruction(tone_adjustment)

            # Topic-scoped: only the last few exchanges matter for transitioning.
            # Full topic history of completed topics is captured in the summary.
            recent_context = build_conversation_context(
                state, self.interview_logger, scope="recent", recent_n=4)
            # Compact line (topic names only) — quality scores and exchange counts
            # add ~350-550 tokens without helping the LLM generate the next question.
            completed_summary = build_covered_topics_line(state)

            last_user_response = ""
            for msg in reversed(state.get("conversation_history", [])):
                if msg.get("role") == "user":
                    last_user_response = msg.get("content", "")
                    break

            # Determine coding topic and language FIRST — both language_scope_note
            # and the sandbox setup block below depend on these values.
            is_coding_topic = (
                next_topic.get("category") == TOPIC_CODING
                or next_topic.get("requires_code") is True
            )
            coding_language = (
                plan.get("coding_language") or "python"
            ).lower()

            # Build a verification hint when this topic is testing a specific resume claim
            is_resume_claim = next_topic.get("source") == "resume_claim"
            resume_claim_note = ""
            if is_resume_claim and next_topic.get("resume_claim"):
                resume_claim_note = f"""
RESUME VERIFICATION MODE: This topic is specifically testing the following claim from the
candidate's resume:
  "{next_topic['resume_claim']}"

Ask a POINTED follow-up that would be difficult for someone who DIDN'T actually do this work.
Probe specific technical decisions, trade-offs, or challenges — not just a generic description.
The goal is to confirm the claim is genuine, not to trap the candidate."""

            # Language scoping for coding topics: explicitly restrict to the detected
            # programming language so the LLM does not mix e.g. MongoDB queries with
            # Python code, or SQL syntax with a Python coding exercise.
            language_scope_note = ""
            if is_coding_topic:
                language_scope_note = (
                    f"\nCODING LANGUAGE CONSTRAINT: The code editor is set up for "
                    f"{coding_language.upper()} only. The coding question and all follow-ups "
                    f"MUST be solvable purely in {coding_language}. "
                    f"Do NOT mix in SQL queries, MongoDB syntax, shell commands, or any other "
                    f"language/tool unless they are the native standard library of {coding_language}."
                )

            prompt = f"""You are conducting an interview. Move to the next planned topic.
{name_note}

NEXT TOPIC TO COVER: {next_topic['topic']}
CATEGORY: {next_topic['category']}
SUGGESTED OPENING: "{next_topic['initial_question']}"
CANDIDATE SENIORITY: {seniority} (expected depth: {expected_depth})
INTERVIEW DIFFICULTY: {difficulty.upper()}{(' — ' + difficulty_suffix) if difficulty_suffix else ''}
ADAPTIVE TONE MODE: {tone_adjustment}
TONE GUIDANCE: {tone_instruction}{question_framing_line}
{resume_claim_note}{language_scope_note}
{job_context}
Resume Context:
{resume_context}
{completed_summary}

Recent Conversation (last 2 exchanges):
{recent_context}

CANDIDATE'S LAST ANSWER (if any): {last_user_response or "None yet (first question)"}

Your response will be spoken aloud.
STRUCTURE:
1. If there was a last answer: start with a brief, genuine acknowledgment (1 sentence, vary phrasing — e.g. "That's really interesting.", "I appreciate you sharing that.", "Good point.", "Nice approach.").
2. Then bridge naturally into the new topic. Don't say "Moving on to..." — weave it conversationally.
3. Ask the opening question for this topic. Use the suggested opening as inspiration but make it feel natural, not scripted. Adjust complexity for {seniority} level and {difficulty} difficulty.

One question only. Keep it conversational."""

            try:
                response = await self.llm_helper.call_llm_with_instructor(
                    system_prompt=(
                        (
                            HR_SYSTEM_PROMPT
                            if (state.get("user_interview_type") or "").strip().lower() == USER_INTERVIEW_HR
                            else COMMON_SYSTEM_PROMPT
                        )
                        + f" Always respond in English. You are calibrating questions for a {seniority}-level candidate "
                        f"in a {difficulty.upper()} difficulty interview (expected depth: {expected_depth}). "
                        f"Acknowledge briefly then transition naturally to the next topic."
                        f" Tone mode: {tone_adjustment}. {tone_instruction}"
                        + (" Ask a pointed, specific question that tests genuine experience." if is_resume_claim else "")
                    ),
                    user_prompt=prompt,
                    response_model=QuestionGeneration,
                    temperature=TEMPERATURE_QUESTION,
                )

                question_text = response.question.strip()
                if self._is_duplicate_question(question_text, state):
                    # Pick first unused bank question; only fall back to initial_question
                    # as a last resort (it may itself be a duplicate on reconnect).
                    bank = next_topic.get("question_bank", [])
                    fallback = next(
                        (bq for bq in bank if not self._is_duplicate_question(bq, state)),
                        None,
                    )
                    question_text = fallback or next_topic["initial_question"]

                question_record: QuestionRecord = {
                    "id": str(uuid.uuid4()),
                    "text": question_text,
                    "source": "plan",
                    "resume_anchor": response.resume_anchor,
                    "aspect": next_topic["category"],
                    "asked_at_turn": state["turn_count"],
                    "planned_topic_id": next_topic["id"],
                }

                # Update plan: mark topic as in_progress
                updated_plan = update_topic_in_plan(plan, next_topic["id"], {
                    "coverage_status": COVERAGE_IN_PROGRESS,
                })

                # Update topics_covered list (simple string tracking, keep for analytics)
                topics_covered = list(state.get("topics_covered", []))
                if next_topic["topic"] not in topics_covered:
                    topics_covered.append(next_topic["topic"])

                # ── Coding topic: auto-show the code editor ───────────────
                if is_coding_topic:
                    existing_sandbox = state.get("sandbox") or {}
                    sandbox_update = {
                        **existing_sandbox,
                        "is_active": True,
                        "exercise_description": question_text,
                        "initial_code": _get_starter_template(coding_language),
                        "exercise_language": coding_language,
                        "exercise_difficulty": state.get("difficulty_mode", "medium"),
                        "exercise_hints": next_topic.get("question_bank", [])[:3],
                    }
                else:
                    # Leaving coding topic: explicitly deactivate sandbox state
                    # so UI polling cannot accidentally keep the editor visible.
                    existing_sandbox = state.get("sandbox") or {}
                    sandbox_update = {
                        **existing_sandbox,
                        "is_active": False,
                    }

                return {
                    "last_node": "question",
                    "phase": "exploration",
                    "current_question": question_text,
                    "next_message": question_text,
                    "questions_asked": [question_record],
                    "current_topic_id": next_topic["id"],
                    "interview_plan": updated_plan,
                    "topics_covered": topics_covered,
                    "topic_idk_offered": False,  # reset IDK offer flag for new topic
                    # Show code editor only for coding topics; hide for all others
                    "show_code_editor": is_coding_topic,
                    "sandbox": sandbox_update,
                }

            except Exception as e:
                logger.error(f"Plan-driven question generation failed: {e}", exc_info=True)
                # Fall through to reactive path

        # ── Reactive fallback (no plan or plan exhausted) ─────────────────────
        topics_covered = state.get("topics_covered", [])
        questions_asked = [q["text"] for q in state.get("questions_asked", [])[-10:]]
        topics_info = (
            f"\nTopics Already Covered: {', '.join(topics_covered)}\nExplore new topics or go deeper."
            if topics_covered else ""
        )

        # Reactive path needs a broader view since there's no structured plan.
        full_conversation_context = build_conversation_context(
            state, self.interview_logger, scope="full")

        prompt = f"""You are conducting an interview. Generate the next response.
{name_note}
{job_context}
Resume Context: {resume_context}
{topics_info}

Full Conversation:
{full_conversation_context}

Questions Already Asked:
{chr(10).join(f"- {q}" for q in questions_asked) if questions_asked else "None yet"}

Your response will be spoken aloud.
STRUCTURE:
1. If there was a previous answer: acknowledge it briefly (1 sentence, vary phrasing).
2. Ask a natural question relevant to their background.
Skip acknowledgment if this is the very first question."""

        try:
            response = await self.llm_helper.call_llm_with_instructor(
                system_prompt=(
                    (
                        HR_SYSTEM_PROMPT
                        if (state.get("user_interview_type") or "").strip().lower() == USER_INTERVIEW_HR
                        else COMMON_SYSTEM_PROMPT
                    )
                    + " Always respond in English. Acknowledge then ask. Be genuine."
                ),
                user_prompt=prompt,
                response_model=QuestionGeneration,
                temperature=TEMPERATURE_QUESTION,
            )

            question_text = response.question.strip()
            if self._is_duplicate_question(question_text, state):
                question_text = "Can you tell me about a challenging project you've worked on?"

            question_record: QuestionRecord = {
                "id": str(uuid.uuid4()),
                "text": question_text,
                "source": "resume",
                "resume_anchor": response.resume_anchor,
                "aspect": response.aspect,
                "asked_at_turn": state["turn_count"],
                "planned_topic_id": None,
            }

            existing_questions = state.get("questions_asked", [])
            already_asked = any(q.get("text") == question_text for q in existing_questions)

            updates: dict = {
                "last_node": "question",
                "phase": "exploration",
                "current_question": question_text,
                "next_message": question_text,
            }
            if not already_asked:
                updates["questions_asked"] = [question_record]

            if response.resume_anchor and response.resume_anchor not in topics_covered:
                current_topics = list(topics_covered)
                current_topics.append(response.resume_anchor)
                updates["topics_covered"] = current_topics

            return updates

        except Exception as e:
            logger.error(f"Reactive question generation failed: {e}", exc_info=True)
            fallback = "Can you tell me about a challenging project you've worked on?"
            return {
                "last_node": "question",
                "phase": "exploration",
                "current_question": fallback,
                "next_message": fallback,
            }

    async def followup_node(self, state: InterviewState) -> InterviewState:
        """Generate follow-up question or clarification.

        Returns partial state update. conversation_history is written by finalize_turn_node.
        """
        last_question = state.get("current_question", "")
        last_answer = state.get("last_response", "")

        active_request = state.get("active_user_request")
        needs_clarification = (
            active_request and
            active_request.get("type") == "clarify"
        )

        # Get seniority-calibrated probe style
        seniority = state.get("seniority_level", SENIORITY_MID)
        depth_rules = DEPTH_RULES.get(seniority, DEPTH_RULES[SENIORITY_MID])
        expected_depth = depth_rules["expected_depth"]
        probe_style = depth_rules["probe_style"]
        # probe_examples / probe_hint removed — probe_style already encodes the
        # expected probing behaviour; the example phrases added ~20 tokens each call
        # without improving follow-up quality.

        # Difficulty-mode probe modifier (appended to prompt to tune aggressiveness)
        difficulty = state.get("difficulty_mode") or DIFFICULTY_MEDIUM
        difficulty_suffix = DIFFICULTY_PROBE_STYLE_SUFFIX.get(difficulty, "")
        tone_adjustment = self._derive_tone_adjustment(state)
        tone_instruction = self._tone_instruction(tone_adjustment)

        # Adaptive difficulty: adjust follow-up complexity based on rolling performance.
        # Strong performers get harder follow-ups; weak performers get more scaffolding.
        performance_signal = float(state.get("performance_signal") or 0.0)
        if performance_signal >= 0.72:
            adaptive_note = (
                "\nADAPTIVE DIFFICULTY: Candidate is performing strongly. "
                "Increase the complexity of your follow-up — probe edge cases, "
                "failure modes, or trade-offs they haven't mentioned yet."
            )
        elif performance_signal <= 0.35 and performance_signal > 0.0:
            adaptive_note = (
                "\nADAPTIVE DIFFICULTY: Candidate is struggling. "
                "Simplify your follow-up — try a more concrete, guided question "
                "or break the topic into a smaller, more approachable piece."
            )
        else:
            adaptive_note = ""

        # Get current topic context from plan
        plan = state.get("interview_plan")
        current_topic_id = state.get("current_topic_id")
        current_topic = get_topic_by_id(plan, current_topic_id) if plan and current_topic_id else None
        topic_context = (
            f"\nCurrent topic: {current_topic['topic']} (category: {current_topic['category']})"
            if current_topic else ""
        )

        # ── Dedicated coding topic arc ─────────────────────────────────────────
        # For coding topics, we guide the conversation through a structured arc:
        # approach → complexity → edge cases → invite to code → post-submission → optimization
        # This fires BEFORE the generic bank/probe logic so the arc stays on rails.
        is_coding_topic = (
            current_topic and current_topic.get("category") == TOPIC_CODING
            and not needs_clarification
            and not (_is_idk_response(last_answer or "") and not state.get("topic_idk_offered"))
        )
        if is_coding_topic:
            coding_followup = await self._coding_arc_followup(state, current_topic, current_topic_id, seniority, tone_adjustment, difficulty, difficulty_suffix)
            if coding_followup is not None:
                return coding_followup
        # ── End coding arc ────────────────────────────────────────────────────

        is_coding_clarify_context = bool(
            current_topic and current_topic.get("category") == TOPIC_CODING
        )

        if needs_clarification:
            if is_coding_clarify_context:
                sandbox = state.get("sandbox") or {}
                hint_pool = list(sandbox.get("exercise_hints") or [])
                hints_block = (
                    "\n".join(f"  - {h}" for h in hint_pool[:4])
                    if hint_pool
                    else "  (No pre-authored hints — give ONE small hint, e.g. counting occurrences or scanning order.)"
                )
                prompt = f"""The candidate needs help while discussing THIS coding exercise (same session).

LAST THING YOU ASKED / SAID: "{last_question}"
THEIR MESSAGE: "{last_answer}"

AUTHORIZED HINT IDEAS (use at most ONE, paraphrase in your own words — do not read verbatim as a list):
{hints_block}

Rules (spoken aloud):
1. Do NOT repeat the entire problem statement from scratch or sound like you're ignoring their prior answers.
2. If they asked for hints or said they're stuck: give ONE concrete hint only.
3. If they didn't understand a phrase: clarify that phrase in plain language (2 short sentences max).
4. Tell them clearly: **use the code editor on your screen** — implement the solution there, then use **Submit** when ready; you will review their code and ask follow-ups (complexity, edge cases) after.
5. Keep total response under 6 sentences. Stay warm and collaborative — like a real pair-programming interviewer."""

            else:
                prompt = f"""The candidate asked for clarification on this question: "{last_question}"

        Your response will be spoken aloud.
        - Briefly acknowledge their confusion (e.g. "Of course, let me rephrase that.")
        - Rephrase clearly and simply. Break it down if needed."""
        elif _is_idk_response(last_answer or "") and not state.get("topic_idk_offered"):
            # First time candidate says they don't know — offer a retry or skip.
            # The decision node acts on their next reply (yes → rephrase, no → skip topic).
            offer = (
                "No problem at all! Would you like to try this question again? "
                "Take your time — I can rephrase it or give you a moment to think. "
                "Just say 'yes' to try again or 'no' to move on to the next topic."
            )
            return {
                "last_node": "followup",
                "topic_idk_offered": True,
                "current_question": offer,
                "next_message": offer,
            }
        else:
            # If a pending bank question was interrupted, resume it before selecting
            # any new bank item.
            if (
                state.get("pending_question_status") == "interrupted"
                and state.get("pending_question_text")
                and state.get("pending_topic_id") == current_topic_id
            ):
                pending_text = state["pending_question_text"]
                resume_prompt = f"""You are resuming an interview question that was interrupted by a side query.

CANDIDATE SENIORITY: {seniority}
INTERVIEW DIFFICULTY: {difficulty.upper()}{(' — ' + difficulty_suffix) if difficulty_suffix else ''}
ADAPTIVE TONE MODE: {tone_adjustment}
TONE GUIDANCE: {tone_instruction}
CURRENT TOPIC: {current_topic.get('topic', '') if current_topic else ''}
INTERRUPTED QUESTION TO RESUME:
"{pending_text}"

Your response will be spoken aloud.
STRUCTURE:
1. Briefly acknowledge the interruption was handled.
2. Naturally bring the candidate back and ask the same question (you may lightly rephrase, preserve intent).
One question only."""

                try:
                    resumed = await self.llm_helper.call_llm_creative(
                        system_prompt=(
                            COMMON_SYSTEM_PROMPT +
                            " Always respond in English. Smoothly resume the same unresolved question."
                        ),
                        user_prompt=resume_prompt,
                    )
                except Exception:
                    resumed = f"Thanks for that. Let's continue with this question: {pending_text}"

                return {
                    "last_node": "followup",
                    "current_question": resumed,
                    "next_message": resumed,
                    "pending_question_status": "asked",
                }

            # ── Question bank: serve next unused sub-question before LLM probing ─
            # The bank holds distinct angles on the topic (not just deeper probes of
            # the opener). Pick the next one before falling back to LLM generation.
            if current_topic and current_topic_id:
                bank = current_topic.get("question_bank", [])
                topic_iter_entry = state.get("topic_iterations", {}).get(current_topic_id, {})
                bank_index = topic_iter_entry.get("question_bank_index", 0)
                if not bank:
                    logger.debug(
                        "Question bank empty for topic '%s' (id=%s); falling back to generated probe",
                        current_topic.get("topic", ""),
                        current_topic_id,
                    )

                if bank_index < len(bank):
                    bank_question = bank[bank_index]

                    # Deliver the bank question with a natural acknowledgment transition
                    bank_prompt = f"""You are conducting an interview and need to pivot to a new angle on the current topic.

CANDIDATE SENIORITY: {seniority}
INTERVIEW DIFFICULTY: {difficulty.upper()}{(' — ' + difficulty_suffix) if difficulty_suffix else ''}
ADAPTIVE TONE MODE: {tone_adjustment}
TONE GUIDANCE: {tone_instruction}
CURRENT TOPIC: {current_topic.get('topic', '')}
CANDIDATE'S LAST ANSWER: "{last_answer}"

NEXT QUESTION TO ASK (from the pre-planned question bank):
"{bank_question}"

Your response will be spoken aloud.
STRUCTURE:
1. Acknowledge the candidate's last answer briefly (1 sentence, genuine, varied).
2. Naturally transition into the next question. Use the bank question above — you may rephrase it slightly to flow naturally, but preserve its intent. Do NOT ask a different question.
Keep it conversational. One question only."""

                    try:
                        delivered = await self.llm_helper.call_llm_creative(
                            system_prompt=(
                                COMMON_SYSTEM_PROMPT +
                                f" Always respond in English. Acknowledge briefly then ask the specified question naturally."
                                f" Tone mode: {tone_adjustment}. {tone_instruction}"
                            ),
                            user_prompt=bank_prompt,
                        )
                        question_record: QuestionRecord = {
                            "id": str(uuid.uuid4()),
                            "text": delivered,
                            "source": "bank",
                            "resume_anchor": None,
                            "aspect": current_topic.get("category", "technical"),
                            "asked_at_turn": state["turn_count"],
                            "planned_topic_id": current_topic_id,
                        }
                        logger.debug(
                            f"Bank question served for topic '{current_topic.get('topic')}': "
                            f"index {bank_index} → '{bank_question[:80]}'"
                        )
                        updates: dict = {
                            "last_node": "followup",
                            "current_question": delivered,
                            "next_message": delivered,
                            "pending_question_id": question_record["id"],
                            "pending_question_text": delivered,
                            "pending_topic_id": current_topic_id,
                            "pending_bank_index": bank_index,
                            "pending_question_status": "asked",
                        }
                        if not self._is_duplicate_question(delivered, state):
                            updates["questions_asked"] = [question_record]
                        return updates
                    except Exception as bank_err:
                        logger.warning(f"Bank question delivery failed ({bank_err}), falling back to LLM probe")
            # ─────────────────────────────────────────────────────────────────

            prompt = f"""Generate a depth-probe follow-up. The interviewer is pushing deeper on the current topic.

        CANDIDATE SENIORITY: {seniority} — probe style: {probe_style}
        INTERVIEW DIFFICULTY: {difficulty.upper()}{(' — ' + difficulty_suffix) if difficulty_suffix else ''}
        ADAPTIVE TONE MODE: {tone_adjustment}
        TONE GUIDANCE: {tone_instruction}{adaptive_note}
        {topic_context}

        Previous Question: "{last_question}"
        Candidate's Answer: "{last_answer}"

        This Topic's Conversation:
        {build_conversation_context(state, self.interview_logger, scope="topic")[-1500:]}

        Your response will be spoken aloud.
        STRUCTURE:
        1. Acknowledge the answer briefly (1 sentence, genuine, varied — not "Great!" every time).
        2. Probe DEEPER with a follow-up calibrated to {seniority} level and {difficulty} difficulty. For {expected_depth} depth, ask about specifics, trade-offs, and real examples.
        Keep it conversational. One question only."""

        if needs_clarification and is_coding_clarify_context:
            _followup_system = (
                COMMON_SYSTEM_PROMPT
                + " Always respond in English. This is a live coding interview: give a short hint or clarify one phrase only. "
                "Never paste or re-read the entire exercise. Tell them to type in the code editor panel and press Submit when ready — "
                "you will discuss their solution and complexity afterward. "
                f"Tone mode: {tone_adjustment}. {tone_instruction}"
            )
        elif (state.get("user_interview_type") or "").strip().lower() == USER_INTERVIEW_HR:
            # HR interviews: always use HR_SYSTEM_PROMPT so STAR probing rules and the
            # Priya question bank are in scope for every follow-up.
            _followup_system = (
                HR_SYSTEM_PROMPT
                + f" Always respond in English. You are following up on a {seniority}-level candidate's answer. "
                f"If the answer lacks STAR structure (Situation, Task, Action, Result), probe with: "
                f"'Could you walk me through a specific situation where that happened?' "
                f"Acknowledge briefly (1 sentence) then ask ONE focused follow-up. "
                f"Tone mode: {tone_adjustment}."
            )
        else:
            _followup_system = (
                COMMON_SYSTEM_PROMPT
                + f" Always respond in English. You are probing a {seniority}-level candidate "
                f"in a {difficulty.upper()} difficulty interview. "
                f"Probe style: {probe_style}. Acknowledge briefly then push deeper with a targeted follow-up. "
                f"Tone mode: {tone_adjustment}. {tone_instruction}"
            )

        try:
            followup = await self.llm_helper.call_llm_creative(
                system_prompt=_followup_system,
                user_prompt=prompt,
            )
            if self._is_duplicate_question(followup, state):
                followup = "Can you give me a concrete example of that from your experience?"

            question_record: QuestionRecord = {
                "id": str(uuid.uuid4()),
                "text": followup,
                "source": "followup",
                "resume_anchor": None,
                "aspect": "deep_dive",
                "asked_at_turn": state["turn_count"],
            }

            updates = {
                "last_node": "followup",
                "current_question": followup,
                "next_message": followup,
            }
            if not self._is_duplicate_question(followup, state):
                updates["questions_asked"] = [question_record]

            return updates
        except Exception:
            fallback = "Can you provide more details about that?"
            return {
                "last_node": "followup",
                "current_question": fallback,
                "next_message": fallback,
            }

    async def _coding_arc_followup(
        self,
        state: "InterviewState",
        current_topic: dict,
        current_topic_id: str,
        seniority: str,
        tone_adjustment: str,
        difficulty: str = DIFFICULTY_MEDIUM,
        difficulty_suffix: str = "",
    ) -> "InterviewState | None":
        """Drive the coding interview arc: approach → complexity → edge cases → code → review → optimize.

        Returns a state update dict, or None to fall through to the generic followup logic.

        Arc stages (keyed on question_bank_index + code submission presence):
          0 = complexity & approach discussion
          1 = edge cases
          2 = invite to code (unless code already submitted)
          3 = post-submission walkthrough / review (only after code submitted)
          4 = optimization follow-up
        """
        last_answer = state.get("last_response", "") or ""
        last_question = state.get("current_question", "") or ""
        topic_iterations = state.get("topic_iterations", {})
        iter_entry = topic_iterations.get(current_topic_id, {})
        bank_index = iter_entry.get("question_bank_index", 0)
        bank = current_topic.get("question_bank", [])
        topic_name = current_topic.get("topic", "the coding problem")
        code_submissions = state.get("code_submissions", [])
        code_submitted = bool(code_submissions)

        # Determine which arc stage to run based on bank_index and code submission
        arc_stage = bank_index  # 0, 1, 2, 3, 4 ...

        # Stage 2 is the "invite to code" step — skip it if code is already submitted
        if arc_stage == 2 and code_submitted:
            arc_stage = 3  # jump to post-submission review

        # Stages 3+ require code to be submitted. If the candidate is still
        # working (bank_index advanced past invite-to-code but no submission yet),
        # hold here and warmly remind them to hit Submit — don't ask walkthrough
        # or optimization questions before there's any code to discuss.
        if arc_stage >= 3 and not code_submitted:
            return await self._await_code_submission_prompt(
                state, current_topic, current_topic_id, seniority, tone_adjustment
            )

        # If we've already exhausted the bank, fall through to generic LLM probe
        if arc_stage >= len(bank):
            return None

        bank_question = bank[arc_stage]

        # Build a stage-aware system instruction
        stage_context: dict[int, str] = {
            0: (
                "The candidate just explained their high-level approach. "
                "Now probe the time and space complexity, and whether a more optimal solution exists. "
                "Be encouraging but technically precise."
            ),
            1: (
                "The candidate has discussed complexity. "
                "Now ask about edge cases — empty input, duplicates, no valid answer, overflow. "
                "Keep it conversational, not an interrogation."
            ),
            2: (
                "Approach, complexity, and edge cases have all been discussed — time to write code! "
                "Enthusiastically invite the candidate to implement their solution in the code editor "
                "panel visible on their right-hand side. Tell them to click 'Submit' when they're done "
                "and you'll walk through it together. "
                "Be warm and energising — like a real interviewer: 'Go ahead and code it up! Take your "
                "time, think out loud as you write, and just hit Submit when you're happy with it.' "
                "Do NOT re-state the full problem (they can read it in the panel). Keep it brief."
            ),
            3: (
                "The candidate has submitted their code. "
                "Now walk through the implementation together — ask them to explain specific choices. "
                "Look for differences between their stated approach and what they actually coded. "
                "Be collaborative, not adversarial."
            ),
            4: (
                "The implementation walkthrough is done. "
                "Push for optimization — time/space trade-offs, alternative data structures, "
                "or how the solution would scale to very large inputs."
            ),
        }
        stage_instruction = stage_context.get(arc_stage, "Continue probing the coding topic naturally.")

        tone_instruction = self._tone_instruction(tone_adjustment)
        last_submission_code = ""
        if code_submitted and arc_stage >= 3:
            last_submission = code_submissions[-1]
            last_submission_code = (last_submission.get("code", "") or "")[:600]

        code_context = (
            f"\nCandidate's submitted code (latest):\n```\n{last_submission_code}\n```"
            if last_submission_code else ""
        )

        arc_prompt = f"""You are conducting a coding interview. You're at a specific stage of the conversation arc.

TOPIC: {topic_name}
CANDIDATE SENIORITY: {seniority}
INTERVIEW DIFFICULTY: {difficulty.upper()}{(' — ' + difficulty_suffix) if difficulty_suffix else ''}
ADAPTIVE TONE MODE: {tone_adjustment}
TONE GUIDANCE: {tone_instruction}

STAGE CONTEXT: {stage_instruction}

CANDIDATE'S LAST ANSWER: "{last_answer}"
LAST QUESTION YOU ASKED: "{last_question}"{code_context}

NEXT QUESTION TO ASK (use this as your anchor — rephrase naturally but keep its intent):
"{bank_question}"

Your response will be spoken aloud.
STRUCTURE:
1. Acknowledge the candidate's last response briefly (1 sentence, genuine, varied).
2. Smoothly transition into the next question above.
Keep it conversational and warm. One question only.
CRITICAL: Do NOT repeat the original coding problem headline or restate the full exercise — they already have it in the editor. Build on what they said. Do not read the candidate's words back verbatim."""

        try:
            delivered = await self.llm_helper.call_llm_creative(
                system_prompt=(
                    COMMON_SYSTEM_PROMPT
                    + f" You are conducting a {difficulty.upper()} difficulty coding interview. "
                    "Guide the candidate through a structured arc: "
                    "approach → complexity → edge cases → implement → review → optimize. "
                    "Always respond in English. Be conversational, technical, and encouraging."
                    f" Tone mode: {tone_adjustment}. {tone_instruction}"
                ),
                user_prompt=arc_prompt,
            )
        except Exception as arc_err:
            logger.warning(f"Coding arc LLM call failed ({arc_err}), using bank question directly")
            delivered = f"Thanks for that. {bank_question}"

        question_record: QuestionRecord = {
            "id": str(uuid.uuid4()),
            "text": delivered,
            "source": "bank",
            "resume_anchor": None,
            "aspect": TOPIC_CODING,
            "asked_at_turn": state["turn_count"],
            "planned_topic_id": current_topic_id,
        }
        updates: dict = {
            "last_node": "followup",
            "current_question": delivered,
            "next_message": delivered,
            "pending_question_id": question_record["id"],
            "pending_question_text": delivered,
            "pending_topic_id": current_topic_id,
            "pending_bank_index": arc_stage,
            "pending_question_status": "asked",
        }
        if not self._is_duplicate_question(delivered, state):
            updates["questions_asked"] = [question_record]
        return updates

    async def _await_code_submission_prompt(
        self,
        state: "InterviewState",
        current_topic: dict,
        current_topic_id: str,
        seniority: str,
        tone_adjustment: str,
    ) -> "InterviewState":
        """Generate a warm 'I'm waiting for your code submission' response.

        Called when the candidate is at the implement stage (bank_index >= 3)
        but hasn't clicked Submit yet. Acknowledges whatever they said and
        gently reminds them to submit. Does NOT advance the bank index so the
        arc stays on the same stage until code arrives.
        """
        last_answer = (state.get("last_response") or "").strip()
        topic_name = current_topic.get("topic", "the problem")
        tone_instruction = self._tone_instruction(tone_adjustment)

        prompt = f"""You are conducting a coding interview. You've asked the candidate to implement
"{topic_name}" in the code editor panel on their screen. They haven't clicked Submit yet —
you're waiting for their code.

THEIR LAST MESSAGE: "{last_answer}"
CANDIDATE SENIORITY: {seniority}

Your response will be spoken aloud. Rules:
1. If they asked a question or said something relevant, address it briefly (1-2 sentences max).
2. Then warmly encourage them to keep coding and remind them to click "Submit" when ready.
   Examples of good phrasing:
   - "No worries, take your time! Just hit Submit in the editor whenever you're happy with it and we'll walk through it together."
   - "That's a great point — go ahead and code that up! I'll be right here. Just Submit when you're ready."
   - "Sounds good! Keep going — whenever you feel good about your solution, click Submit and we'll review it."
3. Do NOT ask a new technical question — just hold the space warmly until they submit.
4. Keep it to 1–3 sentences. Natural and conversational, not robotic."""

        try:
            delivered = await self.llm_helper.call_llm_creative(
                system_prompt=(
                    COMMON_SYSTEM_PROMPT
                    + " You are a coding interviewer waiting for the candidate to submit their code. "
                    "Be warm, patient, and brief. Don't ask technical questions — just encourage them to Submit."
                    f" Tone mode: {tone_adjustment}. {tone_instruction}"
                ),
                user_prompt=prompt,
            )
        except Exception as e:
            logger.warning(f"_await_code_submission_prompt LLM call failed ({e}), using fallback")
            delivered = (
                "Take your time! Whenever you feel good about your solution, just click Submit "
                "and we'll walk through it together."
            )

        # Return WITHOUT setting pending_bank_index — this keeps the bank_index
        # frozen so the arc re-evaluates the same stage on the next voice turn.
        return {
            "last_node": "followup",
            "current_question": delivered,
            "next_message": delivered,
        }

    async def answer_candidate_question_node(self, state: InterviewState) -> InterviewState:
        """Answer a question the candidate asked about the company, role, or process.

        Returns partial state update. Answers directly then transitions back to interview.
        """
        candidate_question = state.get("last_response", "")
        last_interviewer_question = state.get("current_question", "")

        if not self._is_interview_scope_question(candidate_question):
            redirect = (
                "Good question. I want to keep us focused on the interview for now. "
                f"Let's come back to your last response, {last_interviewer_question}"
                if last_interviewer_question
                else "Good question. I want to keep us focused on the interview for now. Let's continue."
            )
            return {
                "last_node": "answer_candidate_question",
                "next_message": redirect,
            }

        # Extract interviewer persona from conversation history
        interviewer_name = "the interviewer"
        company_name = "our company"
        interviewer_role = "Engineering Manager"
        conv_history = state.get("conversation_history", [])
        for msg in conv_history:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                # Attempt to extract persona from greeting (e.g. "I'm Sarah, a ... at TechCorp")
                name_match = re.search(r"I'm ([A-Z][a-z]+)", content)
                company_match = re.search(r"at ([A-Z][A-Za-z\s]+?)[\.,]", content)
                role_match = re.search(r"(?:a |an )([A-Za-z\s]+?) at", content)
                if name_match:
                    interviewer_name = name_match.group(1)
                if company_match:
                    company_name = company_match.group(1).strip()
                if role_match:
                    interviewer_role = role_match.group(1).strip()
                break

        # 300 chars captures role title, company, and primary tech stack — enough
        # for a 2-4 sentence spoken answer; no need for the full JD here.
        _jd_snippet = (state.get("job_description") or "")[:300]
        job_context = f"Job Requirements:\n{_jd_snippet}\n\n" if _jd_snippet else ""

        prompt = f"""The candidate has asked you a question during the interview. Answer it directly and professionally, then transition back to the interview.

Interviewer persona: {interviewer_name}, {interviewer_role} at {company_name}
{job_context}

Candidate's question: "{candidate_question}"
Last interview question you asked (to return to): "{last_interviewer_question}"

Your response will be spoken aloud.
- Answer the candidate's question clearly and concisely
- Be honest. If you don't have specific info (e.g. salary), say so briefly
        - After answering, use a natural transition back to the interview in your own words
- Keep the answer brief — 2-4 sentences max before transitioning back"""

        try:
            answer = await self.llm_helper.call_llm(
                system_prompt=COMMON_SYSTEM_PROMPT +
                f" You are {interviewer_name}, a {interviewer_role} at {company_name}. Always respond in English. Answer the candidate's question honestly and transition back to the interview naturally.",
                user_prompt=prompt,
                temperature=0.1,  # Low temp — factual answer, strict instruction adherence
            )
            return {
                "last_node": "answer_candidate_question",
                "next_message": answer,
            }
        except Exception:
            fallback = f"That's a good question. {company_name} is a great place to work and I'd be happy to tell you more after the interview. For now, let's continue — {last_interviewer_question}"
            return {
                "last_node": "answer_candidate_question",
                "next_message": fallback,
            }

    async def evaluation_node(self, state: InterviewState) -> InterviewState:
        """Generate comprehensive interview evaluation and feedback.

        Returns partial state update. conversation_history is written by finalize_turn_node.
        """

        try:
            topics_covered = state.get("topics_covered", [])

            # Include the last user response if it hasn't been written to conversation_history
            # yet (finalize_turn runs after evaluation, so the final answer can be missing).
            conv_history = list(state.get("conversation_history", []))
            last_resp = (state.get("last_response") or "").strip()
            if last_resp:
                already_present = any(
                    m.get("role") == "user" and m.get("content", "").strip() == last_resp
                    for m in conv_history[-3:]
                )
                if not already_present:
                    from datetime import datetime as _dt
                    conv_history.append({
                        "role": "user",
                        "content": last_resp,
                        "timestamp": _dt.utcnow().isoformat(),
                    })

            comprehensive_feedback = await self.feedback_generator.generate_feedback(
                conversation_history=conv_history,
                resume_context=state.get("resume_structured", {}),
                code_submissions=state.get("code_submissions", []),
                topics_covered=topics_covered,
                job_description=state.get("job_description"),
                evaluation_config=state.get("evaluation_config"),
            )

            feedback_dict = comprehensive_feedback.model_dump()
            return {
                "last_node": "evaluation",
                "phase": "closing",
                "feedback": feedback_dict,
                "next_message": "Thanks, that gives me enough to summarize your interview. I'll wrap up in a moment.",
                "show_code_editor": False,
                "sandbox": {**(state.get("sandbox") or {}), "is_active": False},
            }
        except Exception as e:
            logger.error(
                f"Failed to generate comprehensive feedback: {e}", exc_info=True)
            topics_covered = state.get("topics_covered", [])
            return {
                "last_node": "evaluation",
                "phase": "closing",
                "feedback": {
                    "summary": "Interview completed successfully.",
                    "topics_covered": topics_covered,
                    "turn_count": state.get("turn_count", 0),
                    "overall_score": 0.5,
                },
                "next_message": "Thanks, I'll quickly summarize and then close the interview.",
                "show_code_editor": False,
                "sandbox": {**(state.get("sandbox") or {}), "is_active": False},
            }

    async def termination_node(self, state: InterviewState) -> InterviewState:
        """Immediately terminate the interview due to inappropriate behavior.

        Outputs TERMINATION_MESSAGE verbatim — no LLM call, no variation.
        Sets phase='terminated' so subsequent turns are short-circuited.
        """
        logger.warning(
            f"Interview {state.get('interview_id')} terminated for inappropriate behavior "
            f"at turn {state.get('turn_count', 0)}"
        )
        return {
            "last_node": "termination",
            "phase": "terminated",
            "next_message": TERMINATION_MESSAGE,
            "feedback": {
                **(state.get("feedback") or {}),
                "terminated": True,
                "terminated_reason": "inappropriate_behavior",
                "terminated_at_turn": state.get("turn_count", 0),
            },
            "show_code_editor": False,
            "sandbox": {**(state.get("sandbox") or {}), "is_active": False},
        }

    async def closing_node(self, state: InterviewState) -> InterviewState:
        """Generate closing message.

        Returns partial state update. conversation_history is written by finalize_turn_node.
        """
        # Prefer the LLM-generated rolling summary (already covers the full interview);
        # only fall back to a live context scan when the summary hasn't been built yet.
        # This avoids sending the same information twice (summary + last 12 messages).
        _llm_summary = state.get("conversation_summary", "")
        if _llm_summary and _llm_summary not in ("No conversation yet.", ""):
            context_for_closing = _llm_summary
        else:
            context_for_closing = build_conversation_context(
                state, self.interview_logger, scope="full")
        completed_summary = build_covered_topics_line(state)

        prompt = f"""Generate a closing message for the interview.

        Topics Covered:
        {completed_summary or "See conversation summary below."}

        Conversation Summary:
        {context_for_closing}

        Your closing will be spoken aloud. Thank them genuinely. Reference something specific from the conversation if relevant. Be warm and authentic."""

        try:
            is_hr_interview = (
                (state.get("user_interview_type") or "").strip().lower() == USER_INTERVIEW_HR
            )
            _closing_system = (
                HR_SYSTEM_PROMPT
                + " You are delivering the Stage 6 close of the HR screening interview. "
                "Ask if the candidate has questions (answer up to 2 generically), "
                "then close warmly with the 3 to 5 business day timeline. "
                "Stay in character as Priya throughout."
                if is_hr_interview
                else (
                    COMMON_SYSTEM_PROMPT
                    + " You are closing an interview. Be warm and appreciative. Reference the conversation naturally."
                )
            )
            closing = await self.llm_helper.call_llm_creative(
                system_prompt=_closing_system,
                user_prompt=prompt,
            )
            return {
                "last_node": "closing",
                "phase": "closing",
                "next_message": closing,
                "should_close": True,
                "show_code_editor": False,
                "sandbox": {**(state.get("sandbox") or {}), "is_active": False},
            }
        except Exception:
            return {
                "last_node": "closing",
                "phase": "closing",
                "next_message": "Thank you for your time today. It was great learning more about your background!",
                "should_close": True,
                "show_code_editor": False,
                "sandbox": {**(state.get("sandbox") or {}), "is_active": False},
            }

    async def sandbox_guidance_node(self, state: InterviewState) -> InterviewState:
        """Guide user to use sandbox for writing code, optionally providing an exercise.

        Returns partial state update.
        """
        if self.interview_logger:
            self.interview_logger.log_state("sandbox_guidance", state)

        sandbox = state.get("sandbox", {})
        should_provide_exercise = await self._should_provide_exercise(state)
        exercise = None
        if should_provide_exercise and not sandbox.get("exercise_description"):
            # Generate exercise only once per coding session.
            exercise = await self._generate_coding_exercise(state)

        resume_context = build_resume_context(state)
        job_context = build_job_context(state)
        # Recent only — sandbox guidance just needs the last exchange to stay coherent.
        conversation_context = build_conversation_context(
            state, self.interview_logger, scope="recent", recent_n=4)

        sandbox_already_active = sandbox.get("is_active") and sandbox.get("exercise_description")

        if should_provide_exercise and exercise:
            prompt = f"""Generate a message introducing a coding exercise to the candidate.

        {job_context}

        Resume Context:
        {resume_context}

        Exercise Description:
        {exercise.get('description', '')}

        Your message will be spoken aloud. 
        - Introduce the coding exercise naturally and conversationally
        - Tell them the code sandbox is already set up with starter code
        - Guide them to look at the sandbox on the right side of their screen
        - Be clear and supportive. Keep it natural."""
        elif sandbox_already_active:
            prompt = f"""The candidate has been given a coding exercise but hasn't submitted code yet. Generate a brief, encouraging nudge.

        Conversation Context:
        {conversation_context}

        Your message will be spoken aloud.
        - Acknowledge whatever they just said briefly
        - Gently remind them the code editor is on the right and they should write their solution there
        - Remind them to click the Submit button when they're ready so you can review it together
        - Keep it short and natural — one or two sentences."""
        else:
            prompt = f"""Generate a message guiding the candidate to use the code sandbox.

        Conversation Context:
        {conversation_context}

        Resume Context:
        {resume_context}

        Your message will be spoken aloud. 
        - Acknowledge their request to write code
        - Guide them to the code sandbox on the right side of their screen
        - Let them know you'll review it when they submit
        - Be natural and helpful. Keep it conversational."""

        try:
            guidance_message = await self.llm_helper.call_llm_creative(
                system_prompt=COMMON_SYSTEM_PROMPT +
                " You are guiding a candidate to use the code sandbox. Be clear and helpful.",
                user_prompt=prompt,
            )

            if self.interview_logger:
                self.interview_logger.log_llm_call(
                    "sandbox_guidance", prompt, guidance_message, DEFAULT_MODEL
                )

            sandbox = state.get("sandbox", {})
            sandbox_update = {
                **sandbox,
                "is_active": True,
                "last_activity_ts": time.time(),
            }
            if should_provide_exercise and exercise:
                sandbox_update.update({
                    "initial_code": exercise.get("starter_code", ""),
                    "exercise_description": exercise.get("description", ""),
                    "exercise_difficulty": exercise.get("difficulty", "medium"),
                    "exercise_hints": exercise.get("hints", []),
                    "exercise_language": exercise.get("language", "python"),
                })
                if "hints_provided" not in sandbox_update:
                    sandbox_update["hints_provided"] = []

            return {
                "last_node": "sandbox_guidance",
                "next_message": guidance_message,
                "sandbox": sandbox_update,
                # Signal to frontend: show the code editor panel NOW
                "show_code_editor": True,
            }
        except Exception as e:
            logger.error(
                f"Error generating sandbox guidance: {e}", exc_info=True)
            if self.interview_logger:
                self.interview_logger.log_error("sandbox_guidance", e)
            fallback = "Great! I'd love to see your code. Please use the code sandbox on the right side of your screen. Write your code there and submit it when you're ready, and I'll review it for you."
            return {
                "last_node": "sandbox_guidance",
                "next_message": fallback,
                "show_code_editor": True,
            }

    async def _should_provide_exercise(self, state: InterviewState) -> bool:
        """Determine if agent should provide a coding exercise.

        Plan takes precedence: if we generated a plan and it says no coding,
        we respect that (prevents code editor from popping up for non-tech roles).
        """
        active_request = state.get("active_user_request")
        plan = state.get("interview_plan")
        if plan is not None:
            return plan.get("requires_coding", False)

        if active_request and active_request.get("type") == "write_code":
            return True

        # Fallback heuristic (no plan available)
        job_desc = state.get("job_description", "").lower() if state.get("job_description") else ""
        coding_keywords = ["python", "javascript", "code", "programming", "developer", "engineer", "software"]
        if job_desc and any(keyword in job_desc for keyword in coding_keywords):
            return True

        conversation = build_conversation_context(
            state, self.interview_logger, scope="recent", recent_n=6)
        return "technical" in conversation.lower() or "coding" in conversation.lower()

    async def _generate_coding_exercise(self, state: InterviewState) -> dict:
        """Generate a coding exercise based on job description and resume."""
        # Exercise quality depends on language + level + stack signals only —
        # not the full resume narrative or JD. Compact context saves ~800-1500 tokens.
        _plan = state.get("interview_plan") or {}
        _coding_lang = (_plan.get("coding_language") or "python").lower()
        _seniority = state.get("seniority_level") or "mid"
        _difficulty = state.get("difficulty_mode") or "medium"
        _resume_struct = state.get("resume_structured") or {}
        _raw_skills = _resume_struct.get("skills", [])
        _skills = _raw_skills[:5] if isinstance(_raw_skills, list) else []
        exercise_context = (
            f"Language: {_coding_lang} | Level: {_seniority} | Difficulty: {_difficulty}"
            + (f" | Candidate tech stack: {', '.join(str(s) for s in _skills)}" if _skills else "")
        )

        prompt = f"""Generate a coding exercise for an interview candidate.

        {exercise_context}

        Create a coding exercise that:
        1. Is relevant to the job requirements
        2. Matches the candidate's experience level
        3. Can be completed in 15-30 minutes
        4. Tests practical programming skills

        Return a JSON object with:
        - "description": Clear problem description (2-3 sentences)
        - "starter_code": Python code with function signatures and docstrings, comments explaining the problem
        - "language": "python" or "javascript"
        - "difficulty": "easy", "medium", or "hard"
        - "hints": List of 2-3 hints if candidate gets stuck

        Example starter_code format:
        ```python
    def solve_problem(input_data):
        \"\"\"
        Problem: [description]

        Args:
        input_data: [description]

        Returns:
        [description]
        \"\"\"
        # TODO: Implement your solution here
        pass
        ```

        Return ONLY valid JSON, no markdown formatting."""

        try:
            exercise_json = await self.llm_helper.call_llm_json(
                system_prompt=COMMON_SYSTEM_PROMPT +
                " You are creating coding interview exercises. Generate practical, relevant problems.",
                user_prompt=prompt,
                temperature=TEMPERATURE_CREATIVE,
            )
            exercise = json.loads(exercise_json)
            return exercise
        except Exception as e:
            logger.error(f"Error generating exercise: {e}", exc_info=True)
            return {
                "description": "Implement a function that finds the maximum value in a list.",
                "starter_code": """def find_max(numbers):
        \"\"\"
        Find the maximum value in a list of numbers.

        Args:
        numbers: List of integers

        Returns:
        Maximum integer in the list
        \"\"\"
        # TODO: Implement your solution here
        pass""",
                "language": "python",
                "difficulty": "easy",
                "hints": ["Think about iterating through the list", "Keep track of the maximum value seen so far"]
            }

    async def check_sandbox_code_changes(self, state: InterviewState) -> dict:
        """Poll sandbox for code changes and provide real-time guidance if needed.

        Returns state updates (no mutations). Used by polling endpoint.
        """
        updates: dict = {}

        if not state.get("sandbox", {}).get("is_active"):
            return updates

        sandbox = state.get("sandbox", {})
        last_poll = sandbox.get("last_poll_time", 0.0)
        current_time = time.time()

        if current_time - last_poll < SANDBOX_POLL_INTERVAL_SECONDS:
            return updates

        current_code = state.get("current_code", "")
        last_snapshot = sandbox.get("last_code_snapshot", "")
        initial_code = sandbox.get("initial_code", "")
        last_activity_ts = sandbox.get("last_activity_ts", 0.0)

        # Simplified: Check if stuck based on time since last activity
        time_since_activity = current_time - \
            last_activity_ts if last_activity_ts > 0 else 0
        is_stuck = time_since_activity > SANDBOX_STUCK_THRESHOLD_SECONDS and current_code != initial_code

        sandbox_updates = {
            **sandbox,
            "last_poll_time": current_time,
        }

        # Track code changes
        if current_code and current_code != last_snapshot:
            if current_code != initial_code:
                sandbox_updates["last_code_snapshot"] = current_code
                sandbox_updates["last_activity_ts"] = current_time

        # Provide hints if stuck
        if is_stuck:
            exercise_hints = sandbox.get("exercise_hints", [])
            hints_provided = list(sandbox.get("hints_provided", []))

            if exercise_hints and len(hints_provided) < len(exercise_hints):
                next_hint_index = len(hints_provided)
                next_hint = exercise_hints[next_hint_index]
                hint_message = f"You seem to be stuck. Here's a hint to help you: {next_hint}"

                updates["next_message"] = hint_message
                hints_provided.append(next_hint)
                sandbox_updates["hints_provided"] = hints_provided

                signals = list(sandbox.get("signals", []))
                if "needs_help" not in signals:
                    signals.append("needs_help")
                    sandbox_updates["signals"] = signals

        if sandbox_updates != sandbox:
            updates["sandbox"] = sandbox_updates

        return updates

    async def code_review_node(self, state: InterviewState) -> InterviewState:
        """Execute code, analyze it, and generate feedback.

        Returns partial state update. conversation_history is written by finalize_turn_node.
        """
        if self.interview_logger:
            self.interview_logger.log_state("code_review_start", state)

        code = state.get("current_code")
        if not code:
            return {
                "last_node": "code_review",
                "next_message": "I don't see any code to review. Please submit your code.",
            }

        sandbox = state.get("sandbox", {})
        exercise_description = sandbox.get("exercise_description", "")
        initial_code = sandbox.get("initial_code", "")
        language_str_early = (
            state.get("current_language")
            or sandbox.get("exercise_language")
            or "python"
        ).lower()
        if language_str_early in ("c++", "c_plus_plus"):
            language_str_early = "cpp"

        # Syntax validation — catch obvious errors before spinning up Docker execution
        syntax_error = _validate_code_syntax(code, language_str_early)
        if syntax_error:
            msg = (
                f"Hold on — {syntax_error} "
                "Fix it in the editor and hit Submit again when you're ready."
            )
            return {
                "last_node": "code_review",
                "next_message": msg,
                "pending_voice_message": msg,
                "show_code_editor": True,
            }

        # Check whether the candidate submitted the unmodified starter template.
        # Comparing stripped code catches trivial whitespace differences.
        # If they hit Submit without writing anything meaningful, prompt them to
        # actually implement a solution rather than running a full code review.
        if initial_code and code.strip() == initial_code.strip():
            reminder = (
                "It looks like you submitted the starter template without any implementation. "
                "Go ahead and write your solution in the editor, then hit Submit again when you're ready. "
                "Take your time — think through the approach we discussed and start coding it up!"
            )
            return {
                "last_node": "code_review",
                "next_message": reminder,
                "pending_voice_message": reminder,
                "show_code_editor": True,  # keep editor open — they still need to code
            }

        # Guard against re-reviewing identical code submitted multiple times.
        # Use sandbox.submissions (persisted to DB via _sandbox) as the source of
        # truth — code_submissions in state is rebuilt from conversation_history
        # metadata which finalize_turn_node never populates, so it's always empty.
        existing_sandbox_subs = sandbox.get("submissions", [])
        already_reviewed = any(
            (sub.get("code") or "").strip() == code.strip()
            for sub in existing_sandbox_subs
        )
        if already_reviewed:
            reminder = (
                "I've already reviewed that solution! "
                "Feel free to update your code and submit again, "
                "or just continue our conversation about it."
            )
            return {
                "last_node": "code_review",
                "next_message": reminder,
                "pending_voice_message": reminder,
                "show_code_editor": True,
            }

        sandbox_update = {
            **sandbox,
            "is_active": True,
            "last_activity_ts": datetime.utcnow().timestamp(),
            "signals": list(set(sandbox.get("signals", []) + ["code_submitted"])),
        }

        language_str = (
            state.get("current_language")
            or sandbox.get("exercise_language")
            or "python"
        ).lower()
        if language_str in ("c++", "c_plus_plus"):
            language_str = "cpp"
        try:
            sandbox_language = SandboxLanguage(language_str)
        except ValueError:
            logger.warning(
                f"Unknown sandbox language '{language_str}', defaulting to python"
            )
            sandbox_language = SandboxLanguage.PYTHON
            language_str = "python"

        try:
            execution_result = await self.sandbox_service.execute_code(
                code=code,
                language=sandbox_language,
            )

            exec_result_dict = execution_result.to_dict()
            # Will be included in return statement below

            # Code review needs only the coding topic's conversation, not the full history.
            conversation_summary = build_conversation_context(
                state, self.interview_logger, scope="topic")
            job_context = build_job_context(state)
            code_quality = await self.code_analyzer.analyze_code(
                code=code,
                language=language_str,
                execution_result=exec_result_dict,
                context={
                    "question": state.get("current_question", ""),
                    "conversation_summary": conversation_summary,
                    "job_description": job_context,
                },
            )

            code_quality_dict = {
                "quality_score": code_quality.quality_score,
                "correctness_score": code_quality.correctness_score,
                "efficiency_score": code_quality.efficiency_score,
                "readability_score": code_quality.readability_score,
                "best_practices_score": code_quality.best_practices_score,
                "strengths": code_quality.strengths,
                "weaknesses": code_quality.weaknesses,
                "feedback": code_quality.feedback,
                "suggestions": code_quality.suggestions,
            }

            feedback_message = await self.code_analyzer.generate_code_feedback_message(
                code_quality=code_quality,
                execution_result=exec_result_dict,
            )

            followup_question = await self.code_analyzer.generate_adaptive_question(
                code_quality=code_quality,
                execution_result=exec_result_dict,
                conversation_context=conversation_summary,
            )

            combined_message = f"{feedback_message}\n\n{followup_question}"

            submission = {
                "code": code,
                "language": language_str,
                "execution_result": exec_result_dict,
                "code_quality": code_quality_dict,
                "timestamp": datetime.utcnow().isoformat(),
            }

            sandbox_update = {
                **sandbox,
                "is_active": True,
                "last_activity_ts": datetime.utcnow().timestamp(),
                "submissions": sandbox.get("submissions", []) + [submission],
                "signals": list(set(sandbox.get("signals", []) + ["code_submitted"])),
            }

            try:
                metrics = get_code_metrics()
                metrics.record_execution(
                    user_id=state["user_id"],
                    interview_id=state["interview_id"],
                    code=code,
                    language=language_str,
                    execution_result=exec_result_dict,
                    code_quality=code_quality_dict,
                )
            except Exception:
                pass

            return {
                "last_node": "code_review",
                "next_message": combined_message,
                "current_question": followup_question,
                "code_execution_result": exec_result_dict,
                "code_quality": code_quality_dict,
                "sandbox": sandbox_update,
                # Keep editor visible after review while the coding topic is still
                # active. It will be hidden by question_node when we actually
                # transition to a non-coding topic.
                "show_code_editor": True,
                # Signal the LiveKit voice agent to speak this feedback via TTS.
                # The agent's monitoring loop reads _pending_voice_message from the
                # DB, calls session.say(), then clears the flag.
                "pending_voice_message": combined_message,
                # already_reviewed guard above ensures this is always a new submission
                "code_submissions": [submission],
            }

        except Exception as e:
            logger.error(f"Error in code review: {e}", exc_info=True)
            sandbox = state.get("sandbox", {})
            sandbox_update = {
                **sandbox,
                "signals": list(set(sandbox.get("signals", []) + ["execution_error"])),
            }
            return {
                "last_node": "code_review",
                "next_message": "I encountered an issue reviewing your code. Please try submitting it again.",
                "code_execution_result": {"error": str(e)},
                "sandbox": sandbox_update,
            }