"""Interview management endpoints."""

import asyncio
import logging
from datetime import datetime
from typing import Annotated
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status, Query
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.core.database import get_db
from src.models.user import User
from src.models.resume import Resume
from src.models.interview import Interview
from src.schemas.interview import (
    InterviewCreate,
    InterviewResponse,
    InterviewStart,
    InterviewRespond,
    InterviewComplete,
    InterviewSubmitCode,
    SandboxCodeUpdate,
)
from src.services.orchestrator.langgraph_orchestrator import get_or_create_orchestrator, evict_orchestrator
from src.services.data.state_manager import interview_to_state, state_to_interview
from src.services.data.runtime_interview_context import (
    set_runtime_context,
    get_runtime_context,
    pop_runtime_context,
)
from src.services.analysis.feedback_generator import FeedbackGenerator
from src.services.analytics.analytics_service import InterviewAnalytics
from src.services.voice.livekit_service import LiveKitService
from src.api.v1.dependencies import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("", response_model=InterviewResponse, status_code=status.HTTP_201_CREATED)
async def create_interview(
    interview_data: InterviewCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new interview session."""

    resume_context = None
    # Prefer inline resume_context dict (passed directly, e.g. from tests)
    if interview_data.resume_context:
        resume_context = interview_data.resume_context
        logger.info(
            "[RESUME] create_interview: using inline resume_context "
            "(keys=%s)", list(resume_context.keys()) if isinstance(resume_context, dict) else "non-dict"
        )
    elif interview_data.resume_id:
        result = await db.execute(
            select(Resume).where(
                Resume.id == interview_data.resume_id,
                Resume.user_id == user.id,
                Resume.analysis_status == "completed"
            )
        )
        resume = result.scalar_one_or_none()

        if not resume:
            logger.warning(
                "[RESUME] create_interview: resume_id=%s not found or analysis not complete "
                "(user=%s) — interview will have NO resume context",
                interview_data.resume_id, user.id,
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Resume not found or not analyzed yet",
            )

        resume_context = resume.extracted_data or {}

        # Log what fields are populated vs empty
        populated = [k for k, v in resume_context.items() if v and not k.startswith("_")]
        empty = [k for k, v in resume_context.items() if not v and not k.startswith("_")]
        logger.info(
            "[RESUME] create_interview: attached resume_id=%s (%s) to interview — "
            "populated fields: %s | empty fields: %s | has_job_desc=%s",
            resume.id,
            resume.file_name,
            populated or "NONE",
            empty or "none",
            bool(interview_data.job_description),
        )
        if not populated:
            logger.warning(
                "[RESUME] create_interview: resume_id=%s has ALL-EMPTY extracted_data — "
                "AI will have no resume context. Check if resume analysis succeeded.",
                resume.id,
            )
    else:
        logger.info(
            "[RESUME] create_interview: no resume_id provided — "
            "interview will be created without resume context (has_job_desc=%s)",
            bool(interview_data.job_description),
        )

    if interview_data.interview_mode == "skills_only":
        sk = interview_data.skills or []
        if len(sk) == 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="skills is required when interview_mode is skills_only",
            )

    # Wizard selections must be DB-persisted (not just in-memory) because the
    # LiveKit voice agent runs in a separate process and cannot read the in-memory
    # _runtime_context set by this FastAPI process.  Using `_`-prefixed keys in
    # resume_context JSONB keeps them separate from actual resume data.
    wizard_context: dict = {}
    if interview_data.skills:
        wizard_context["_user_skills"] = interview_data.skills
    if interview_data.interview_mode:
        wizard_context["_forced_interview_mode"] = interview_data.interview_mode
    if interview_data.interview_type:
        wizard_context["_user_interview_type"] = interview_data.interview_type
        wizard_context["_interview_type"] = interview_data.interview_type
    if interview_data.job_description:
        # Persist JD to DB so the LiveKit agent process (separate process with
        # empty runtime_context) can read it back via db_ctx["_job_description"].
        wizard_context["_job_description"] = interview_data.job_description
    if interview_data.duration_minutes is not None:
        wizard_context["_duration_minutes"] = interview_data.duration_minutes
    if interview_data.total_questions is not None:
        wizard_context["_total_questions"] = interview_data.total_questions
    if interview_data.dynamic_questions is not None:
        wizard_context["_dynamic_questions"] = interview_data.dynamic_questions
    if interview_data.evaluation_config:
        wizard_context["_evaluation_config"] = interview_data.evaluation_config
    if interview_data.proctoring_config:
        wizard_context["_proctoring_config"] = interview_data.proctoring_config
    if interview_data.session_source:
        wizard_context["_session_source"] = interview_data.session_source
    if resume_context and isinstance(resume_context, dict):
        # Persist resume structured data to DB so the LiveKit agent process
        # (separate process with empty runtime_context) can read it back via
        # db_ctx["_resume_structured"]. Without this the agent sees no resume
        # data and asks generic questions instead of resume-relevant ones.
        wizard_context["_resume_structured"] = resume_context

    interview = Interview(
        user_id=user.id,
        resume_id=interview_data.resume_id,
        title=interview_data.title,
        status="pending",
        difficulty_mode=interview_data.difficulty_mode,
        resume_context=wizard_context if wizard_context else None,
        job_description=None,
        conversation_history=[],
        turn_count=0,
    )

    db.add(interview)
    await db.commit()
    await db.refresh(interview)
    set_runtime_context(
        interview.id,
        resume_context if isinstance(resume_context, dict) else {},
        interview_data.job_description,
        user_skills=interview_data.skills,
        forced_interview_mode=interview_data.interview_mode,
        user_interview_type=interview_data.interview_type,
    )

    return _interview_to_response(interview)


@router.get("", response_model=list[InterviewResponse])
async def list_interviews(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all interviews for the current user."""
    result = await db.execute(
        select(Interview)
        .where(Interview.user_id == user.id)
        .order_by(Interview.created_at.desc())
    )
    interviews = result.scalars().all()

    return [_interview_to_response(interview) for interview in interviews]


@router.get("/{interview_id}", response_model=InterviewResponse)
async def get_interview(
    interview_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get a specific interview by ID."""
    result = await db.execute(
        select(Interview).where(
            Interview.id == interview_id, Interview.user_id == user.id
        )
    )
    interview = result.scalar_one_or_none()

    if not interview:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Interview not found",
        )

    return _interview_to_response(interview)


async def _prewarm_plan_background(interview_id: int) -> None:
    """Background task: generate the interview plan and cache it in runtime state.

    Runs at /start time (~30-120 seconds before the voice agent joins) so that
    when the agent calls generate_reply() for the greeting, the plan is already
    in the database and plan_interview_node returns immediately instead of
    spending 40-120 seconds generating it inline.
    """
    from src.core.database import AsyncSessionLocal
    from src.services.data.state_manager import interview_to_state
    from src.services.orchestrator.plan_generator import generate_interview_plan
    from src.services.orchestrator.llm_helpers import LLMHelper
    from src.core.config import settings

    try:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(Interview).where(Interview.id == interview_id)
            )
            interview = result.scalar_one_or_none()
            if interview is None:
                logger.warning(
                    "_prewarm_plan_background: interview %s not found", interview_id
                )
                return

            runtime_ctx = get_runtime_context(interview_id)
            resume_structured = runtime_ctx.get("resume_structured") or {}
            job_desc = runtime_ctx.get("job_description")
            # Skip if plan already exists in runtime context.
            if runtime_ctx.get("interview_plan"):
                logger.info(
                    "_prewarm_plan_background: plan already exists for interview %s — skipping",
                    interview_id,
                )
                return

            logger.info(
                "_prewarm_plan_background: generating plan for interview %s (resume_id=%s, has_jd=%s)",
                interview_id, interview.resume_id, bool(job_desc),
            )

            # Build state from DB (this reads wizard context from resume_context _-keys
            # and from runtime_ctx — both paths work regardless of which process we're in).
            state = interview_to_state(interview)
            state["resume_structured"] = resume_structured
            state["job_description"] = job_desc
            openai_client = settings.get_azure_openai_client()
            llm_helper = LLMHelper(openai_client)
            plan = await generate_interview_plan(state, llm_helper)

            if not plan:
                logger.warning(
                    "_prewarm_plan_background: plan generation returned nothing for interview %s",
                    interview_id,
                )
                return

            # Preserve all existing runtime_ctx keys (wizard selections, etc.) when
            # storing the prewarm plan — do not wipe skills/mode/type.
            set_runtime_context(
                interview_id,
                resume_structured,
                job_desc,
                interview_plan=plan,
                seniority_level=plan.get("seniority_level"),
                topic_iterations={
                    t["id"]: {"iterations_done": 0, "last_quality_score": None}
                    for t in plan.get("topics", [])
                },
                # Re-pass wizard selections so set_runtime_context merge preserves them
                user_skills=state.get("user_skills"),
                forced_interview_mode=state.get("forced_interview_mode"),
                user_interview_type=state.get("user_interview_type"),
            )

            # Persist plan to DB so the LiveKit agent process (separate process,
            # empty runtime_context) can read it back via db_ctx["_interview_plan"].
            # Without this the agent re-generates the plan without the JD and
            # ends up asking irrelevant generic questions.
            try:
                from sqlalchemy.orm.attributes import flag_modified as _flag_modified
                existing_rc = interview.resume_context or {}
                existing_rc["_interview_plan"] = plan
                interview.resume_context = existing_rc
                _flag_modified(interview, "resume_context")
                await db.commit()
                logger.info(
                    "_prewarm_plan_background: plan persisted to DB for interview %s",
                    interview_id,
                )
            except Exception as persist_err:
                logger.warning(
                    "_prewarm_plan_background: failed to persist plan to DB for interview %s (non-fatal): %s",
                    interview_id, persist_err,
                )
                
            logger.info(
                "_prewarm_plan_background: plan cached for interview %s "
                "(%d topics, seniority=%s, fit=%s)",
                interview_id,
                len(plan.get("topics", [])),
                plan.get("seniority_level", "?"),
                plan.get("job_fit", {}).get("fit_verdict", "unknown"),
            )

    except Exception as exc:
        logger.warning(
            "_prewarm_plan_background: failed for interview %s (non-fatal): %s",
            interview_id, exc, exc_info=True,
        )


@router.post("/start", response_model=InterviewResponse)
async def start_interview(
    data: InterviewStart,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Start an interview session - marks interview as in_progress.

    NOTE: Greeting is now handled automatically by LangGraph when the agent connects.
    This endpoint just marks the interview as ready. The agent will execute LangGraph
    on first turn, which will automatically route to greeting node.
    """
    result = await db.execute(
        select(Interview).where(
            Interview.id == data.interview_id, Interview.user_id == user.id
        )
    )
    interview = result.scalar_one_or_none()

    if not interview:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Interview not found",
        )

    if interview.status != "pending":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Interview is already {interview.status}",
        )

    # Simply mark interview as in_progress
    # LangGraph will handle greeting automatically when agent executes first turn
    interview.status = "in_progress"
    interview.started_at = datetime.utcnow()

    await db.commit()
    await db.refresh(interview)

    # Log resume availability at start time so it's easy to verify in logs
    runtime_ctx = get_runtime_context(interview.id)
    resume_ctx = runtime_ctx.get("resume_structured") or {}
    resume_fields = {k: v for k, v in resume_ctx.items() if not str(k).startswith("_")}
    populated = [k for k, v in resume_fields.items() if v]
    if interview.resume_id and populated:
        logger.info(
            "[RESUME] start_interview: interview=%s has resume_id=%s with populated fields: %s | "
            "has_job_desc=%s — AI WILL have resume context",
            interview.id, interview.resume_id, populated, bool(runtime_ctx.get("job_description")),
        )
    elif interview.resume_id and not populated:
        logger.warning(
            "[RESUME] start_interview: interview=%s has resume_id=%s but ALL resume fields are empty — "
            "AI will NOT have resume context. Resume analysis may have failed.",
            interview.id, interview.resume_id,
        )
    else:
        logger.info(
            "[RESUME] start_interview: interview=%s has NO resume — "
            "AI will interview without resume context (has_job_desc=%s)",
            interview.id, bool(runtime_ctx.get("job_description")),
        )

    logger.info(
        "Interview %s marked as in_progress. "
        "LangGraph will handle greeting automatically on first turn.",
        interview.id,
    )

    # Kick off plan pre-generation in the background.
    # This runs 30-120 seconds before the voice agent joins, so by the time
    # generate_reply() fires for the greeting, the plan is already in the DB
    # and plan_interview_node returns immediately (~0s instead of ~60s).
    background_tasks.add_task(_prewarm_plan_background, interview.id)
    logger.info(
        "Interview %s: plan pre-warm background task queued",
        interview.id,
    )

    return _interview_to_response(interview)


@router.post("/respond", response_model=InterviewResponse)
async def respond_to_interview(
    data: InterviewRespond,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Submit a response to the interview - runs adaptive conversation flow."""
    result = await db.execute(
        select(Interview).where(
            Interview.id == data.interview_id, Interview.user_id == user.id
        )
    )
    interview = result.scalar_one_or_none()

    if not interview:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Interview not found",
        )

    if interview.status != "in_progress":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Interview is not in progress",
        )

    try:
        orchestrator = get_or_create_orchestrator(interview.id)
        orchestrator.set_db_session(db)

        # Convert interview to state (pass user for name extraction)
        # On subsequent turns the orchestrator uses its MemorySaver checkpoint,
        # so this only fully matters on the first turn per orchestrator instance.
        state = interview_to_state(interview, user=user)

        # Execute graph with user response
        state = await orchestrator.execute_step(state, user_response=data.message)

        # Persist conversation + counts to DB (plan lives in LangGraph checkpoint)
        state_to_interview(state, interview)

        # Check if interview should be closed (closing_node sets should_close=True)
        if state.get("should_close"):
            interview.status = "completed"
            interview.completed_at = datetime.utcnow()
            try:
                await orchestrator.cleanup_interview(interview.id)
            except Exception as e:
                logger.warning(
                    f"Failed to cleanup interview {interview.id}: {e}", exc_info=True)

        await db.commit()
        await db.refresh(interview)

        return _interview_to_response(interview, state)

    except Exception as e:
        logger.error(
            f"Failed to process interview response: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process response",
        )


@router.post("/complete", response_model=InterviewResponse)
async def complete_interview(
    data: InterviewComplete,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Complete an interview session - marks interview as completed instantly.

    Fast path: just flips status to 'completed' in the DB and evicts the
    orchestrator. Feedback is generated lazily by GET /{id}/feedback when
    the results page loads — no need to run evaluation + closing nodes here
    (those were adding 6-12 seconds of unnecessary blocking LLM calls).
    """
    result = await db.execute(
        select(Interview).where(
            Interview.id == data.interview_id, Interview.user_id == user.id
        )
    )
    interview = result.scalar_one_or_none()

    if not interview:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Interview not found",
        )

    if interview.status == "completed":
        return _interview_to_response(interview)

    if interview.status != "in_progress":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Interview is not in progress (current status: {interview.status})",
        )

    # Fast path: mark as completed directly — no LangGraph, no LLM calls.
    # Feedback generation happens lazily when GET /{id}/feedback is called
    # by the results page, so it doesn't block the user from leaving.
    try:
        interview.status = "completed"
        interview.completed_at = datetime.utcnow()
        await db.commit()
        await db.refresh(interview)
        logger.info(
            "Interview %s marked as completed (fast path).",
            interview.id,
        )
        return _interview_to_response(interview)
    except Exception as e:
        logger.error(f"Failed to complete interview: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to complete interview",
        )
    finally:
        # Evict orchestrator and runtime context to free memory regardless of outcome
        try:
            orchestrator = get_or_create_orchestrator(interview.id)
            await orchestrator.cleanup_interview(interview.id)
        except Exception as cleanup_err:
            logger.warning(f"Cleanup after complete failed (non-fatal): {cleanup_err}")
        finally:
            pop_runtime_context(interview.id)


@router.post("/submit-code", response_model=InterviewResponse)
async def submit_code_to_interview(
    data: InterviewSubmitCode,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Submit code for execution and review during an interview."""
    result = await db.execute(
        select(Interview).where(
            Interview.id == data.interview_id, Interview.user_id == user.id
        )
    )
    interview = result.scalar_one_or_none()

    if not interview:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Interview not found",
        )

    if interview.status != "in_progress":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Interview is not in progress",
        )

    try:
        orchestrator = get_or_create_orchestrator(interview.id)
        orchestrator.set_db_session(db)
        state = interview_to_state(interview, user=user)
        state = await orchestrator.execute_step(
            state, code=data.code, language=data.language
        )

        # Update interview from state
        state_to_interview(state, interview)

        await db.commit()
        await db.refresh(interview)

        return _interview_to_response(interview, state)

    except Exception as e:
        logger.error(f"Failed to process code submission: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process code submission",
        )


@router.put("/{interview_id}/sandbox/code")
async def update_sandbox_code(
    interview_id: int,
    data: SandboxCodeUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Update current sandbox code (for polling).

    Frontend calls this periodically to update the agent's view of current code.
    This enables real-time interaction like a real interviewer watching over your shoulder.
    """
    code = data.code
    result = await db.execute(
        select(Interview).where(
            Interview.id == interview_id, Interview.user_id == user.id
        )
    )
    interview = result.scalar_one_or_none()

    if not interview:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Interview not found",
        )

    if interview.status != "in_progress":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Interview is not in progress",
        )

    try:
        orchestrator = get_or_create_orchestrator(interview_id)
        orchestrator.set_db_session(db)
        state = interview_to_state(interview, user=user)

        # Update state with current code for polling
        state["current_code"] = code
        if "sandbox" not in state:
            state["sandbox"] = {}
        state["sandbox"]["is_active"] = True
        state["sandbox"]["last_activity_ts"] = datetime.utcnow().timestamp()

        # Get node handler to call helper method (now returns updates, no mutations)
        node_handler = orchestrator._get_node_handler()
        updates = await node_handler.check_sandbox_code_changes(state)

        # Merge updates into state
        state = {**state, **updates}

        # Update interview from state
        state_to_interview(state, interview)
        await db.commit()

        return {"status": "updated", "has_guidance": state.get("next_message") is not None}

    except Exception as e:
        logger.error(f"Failed to update sandbox code: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update sandbox code",
        )


@router.get("/{interview_id}/feedback")
async def get_interview_feedback(
    interview_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get comprehensive feedback for a completed interview."""
    result = await db.execute(
        select(Interview).where(
            Interview.id == interview_id, Interview.user_id == user.id
        )
    )
    interview = result.scalar_one_or_none()

    if not interview:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Interview not found",
        )

    if interview.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Interview must be completed to generate feedback",
        )

    try:
        # Use existing feedback if available
        if interview.feedback and isinstance(interview.feedback, dict):
            # Check if it's comprehensive feedback (has overall_score)
            if "overall_score" in interview.feedback:
                return interview.feedback

        # Generate new comprehensive feedback
        feedback_generator = FeedbackGenerator()

        # Extract code submissions from conversation history
        code_submissions = []
        if interview.conversation_history:
            for msg in interview.conversation_history:
                if msg.get("metadata", {}).get("type") == "code_review":
                    code_submissions.append({
                        "code": msg.get("metadata", {}).get("code", ""),
                        "code_quality": msg.get("metadata", {}).get("code_quality", {}),
                        "execution_result": msg.get("metadata", {}).get("execution_result", {}),
                    })

        # Pull job_description and job_fit from the DB-persisted context
        # (stored under _-prefixed keys in resume_context during create_interview)
        db_ctx = interview.resume_context or {}
        job_description = db_ctx.get("_job_description")
        job_fit = db_ctx.get("_job_fit")

        feedback = await feedback_generator.generate_feedback(
            conversation_history=interview.conversation_history or [],
            resume_context=interview.resume_context,
            code_submissions=code_submissions,
            topics_covered=interview.feedback.get(
                "topics_covered", []) if interview.feedback else [],
            job_description=job_description,
            job_fit=job_fit,
            evaluation_config=db_ctx.get("_evaluation_config"),
        )

        # Update interview with comprehensive feedback
        interview.feedback = feedback.model_dump()
        await db.commit()

        return feedback.model_dump()

    except Exception as e:
        logger.error(f"Failed to generate feedback: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate feedback",
        )


@router.get("/analytics/user")
async def get_user_analytics(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get analytics for the current user."""
    try:
        analytics_service = InterviewAnalytics()
        analytics = await analytics_service.get_user_analytics(user.id, db)
        return analytics
    except Exception as e:
        logger.error(f"Failed to get user analytics: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get analytics",
        )


@router.get("/analytics/skills/progression")
async def get_skill_progression(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get skill progression over time for charts."""
    try:
        analytics_service = InterviewAnalytics()
        progression = await analytics_service.get_skill_progression(user.id, db)
        return progression
    except Exception as e:
        logger.error(f"Failed to get skill progression: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get skill progression",
        )


@router.get("/analytics/skills/averages")
async def get_skill_averages(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get average skill scores across all completed interviews."""
    try:
        analytics_service = InterviewAnalytics()
        averages = await analytics_service.get_skill_averages(user.id, db)
        return averages
    except Exception as e:
        logger.error(f"Failed to get skill averages: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get skill averages",
        )


@router.get("/analytics/skills/compare")
async def compare_interview_skills(
    interview_ids: str = Query(...,
                               description="Comma-separated list of interview IDs"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Compare skills across multiple interviews."""
    try:
        # Parse comma-separated interview IDs
        interview_id_list = [int(id.strip()) for id in interview_ids.split(
            ",") if id.strip().isdigit()]

        if not interview_id_list:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Please provide valid interview IDs (comma-separated)",
            )

        # Verify all interviews belong to user
        result = await db.execute(
            select(Interview).where(
                Interview.id.in_(interview_id_list),
                Interview.user_id == user.id
            )
        )
        interviews = result.scalars().all()

        if len(interviews) != len(interview_id_list):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="One or more interviews not found",
            )

        analytics_service = InterviewAnalytics()
        comparison = await analytics_service.get_skill_comparison(interview_id_list, db)

        # Add interview metadata
        return {
            "comparison": comparison,
            "interviews": [
                {
                    "id": i.id,
                    "title": i.title,
                    "completed_at": i.completed_at.isoformat() if i.completed_at else None,
                }
                for i in interviews
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to compare interview skills: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to compare interview skills",
        )


@router.get("/{interview_id}/skills")
async def get_interview_skill_breakdown(
    interview_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get detailed skill breakdown for a specific interview."""
    result = await db.execute(
        select(Interview).where(
            Interview.id == interview_id,
            Interview.user_id == user.id
        )
    )
    interview = result.scalar_one_or_none()

    if not interview:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Interview not found",
        )

    try:
        # If interview is completed but feedback is missing/incomplete, generate it
        # before computing skill breakdown so frontend doesn't see all-zero cards.
        if interview.status == "completed":
            missing_feedback = not interview.feedback or not isinstance(interview.feedback, dict)
            missing_scores = (
                isinstance(interview.feedback, dict)
                and interview.feedback.get("overall_score") is None
            )
            if missing_feedback or missing_scores:
                feedback_generator = FeedbackGenerator()
                db_ctx = interview.resume_context or {}
                generated_feedback = await feedback_generator.generate_feedback(
                    conversation_history=interview.conversation_history or [],
                    resume_context=interview.resume_context,
                    code_submissions=[],
                    topics_covered=db_ctx.get("_topics_covered", []),
                    job_description=db_ctx.get("_job_description"),
                    job_fit=db_ctx.get("_job_fit"),
                    evaluation_config=db_ctx.get("_evaluation_config"),
                )
                interview.feedback = generated_feedback.model_dump()
                await db.commit()

        analytics_service = InterviewAnalytics()
        breakdown = await analytics_service.get_skill_breakdown(interview_id, db)

        return {
            "interview_id": interview.id,
            "interview_title": interview.title,
            "completed_at": interview.completed_at.isoformat() if interview.completed_at else None,
            "skill_breakdown": breakdown,
        }
    except Exception as e:
        logger.error(f"Failed to get skill breakdown: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get skill breakdown",
        )


@router.get("/{interview_id}/insights")
async def get_interview_insights(
    interview_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get detailed insights for a specific interview."""
    result = await db.execute(
        select(Interview).where(
            Interview.id == interview_id, Interview.user_id == user.id
        )
    )
    interview = result.scalar_one_or_none()

    if not interview:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Interview not found",
        )

    try:
        analytics_service = InterviewAnalytics()
        insights = await analytics_service.get_interview_insights(interview_id, db)
        return insights
    except Exception as e:
        logger.error(f"Failed to get interview insights: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get insights",
        )


@router.delete("/{interview_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_interview(
    interview_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete an interview and clean up associated resources."""
    # Find the interview
    result = await db.execute(
        select(Interview).where(
            Interview.id == interview_id,
            Interview.user_id == user.id
        )
    )
    interview = result.scalar_one_or_none()

    if not interview:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Interview not found",
        )

    try:
        # Delete associated LiveKit room if it exists
        room_name = f"interview-{interview_id}"
        try:
            livekit_service = LiveKitService()
            await livekit_service.delete_room(room_name)
            logger.info(f"Deleted LiveKit room: {room_name}")
        except Exception as e:
            # Don't fail if room doesn't exist or can't be deleted
            logger.warning(f"Could not delete LiveKit room {room_name}: {e}")

        # Delete the interview from database
        await db.delete(interview)
        await db.commit()
        pop_runtime_context(interview_id)

        logger.info(f"Deleted interview {interview_id} for user {user.id}")
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    except Exception as e:
        logger.error(
            f"Failed to delete interview {interview_id}: {e}", exc_info=True)
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete interview",
        )


def _interview_to_response(interview: Interview, state: dict | None = None) -> InterviewResponse:
    """Convert Interview model to InterviewResponse schema.

    Args:
        interview: Interview model
        state: Optional InterviewState dict to extract sandbox from
    """
    # Get current message from last assistant message in conversation_history
    current_message = None
    if interview.conversation_history:
        for msg in reversed(interview.conversation_history):
            if msg.get("role") == "assistant":
                current_message = msg.get("content")
                break

    runtime_ctx = get_runtime_context(interview.id)
    db_ctx = interview.resume_context or {}
    interview_plan = state.get("interview_plan") if state else runtime_ctx.get("interview_plan")

    # Extract sandbox state if available — check state, then runtime_ctx, then DB
    # (DB fallback is essential for GET polling from a different process than the agent).
    sandbox_state = None
    if state and "sandbox" in state:
        sandbox_state = state["sandbox"]
    else:
        sandbox_state = runtime_ctx.get("sandbox") or db_ctx.get("_sandbox")

    # Read show_code_editor from state → runtime_ctx → DB (cross-process fallback).
    show_code_editor = False
    if state:
        show_code_editor = bool(state.get("show_code_editor", False))
    else:
        show_code_editor = bool(
            runtime_ctx.get("show_code_editor", False)
            or db_ctx.get("_show_code_editor", False)
        )

    # ── Runtime debug fields (populated only when LangGraph state is available) ──
    if state:
        interview_mode           = state.get("interview_mode")
        current_topic_id         = state.get("current_topic_id")
        seniority_level          = state.get("seniority_level")
        next_node                = state.get("next_node")
        last_node                = state.get("last_node")
        answer_quality           = state.get("answer_quality")
        answer_quality_cache     = state.get("answer_quality_cache")
        performance_signal       = state.get("performance_signal")
        performance_signal_hist  = state.get("performance_signal_history")
        tone_adjustment          = state.get("tone_adjustment")
        pending_question_status  = state.get("pending_question_status")
        pending_question_text    = state.get("pending_question_text")
        pending_topic_id         = state.get("pending_topic_id")
        pending_bank_index       = state.get("pending_bank_index")
        off_topic_streak         = state.get("off_topic_streak")
    else:
        interview_mode           = runtime_ctx.get("interview_mode")
        current_topic_id         = runtime_ctx.get("current_topic_id")
        seniority_level          = runtime_ctx.get("seniority_level")
        next_node = last_node = answer_quality = answer_quality_cache = None
        performance_signal = performance_signal_hist = tone_adjustment = None
        pending_question_status = pending_question_text = pending_topic_id = None
        pending_bank_index = off_topic_streak = None

    # Wizard selections: runtime_ctx (same-process fast path) → DB (cross-process fallback)
    sp_mode = runtime_ctx.get("forced_interview_mode") or db_ctx.get("_forced_interview_mode")
    sp_type = (
        runtime_ctx.get("user_interview_type")
        or db_ctx.get("_user_interview_type")
        or db_ctx.get("_interview_type")
    )
    sp_skills = (
        runtime_ctx.get("user_skills")
        if runtime_ctx.get("user_skills") is not None
        else db_ctx.get("_user_skills")
    )
    session_preferences = None
    if sp_mode is not None or sp_type is not None or sp_skills is not None:
        session_preferences = {
            "interview_mode": sp_mode,
            "interview_type": sp_type,
            "skills": sp_skills,
            "difficulty_mode": getattr(interview, "difficulty_mode", "medium") or "medium",
        }

    return InterviewResponse(
        id=interview.id,
        user_id=interview.user_id,
        resume_id=interview.resume_id,
        title=interview.title,
        status=interview.status,
        conversation_history=interview.conversation_history,
        resume_context=None,
        interview_plan=interview_plan,
        feedback=interview.feedback,
        turn_count=interview.turn_count,
        current_message=current_message,
        sandbox=sandbox_state,
        difficulty_mode=getattr(interview, "difficulty_mode", "medium") or "medium",
        session_preferences=session_preferences,
        show_code_editor=show_code_editor,
        started_at=interview.started_at.isoformat() if interview.started_at else None,
        completed_at=interview.completed_at.isoformat() if interview.completed_at else None,
        created_at=interview.created_at.isoformat(),
        updated_at=interview.updated_at.isoformat(),
        # Runtime debug fields
        interview_mode=interview_mode,
        current_topic_id=current_topic_id,
        seniority_level=seniority_level,
        next_node=next_node,
        last_node=last_node,
        answer_quality=answer_quality,
        answer_quality_cache=answer_quality_cache,
        performance_signal=performance_signal,
        performance_signal_history=performance_signal_hist,
        tone_adjustment=tone_adjustment,
        pending_question_status=pending_question_status,
        pending_question_text=pending_question_text,
        pending_topic_id=pending_topic_id,
        pending_bank_index=pending_bank_index,
        off_topic_streak=off_topic_streak,
    )