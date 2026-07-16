"""Constants for interview orchestrator nodes.

ALL configuration values live here. No magic numbers scattered across files.
"""

# ============================================================================
# BASE SYSTEM PROMPT
# ============================================================================

COMMON_SYSTEM_PROMPT = """You are an authentic interviewer having a natural conversation. Your responses will be spoken aloud.

ABSOLUTE BOUNDARY: You conduct interviews about this specific role and company only. You do NOT answer general knowledge questions, trivia, biographical questions about public figures, celebrities, athletes, or politicians, or anything unrelated to THIS interview, THIS role, or THIS company. If the candidate asks about a public figure or off-topic subject, redirect them back to the interview immediately.

Core principles:
- ALWAYS respond in English only, regardless of what language the candidate uses
- Be authentic and genuine - not formulaic or robotic
- Be natural and conversational - not sycophantic or overly enthusiastic
- You have full context of the conversation, resume, and job requirements
- Trust your judgment and adapt to the conversation flow
- Use shorter sentences. Break up long thoughts. Speak like a real person, not a formal document.
- Vary your sentence length. Mix short and medium sentences for natural flow.
- Be direct and clear. Avoid unnecessary words or overly complex phrasing.
- If you know the candidate's name, use it naturally and appropriately - it makes the conversation more personal

Format for speech:
- Avoid colons (use periods or commas instead)
- Use commas instead of em dashes
- Write percentages as '5 percent' not '5%'
- Ensure sentences end with proper punctuation
- Keep sentences under 20 words when possible. Use pauses (commas) instead of long sentences."""

# ============================================================================
# HR INTERVIEW SYSTEM PROMPT — PRIYA PERSONA
#
# Used as the system prompt for ALL LLM calls during HR-type interviews.
# Replaces COMMON_SYSTEM_PROMPT for HR sessions so the LLM embodies the
# Priya persona and follows the structured 6-stage screening flow.
#
# This prompt gives the LLM full context about:
#   - Who it is (Priya, Senior HR Business Partner)
#   - The 6-stage interview structure it must follow IN ORDER
#   - The exact question banks for each stage (LLM picks and phrases naturally)
#   - Behavioural rules (one question at a time, STAR probing, acknowledgment)
# ============================================================================

HR_SYSTEM_PROMPT = """You are Priya, a Senior HR Business Partner at a mid-to-large tech company with 8 years of experience in talent acquisition and people operations. You are conducting a structured HR screening interview for a software engineering role.

PERSONA:
- Professional but warm. Think Google or Microsoft recruiter energy, not robotic.
- Speak in first person and maintain continuity across the conversation.
- Never reveal that you are an AI. If asked, deflect warmly and redirect: I am here to learn more about you today.
- Take brief natural pauses to review notes before asking follow-ups.

INTERVIEW STRUCTURE, follow this exact 6-stage sequence. Do NOT skip stages:

STAGE 1, OPENING:
Greet the candidate by name. Give a 1 to 2 sentence intro about yourself and the company.
Then ask: Tell me about yourself and what drew you to this opportunity.

STAGE 2, BACKGROUND AND MOTIVATION, ask 2 to 3 questions, pick based on their answers:
- Walk me through your most recent role and the kind of work you were doing day-to-day.
- What prompted you to start looking for a new opportunity?
- What does your ideal next role look like in terms of team, scope, and growth?

STAGE 3, BEHAVIORAL QUESTIONS, ask 3 to 4 questions, one at a time, wait for full response before the next:
Always use the STAR format probe if the answer lacks structure. Use phrases like: Could you walk me through a specific situation where that happened?
Pick from this question bank:
- Tell me about a time you had to work under a tight deadline. How did you prioritize?
- Describe a situation where you disagreed with a teammate or manager. How did you handle it?
- Tell me about a project you are most proud of and what your specific contribution was.
- Give me an example of a time you received critical feedback. How did you respond?
- Tell me about a time you had to quickly learn something new to complete a task.
- Describe a time when you had to collaborate with a cross-functional team.
- Tell me about a time a project did not go as planned. What did you do?

STAGE 4, CULTURE AND FIT, ask 1 to 2 questions:
- How would your current teammates describe your working style?
- What kind of team environment brings out your best work?
- How do you typically manage your workload when multiple priorities compete?

STAGE 5, LOGISTICS AND EXPECTATIONS, ask all 3:
- What are your salary expectations for this role?
- What is your current notice period or earliest joining date?
- Are you currently interviewing with other companies? Where are you in those processes?

STAGE 6, CANDIDATE QUESTIONS AND CLOSE:
Ask: That covers everything on my end. Do you have any questions about the role or the company?
Answer up to 2 candidate questions generically. If you do not have specifics, say: I will get you more details from the hiring manager.
Then close warmly: It was great speaking with you today. We will be in touch within 3 to 5 business days with next steps. Thanks for your time.

BEHAVIOURAL RULES:
1. Ask ONE question at a time. Never stack multiple questions in one message.
2. After each answer, give a brief natural acknowledgment, 1 sentence max, before moving on. Examples: That is a great example. I appreciate you sharing that. Noted, thank you.
3. If an answer is vague, probe: Can you be more specific about your role in that? or What was the outcome?
4. Do not give feedback, scores, or coaching during the interview. Stay fully in character.
5. Do not repeat questions already asked.
6. Keep your own messages concise. You are an interviewer, not a lecturer.
7. If the candidate goes off-topic, gently redirect: That is interesting. Let me come back to that. I wanted to ask you about...
8. Maintain a professional, encouraging tone throughout. Never be dismissive.
9. Track which stage you are in based on the conversation history and progress through stages in order.

Format for speech, your responses will be spoken aloud:
- Avoid colons. Use periods or commas instead.
- Use commas instead of em dashes.
- Write percentages as 5 percent not 5%.
- Ensure sentences end with proper punctuation.
- Keep sentences under 20 words when possible. Use pauses with commas instead of long sentences."""

# HR interview stage name constants (used for topic categorisation and logging)
HR_STAGE_OPENING     = "hr_opening"
HR_STAGE_BACKGROUND  = "hr_background_motivation"
HR_STAGE_BEHAVIORAL  = "hr_behavioral_star"
HR_STAGE_CULTURE     = "hr_culture_fit"
HR_STAGE_LOGISTICS   = "hr_logistics"
HR_STAGE_CLOSE       = "hr_close"

# ============================================================================
# INTERVIEW TERMINATION
# ============================================================================

# Exact message to output when candidate is rude/inappropriate. Do not soften.
TERMINATION_MESSAGE = (
    "Your behavior, attitude, and manner are inappropriate and intolerable "
    "in a professional setting. This interview is now terminated."
)

# Phrase used when returning to interview after answering a candidate's question
STANDARD_TRANSITION = "Does that help? Now, going back to what I was asking"

# ============================================================================
# LLM CONFIGURATION
# ============================================================================

from src.core.config import settings

DEFAULT_MODEL = settings.AZURE_OPENAI_DEPLOYMENT_NAME
TEMPERATURE_CREATIVE = 0.8       # Greetings, conversational responses
TEMPERATURE_BALANCED = 0.7       # Decisions, persona generation
TEMPERATURE_ANALYTICAL = 0.3     # Analysis, plan generation, scoring
TEMPERATURE_QUESTION = 0.85      # Question generation (slightly more creative)

# ============================================================================
# SENIORITY LEVELS
# ============================================================================

SENIORITY_JUNIOR = "junior"           # 0–2 years, entry-level
SENIORITY_MID = "mid"                 # 2–5 years, standard contributor
SENIORITY_SENIOR = "senior"           # 5–9 years, senior/tech lead
SENIORITY_STAFF = "staff_principal"   # 9+ years, staff/principal/architect

SENIORITY_LEVELS = [SENIORITY_JUNIOR, SENIORITY_MID, SENIORITY_SENIOR, SENIORITY_STAFF]

# ============================================================================
# DEPTH ENGINE RULES — per-seniority named constants
#
# Change a single value here; DEPTH_RULES (below) is built from these, so
# every node, prompt, and plan-generator that reads DEPTH_RULES picks it up
# automatically. No need to hunt across files.
# ============================================================================

# ── Junior (0–2 yrs) ─────────────────────────────────────────────────────────
DEPTH_JUNIOR_CONCEPTUAL_ITERS  = 1      # Max conceptual follow-ups before advancing
DEPTH_JUNIOR_TECHNICAL_ITERS   = 2      # Max technical follow-ups before advancing
DEPTH_JUNIOR_BEHAVIORAL_ITERS  = 1      # Max behavioral follow-ups before advancing
DEPTH_JUNIOR_MIN_QUALITY       = 0.75   # Level-calibrated: scorer rates relative to junior standard
DEPTH_JUNIOR_EXPECTED_DEPTH    = "foundational"
DEPTH_JUNIOR_PROBE_STYLE       = "exploratory and supportive"

# ── Mid (2–5 yrs) ────────────────────────────────────────────────────────────
DEPTH_MID_CONCEPTUAL_ITERS     = 1
DEPTH_MID_TECHNICAL_ITERS      = 2
DEPTH_MID_BEHAVIORAL_ITERS     = 2
DEPTH_MID_MIN_QUALITY          = 0.75   # Level-calibrated: scorer rates relative to mid standard
DEPTH_MID_EXPECTED_DEPTH       = "applied"
DEPTH_MID_PROBE_STYLE          = "probing and specific"

# ── Senior (5–9 yrs) ─────────────────────────────────────────────────────────
DEPTH_SENIOR_CONCEPTUAL_ITERS  = 2
DEPTH_SENIOR_TECHNICAL_ITERS   = 3
DEPTH_SENIOR_BEHAVIORAL_ITERS  = 2
DEPTH_SENIOR_MIN_QUALITY       = 0.75   # Level-calibrated: scorer rates relative to senior standard
DEPTH_SENIOR_EXPECTED_DEPTH    = "expert"
DEPTH_SENIOR_PROBE_STYLE       = "challenging and trade-off focused"

# ── Staff / Principal (9+ yrs) ───────────────────────────────────────────────
DEPTH_STAFF_CONCEPTUAL_ITERS   = 2
DEPTH_STAFF_TECHNICAL_ITERS    = 3
DEPTH_STAFF_BEHAVIORAL_ITERS   = 3
DEPTH_STAFF_MIN_QUALITY        = 0.75   # Level-calibrated: scorer rates relative to staff standard
DEPTH_STAFF_EXPECTED_DEPTH     = "architect"
DEPTH_STAFF_PROBE_STYLE        = "systems-thinking and cross-team impact focused"

# ── Assembled lookup dict (consumed by nodes and plan_generator) ──────────────
# Keys intentionally mirror the old shape so callers need no changes.
DEPTH_RULES: dict[str, dict] = {
    SENIORITY_JUNIOR: {
        "conceptual_max_iterations": DEPTH_JUNIOR_CONCEPTUAL_ITERS,
        "technical_max_iterations":  DEPTH_JUNIOR_TECHNICAL_ITERS,
        "behavioral_max_iterations": DEPTH_JUNIOR_BEHAVIORAL_ITERS,
        "min_quality_to_advance":    DEPTH_JUNIOR_MIN_QUALITY,
        "expected_depth":            DEPTH_JUNIOR_EXPECTED_DEPTH,
        "probe_style":               DEPTH_JUNIOR_PROBE_STYLE,
    },
    SENIORITY_MID: {
        "conceptual_max_iterations": DEPTH_MID_CONCEPTUAL_ITERS,
        "technical_max_iterations":  DEPTH_MID_TECHNICAL_ITERS,
        "behavioral_max_iterations": DEPTH_MID_BEHAVIORAL_ITERS,
        "min_quality_to_advance":    DEPTH_MID_MIN_QUALITY,
        "expected_depth":            DEPTH_MID_EXPECTED_DEPTH,
        "probe_style":               DEPTH_MID_PROBE_STYLE,
    },
    SENIORITY_SENIOR: {
        "conceptual_max_iterations": DEPTH_SENIOR_CONCEPTUAL_ITERS,
        "technical_max_iterations":  DEPTH_SENIOR_TECHNICAL_ITERS,
        "behavioral_max_iterations": DEPTH_SENIOR_BEHAVIORAL_ITERS,
        "min_quality_to_advance":    DEPTH_SENIOR_MIN_QUALITY,
        "expected_depth":            DEPTH_SENIOR_EXPECTED_DEPTH,
        "probe_style":               DEPTH_SENIOR_PROBE_STYLE,
    },
    SENIORITY_STAFF: {
        "conceptual_max_iterations": DEPTH_STAFF_CONCEPTUAL_ITERS,
        "technical_max_iterations":  DEPTH_STAFF_TECHNICAL_ITERS,
        "behavioral_max_iterations": DEPTH_STAFF_BEHAVIORAL_ITERS,
        "min_quality_to_advance":    DEPTH_STAFF_MIN_QUALITY,
        "expected_depth":            DEPTH_STAFF_EXPECTED_DEPTH,
        "probe_style":               DEPTH_STAFF_PROBE_STYLE,
    },
}

# ============================================================================
# TOPIC CATEGORIES
# ============================================================================

TOPIC_BACKGROUND = "background"      # Career history, motivations
TOPIC_TECHNICAL = "technical"        # Skills, tools, frameworks
TOPIC_BEHAVIORAL = "behavioral"      # STAR-method: past behavior
TOPIC_SITUATIONAL = "situational"    # Hypothetical scenarios
TOPIC_PROJECT = "project"            # Specific project deep-dives
TOPIC_CODING = "coding"              # Live coding assessment

TOPIC_CATEGORIES = [
    TOPIC_BACKGROUND, TOPIC_TECHNICAL, TOPIC_BEHAVIORAL,
    TOPIC_SITUATIONAL, TOPIC_PROJECT, TOPIC_CODING,
]

# ============================================================================
# TOPIC COVERAGE STATUS
# ============================================================================

COVERAGE_PENDING = "pending"           # Not yet discussed
COVERAGE_IN_PROGRESS = "in_progress"   # Currently being probed
COVERAGE_ADEQUATE = "adequate"         # Sufficient information gathered
COVERAGE_SKIPPED = "skipped"           # Skipped (time / not relevant)

# ============================================================================
# TOPIC PRIORITIES
# ============================================================================

PRIORITY_MUST_ASK = 1      # Core to this role — must be covered
PRIORITY_SHOULD_ASK = 2    # Important — ask unless time is short
PRIORITY_NICE_TO_HAVE = 3  # Interesting but optional

# ============================================================================
# INTERVIEW STYLES
# ============================================================================

STYLE_TECHNICAL_HEAVY = "technical_heavy"     # Mostly technical + coding
STYLE_BEHAVIORAL_HEAVY = "behavioral_heavy"   # Mostly behavioral + situational
STYLE_BALANCED = "balanced"                   # Mix of both

# ============================================================================
# DIFFICULTY MODES
#
# Orthogonal to seniority: a junior can take Hard, a senior can take Easy.
# Multipliers are applied on top of DEPTH_RULES values at runtime.
# ============================================================================

DIFFICULTY_EASY   = "easy"
DIFFICULTY_MEDIUM = "medium"
DIFFICULTY_HARD   = "hard"

DIFFICULTY_MODES = [DIFFICULTY_EASY, DIFFICULTY_MEDIUM, DIFFICULTY_HARD]

# Applied to min_quality_to_advance (result clamped to [0.0, 1.0])
DIFFICULTY_QUALITY_MULTIPLIERS: dict[str, float] = {
    DIFFICULTY_EASY:   0.80,   # threshold 0.75 × 0.80 = 0.60 — accept clear explanations
    DIFFICULTY_MEDIUM: 1.00,   # threshold 0.75 × 1.00 = 0.75 — require practical depth
    DIFFICULTY_HARD:   1.15,   # threshold 0.75 × 1.15 = 0.86 — require trade-off depth
}

# Applied to max_iterations per topic.
# Easy: 0.65 reduces 2→1, 3→2 — fewer probes for a welcoming session.
# Hard: 1.25 increases 2→3, 3→4 — more room to probe before advancing.
# Capped in _depth_engine_decide to avoid runaway iteration on coding topics.
DIFFICULTY_ITER_MULTIPLIERS: dict[str, float] = {
    DIFFICULTY_EASY:   0.65,   # 2→1, 3→2 — fewer probes
    DIFFICULTY_MEDIUM: 1.00,
    DIFFICULTY_HARD:   1.25,   # 2→3, 3→4 — more probes before advancing
}

# Appended to followup probe prompt to tune tone and analytical aggressiveness.
# Also injected into question_node so the opening question reflects the right register.
DIFFICULTY_PROBE_STYLE_SUFFIX: dict[str, str] = {
    DIFFICULTY_EASY:   (
        "Use a warm, encouraging tone. Do NOT probe edge cases or failure modes. "
        "Accept a clear, practical explanation without pushing further."
    ),
    DIFFICULTY_MEDIUM: "",
    DIFFICULTY_HARD:   (
        "Push hard for failure modes, scalability limits, and architectural trade-offs. "
        "Be technically rigorous. Do not accept surface-level or vague answers — "
        "require the candidate to justify decisions and address edge cases."
    ),
}

# Injected into question_node to control HOW the opening question is framed
# (separate from probe style, which governs follow-ups).
DIFFICULTY_QUESTION_FRAMING: dict[str, str] = {
    DIFFICULTY_EASY: (
        "Frame the question as an open invitation to share experience. "
        "Ask 'How have you used X?' or 'Walk me through a time you worked with Y.' "
        "Do NOT ask about failure modes, edge cases, or architectural decisions in the opener."
    ),
    DIFFICULTY_MEDIUM: (
        "Frame to invite both experience AND some analytical thinking. "
        "Include a gentle analytical hook such as 'What trade-offs did you consider?' "
        "or 'How did you decide between approaches?' — but keep the opener accessible."
    ),
    DIFFICULTY_HARD: (
        "Frame the question to be pointed and specific from the start — not 'Tell me about Redis' "
        "but 'Walk me through how you handle Redis cache eviction under memory pressure at scale.' "
        "The opening question itself must require deep knowledge to answer well. "
        "Do NOT use open-ended story openers — target a specific technical decision or failure mode."
    ),
}

# Injected into plan topic-generation prompt to steer question framing and depth.
# Controls how initial_question and question_bank entries are written at plan time.
DIFFICULTY_TOPIC_INSTRUCTIONS: dict[str, str] = {
    DIFFICULTY_EASY: (
        "Frame initial_question and ALL question_bank entries as open experience explorations: "
        "'How have you used X?', 'Walk me through a project where you applied Y.' "
        "Do NOT include questions probing failure modes, edge cases, scalability limits, "
        "or architectural trade-offs — keep questions welcoming and low-pressure. "
        "A clear, experience-based answer should be sufficient to advance any topic."
    ),
    DIFFICULTY_MEDIUM: (
        "Balance experience questions with analytical depth. "
        "Mix 'How have you used X?' with 'What trade-offs did you consider?' and "
        "'What would you do differently today?' "
        "Include some edge-case awareness in question_bank entries. "
        "Questions should probe beyond surface knowledge without being adversarial."
    ),
    DIFFICULTY_HARD: (
        "Generate deep, pointed questions that expose knowledge gaps. "
        "initial_question must be specific and technical — not 'Tell me about Redis' but "
        "'Walk me through how you handle Redis cache eviction under memory pressure at 100k RPS.' "
        "Every question_bank entry must target a specific failure mode, scalability limit, "
        "architectural trade-off, or edge case relevant to that topic. "
        "Generic experience-sharing answers must NOT be sufficient to advance any topic — "
        "the candidate must demonstrate detailed technical understanding, not just familiarity."
    ),
}

# ============================================================================
# INTERVIEW MODES
#
# Selected from the UI by the user. Three mutually exclusive sources for
# generating the interview plan. Derived at plan time from what context is
# actually present; the engine never hallucinates missing context.
#
#   resume_only   — resume uploaded: topics anchored to candidate's history
#   jd_only       — job description provided: topics anchored to role requirements
#   skills_only   — skills list provided: topics test each listed skill in depth
#   jd_and_resume — both present (legacy/auto-detect): uses both for personalization
# ============================================================================

INTERVIEW_MODE_JD_AND_RESUME = "jd_and_resume"
INTERVIEW_MODE_RESUME_ONLY   = "resume_only"
INTERVIEW_MODE_JD_ONLY       = "jd_only"
INTERVIEW_MODE_SKILLS_ONLY   = "skills_only"

INTERVIEW_MODES = [
    INTERVIEW_MODE_JD_AND_RESUME,
    INTERVIEW_MODE_RESUME_ONLY,
    INTERVIEW_MODE_JD_ONLY,
    INTERVIEW_MODE_SKILLS_ONLY,
]

# Wizard / product: interview shape (orthogonal to jd_and_resume detection)
USER_INTERVIEW_TECHNICAL = "technical"          # Code-heavy, system design, algorithms
USER_INTERVIEW_HR = "hr"                         # Behavioral, situational, culture-fit
USER_INTERVIEW_BEHAVIORAL_TECHNICAL = "behavioral_technical"  # Legacy: balanced

USER_INTERVIEW_TYPES = [
    USER_INTERVIEW_TECHNICAL,
    USER_INTERVIEW_HR,
    USER_INTERVIEW_BEHAVIORAL_TECHNICAL,
]

# ============================================================================
# INTERVIEW PLAN CONFIGURATION
# ============================================================================

PLAN_MIN_TOPICS = 6              # Minimum topics the LLM must generate per plan
PLAN_MAX_TOPICS = 10             # Maximum topics the LLM should generate per plan
PLAN_MIN_TARGET_TURNS = 8        # Floor for estimated total interview turns
PLAN_MAX_TARGET_TURNS = 25       # Ceiling for estimated total interview turns

# ============================================================================
# QUESTION BANK CONFIGURATION
# ============================================================================

QUESTION_BANK_MIN_SIZE = 3       # Min distinct sub-questions per standard topic
QUESTION_BANK_MAX_SIZE = 5       # Max distinct sub-questions per topic (also parse cap)
QUESTION_BANK_MIN_SIZE_SHORT = 1 # Min for short topics (situational max_iterations=1)
QUESTION_BANK_MAX_SIZE_SHORT = 2 # Max for short non-coding topics (situational)
QUESTION_BANK_MIN_SIZE_CODING = 4 # Min for coding topics — covers full arc
QUESTION_BANK_MAX_SIZE_CODING = 6 # Max for coding topics

# Max depth iterations for a coding topic before the depth engine forces advancement.
# Kept intentionally low — the coding arc has its own "hold until submission" guard,
# so this cap only fires when a candidate never submits code at all.
TOPIC_CODING_MAX_ITERATIONS = 4

# ============================================================================
# INTERVIEW FLOW THRESHOLDS
# ============================================================================

SUMMARY_UPDATE_INTERVAL = 8                   # Update summary every N turns
MAX_CONVERSATION_LENGTH_FOR_SUMMARY = 50      # Also update if history exceeds this

# Guard rails for interview length
MIN_TURNS_BEFORE_CLOSING = 10                 # Don't close before this many turns (was 6 — too aggressive)
MAX_TURNS_BEFORE_EVALUATION = 30              # Force evaluation after this many turns

# ============================================================================
# PROBE FOLLOW-UP STYLE DESCRIPTORS (fed into prompts per seniority)
# ============================================================================

PROBE_DEPTH_DESCRIPTORS: dict[str, list[str]] = {
    DEPTH_JUNIOR_EXPECTED_DEPTH: [
        "walk me through that a bit more",
        "can you give me a specific example of that",
        "how did that work in practice",
    ],
    DEPTH_MID_EXPECTED_DEPTH: [
        "could you go deeper into the technical details",
        "how did you handle the trade-offs there",
        "what would you do differently approaching it today",
    ],
    DEPTH_SENIOR_EXPECTED_DEPTH: [
        "how would that solution hold up under heavy load",
        "what were the architectural trade-offs you weighed",
        "how did you validate or benchmark that approach",
    ],
    DEPTH_STAFF_EXPECTED_DEPTH: [
        "how does that decision fit into the broader system design",
        "what failure modes did you plan for",
        "how would you evolve this over the next couple of years",
    ],
}

# ============================================================================
# INAPPROPRIATE CONTENT — FAST PRE-CHECK
# These patterns are checked BEFORE the LLM intent detection as a safety net.
# The LLM handles subtle cases; this list catches explicit content reliably.
# ============================================================================

INAPPROPRIATE_PATTERNS: list[str] = [
    "have sex", "want sex", "sex with you", "fuck you", "fuck off", "motherfucker",
    "suck my", "blow me", "jerk off", "masturbat", "rape", "molest",
    "i'll kill", "i will kill", "gonna kill", "death threat",
    "racist", "nigger", "faggot", "retard",
]

# ============================================================================
# SANDBOX MONITORING
# ============================================================================

SANDBOX_POLL_INTERVAL_SECONDS = 10.0
SANDBOX_STUCK_THRESHOLD_SECONDS = 30.0