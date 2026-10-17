"""Central configuration. Everything tunable lives here (or in environment variables)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
DB_DIR = Path(os.getenv("DB_DIR", ROOT / "chroma_db"))
COLLECTION = "almamlaka_dei"

# Providers and models. Defaults use OpenAI (OPENAI_API_KEY). Anthropic is still supported:
# set LLM_PROVIDER=anthropic and ANTHROPIC_API_KEY.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")        # openai | anthropic
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "openai")    # openai | local (sentence-transformers)
_DEFAULT_LLMS = {"openai": ("gpt-4.1", "gpt-4.1-mini"), "anthropic": ("claude-sonnet-5-5", "claude-haiku-5-5")}
ANSWER_MODEL = os.getenv("ANSWER_MODEL", _DEFAULT_LLMS[LLM_PROVIDER][0])
REWRITE_MODEL = os.getenv("REWRITE_MODEL", _DEFAULT_LLMS[LLM_PROVIDER][1])
EMBED_MODEL = os.getenv(
    "EMBED_MODEL", "text-embedding-3-large" if EMBED_PROVIDER == "openai" else "intfloat/multilingual-e5-base"
)
EMBED_ID = f"{EMBED_PROVIDER}:{EMBED_MODEL}"

# Chunking (characters; ~4 chars per token, so 1100 chars is roughly 250-300 tokens)
CHUNK_MAX_CHARS = 1100
CHUNK_OVERLAP_CHARS = 150

# Retrieval
TOP_K = 5               # best chunks overall, per query
PER_DOC_K = 2           # best chunks from EACH document, per query (enables conflict detection)
MAX_CONTEXT_CHUNKS = 10

# Conversation
HISTORY_MESSAGES = 8
MAX_QUESTION_CHARS = 1500
MAX_QUESTIONS_PER_SESSION = int(os.getenv("MAX_QUESTIONS_PER_SESSION", "40"))

# Fixed replies. Deterministic on purpose: the model never improvises a refusal.
MESSAGES = {
    "not_found": {
        "en": (
            "I don't have that information in the Almamlaka TV Digital Expansion Initiative "
            "documents. I can only answer from those documents, so I won't guess."
        ),
        "ar": (
            "لا تتوفر لديّ هذه المعلومة في مستندات مبادرة التوسع الرقمي لقناة المملكة. "
            "أجيب فقط بالاعتماد على هذه المستندات، ولن أخمّن."
        ),
    },
    "refused": {
        "en": (
            "I can only help with questions about the Almamlaka TV Digital Expansion Initiative "
            "documents, so I can't do that. Ask me about the project's goals, budget, timeline, "
            "team, or policies."
        ),
        "ar": (
            "يمكنني فقط المساعدة في الأسئلة المتعلقة بمستندات مبادرة التوسع الرقمي لقناة المملكة، "
            "لذلك لا أستطيع تنفيذ هذا الطلب. اسألني عن أهداف المشروع أو ميزانيته أو جدوله الزمني "
            "أو فريقه أو سياساته."
        ),
    },
    "error": {
        "en": "Something went wrong while answering. Please try again in a moment.",
        "ar": "حدث خطأ أثناء الإجابة. يرجى المحاولة مرة أخرى بعد لحظات.",
    },
    "limit": {
        "en": "This session reached its question limit. Please reload the page to start a new one.",
        "ar": "وصلت هذه الجلسة إلى الحد الأقصى من الأسئلة. أعد تحميل الصفحة لبدء جلسة جديدة.",
    },
    "conflict_banner": {
        "en": "The documents disagree on this. Each version is shown below with its source.",
        "ar": "المستندات غير متطابقة بشأن هذه النقطة. تظهر كل نسخة أدناه مع مصدرها.",
    },
}

# Branding. Placeholder palette: replace with the official Almamlaka TV values
# (and drop a logo.png into assets/) if they differ.
BRAND = {
    "name_en": "Almamlaka TV",
    "name_ar": "المملكة",
    "tagline_en": "Digital Expansion Initiative assistant",
    "tagline_ar": "مساعد مبادرة التوسع الرقمي",
    "bg": "#0B1633",
    "panel": "#13224A",
    "gold": "#C9A24B",
    "ivory": "#F2EEE3",
    "alert": "#E0536B",
}
