"""Almamlaka TV | Digital Expansion Initiative document assistant (Streamlit)."""
import os

import streamlit as st

from rag import config
from rag.branding import find_logo, get_palette, logo_data_uri
from rag.pipeline import answer_question
from rag.textutils import escape_md

LOGO = find_logo()
B = get_palette(LOGO)

icon = "\U0001F4FA"
if LOGO:
    try:
        from PIL import Image

        icon = Image.open(LOGO)
    except Exception:
        pass
st.set_page_config(page_title=f"{B['name_en']} | DEI Assistant", page_icon=icon, layout="centered")

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;500;600&display=swap');
.stApp, .stMarkdown, .stChatInput textarea, .stButton button p {{
    font-family: 'IBM Plex Sans Arabic', 'Segoe UI', sans-serif;
}}
.stApp {{ background: {B['bg']}; }}
[data-testid="stSidebar"] {{ background: {B['panel']}; border-right: 1px solid {B['gold']}33; }}
.brand {{ display:flex; align-items:center; gap:1rem; padding:.4rem 0 1rem 0;
         border-bottom: 2px solid {B['gold']}; margin-bottom:1.2rem; flex-wrap:wrap; }}
.brand-ar {{ font-size:2.6rem; font-weight:600; color:{B['gold']}; line-height:1; }}
.brand-en {{ font-size:1.15rem; font-weight:500; color:{B['ivory']}; }}
.brand-tag {{ font-size:.9rem; color:{B['ivory']}b3; }}
.logo-plate {{ background:#FFFFFF; border-radius:8px; padding:6px 14px; display:inline-flex; }}
.logo-plate img {{ height:52px; width:auto; display:block; }}
.conflict {{ border-left: 4px solid {B['alert']}; background:{B['alert']}1f; color:{B['ivory']};
            padding:.6rem .9rem; border-radius:4px; margin-bottom:.6rem; font-size:.95rem; }}
.rtl {{ direction: rtl; text-align: right; }}
[data-testid="stChatMessage"] {{ background: {B['panel']}; border-radius: 10px; }}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- header
tag = f"{B['tagline_en']} / {B['tagline_ar']}"
if LOGO:
    st.markdown(
        f'<div class="brand"><span class="logo-plate"><img src="{logo_data_uri(LOGO)}" alt="{B["name_en"]}"></span>'
        f'<span class="brand-tag">{tag}</span></div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"""<div class="brand"><span class="brand-ar">{B['name_ar']}</span>
        <span><span class="brand-en">{B['name_en']}</span><br>
        <span class="brand-tag">{tag}</span></span></div>""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- resources
def _required_keys():
    keys = set()
    if config.LLM_PROVIDER == "openai" or config.EMBED_PROVIDER == "openai":
        keys.add("OPENAI_API_KEY")
    if config.LLM_PROVIDER == "anthropic":
        keys.add("ANTHROPIC_API_KEY")
    return sorted(keys)


def _api_keys_present() -> bool:
    for k in _required_keys():
        if os.getenv(k):
            continue
        try:
            os.environ[k] = st.secrets[k]
        except Exception:
            return False
    return True


@st.cache_resource(show_spinner="Loading the knowledge base...")
def load_resources():
    from rag.llm import LLM
    from rag.retrieval import Retriever

    return Retriever(), LLM()


if not _api_keys_present():
    st.error(f"Missing API key: set {', '.join(_required_keys())} as an environment variable or secret.")
    st.stop()
retriever, llm = load_resources()

# ---------------------------------------------------------------- sidebar
EXAMPLES = [
    "Who is the project lead?",
    "What is the total budget?",
    "When is the launch date?",
    "Which risks have been flagged?",
    "ما هي أهداف المشروع؟",
    "من هم أعضاء الفريق؟",
]
queued = None
with st.sidebar:
    st.markdown("**Ask in English or Arabic**  \nاسأل بالعربية أو الإنجليزية")
    st.caption(
        "Answers come only from the three Digital Expansion Initiative documents, "
        "with the document, page and section cited."
    )
    for q in EXAMPLES:
        if st.button(q, use_container_width=True):
            queued = q
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------- chat
if "messages" not in st.session_state:
    st.session_state.messages = []


def render_assistant(m: dict):
    if m.get("conflict"):
        st.markdown(
            f'<div class="conflict">{config.MESSAGES["conflict_banner"][m["lang"]]}</div>',
            unsafe_allow_html=True,
        )
    body = escape_md(m["text"])  # keep "$" visible (Streamlit would treat $...$ as LaTeX)
    if m["lang"] == "ar":
        st.markdown(f'<div class="rtl">\n\n{body.replace("<", "&lt;")}\n\n</div>', unsafe_allow_html=True)
    else:
        st.markdown(body)
    if m.get("sources"):
        title = "المصادر" if m["lang"] == "ar" else "Sources"
        with st.expander(f"{title} ({len(m['sources'])})", expanded=True):
            for i, s in enumerate(m["sources"], 1):
                st.markdown(f"**[{i}] {escape_md(s['label'])}**  \n`{s['file']}`")
                st.caption(escape_md(s["excerpt"]).replace("\n", "  \n"))


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m["role"] == "user":
            st.markdown(escape_md(m["text"]))
        else:
            render_assistant(m)

prompt = st.chat_input("Ask about the documents... / اسأل عن المستندات...") or queued
if prompt and sum(m["role"] == "user" for m in st.session_state.messages) >= config.MAX_QUESTIONS_PER_SESSION:
    st.warning(config.MESSAGES["limit"]["en"] + "  \n" + config.MESSAGES["limit"]["ar"])
    prompt = None
if prompt:
    history = [{"role": m["role"], "content": m["text"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.markdown(escape_md(prompt))
    with st.chat_message("assistant"):
        with st.spinner("Searching the documents..."):
            ans = answer_question(prompt, history, retriever, llm)
        msg = {
            "role": "assistant",
            "text": ans.text,
            "lang": ans.lang,
            "conflict": ans.conflict,
            "sources": ans.sources,
        }
        render_assistant(msg)
    st.session_state.messages.append(msg)
