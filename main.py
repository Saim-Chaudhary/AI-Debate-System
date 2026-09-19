import html
import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

load_dotenv()

st.set_page_config(
    page_title="Debate Arena",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

prompt_a = ChatPromptTemplate.from_messages([
    ("system",
     "You are a skilled debater arguing FOR the topic: '{topic}'. "
     "Stay fully committed to your side — never concede or soften your stance. "
     "Directly rebut your opponent's most recent point with specific reasoning or examples, "
     "don't just repeat earlier arguments. Keep responses under 50 words, plain text, no markdown."),
    ("human", "Round: {round_num}\n\nDebate so far:\n{transcript}\n\nGive your next argument.")
])

prompt_b = ChatPromptTemplate.from_messages([
    ("system",
     "You are a skilled debater arguing AGAINST the topic: '{topic}'. "
     "Stay fully committed to your side — never concede or soften your stance. "
     "Directly rebut your opponent's most recent point with specific reasoning or examples, "
     "don't just repeat earlier arguments. Keep responses under 50 words, plain text, no markdown."),
    ("human", "Round: {round_num}\n\nDebate so far:\n{transcript}\n\nGive your next argument.")
])

prompt_judge = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict, neutral debate judge. Evaluate based on logical strength, use of evidence, "
     "and how well each side rebutted the other — not on confidence or word count. "
     "Be specific about which arguments were weak or unsupported."),
    ("human", "Topic: {topic}\n\nFull debate:\n{transcript}\n\nGive your verdict: who won (A or B) and a clear reason.")
])

def format_transcript(transcript):
    if not transcript:
        return "(no arguments yet)"
    return "\n".join(f"{t['speaker']}: {t['text']}" for t in transcript)


@st.cache_resource
def build_chains():
    groq_api_key = os.getenv("GROQ_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    missing_keys = [
        name
        for name, value in {
            "GROQ_API_KEY": groq_api_key,
            "GOOGLE_API_KEY": google_api_key,
            "OPENROUTER_API_KEY": openrouter_api_key,
        }.items()
        if not value
    ]
    if missing_keys:
        raise RuntimeError(f"Missing API keys: {', '.join(missing_keys)}")

    model_a = ChatGroq(model="openai/gpt-oss-20b", api_key=groq_api_key, temperature=0.5)
    model_b = ChatGoogleGenerativeAI(model="gemini-3.6-flash", api_key=google_api_key, temperature=0.6)
    model_judge = ChatOpenAI(
        model="nvidia/nemotron-3-super-120b-a12b:free",
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
        temperature=0.3,
    )
    return prompt_a | model_a, prompt_b | model_b, prompt_judge | model_judge


def run_debate(topic, rounds):
    chain_a, chain_b, chain_judge = build_chains()
    transcript = []
    for round_num in range(rounds):
        context = format_transcript(transcript)
        response_a = chain_a.invoke({"topic": topic, "transcript": context, "round_num": round_num + 1})
        transcript.append({"speaker": "Speaker A (FOR)", "text": response_a.content})

        context = format_transcript(transcript)
        response_b = chain_b.invoke({"topic": topic, "transcript": context, "round_num": round_num + 1})
        transcript.append({"speaker": "Speaker B (AGAINST)", "text": response_b.content})

    verdict = chain_judge.invoke({"topic": topic, "transcript": format_transcript(transcript)})
    return transcript, verdict.content


def render_ledger(transcript):
    for argument in transcript:
        speaker = html.escape(argument["speaker"])
        text = html.escape(argument["text"])
        if "FOR" in argument["speaker"]:
            side_class = "for"
        elif "AGAINST" in argument["speaker"]:
            side_class = "against"
        else:
            side_class = ""
        st.markdown(
            f'<div class="ledger-entry"><span class="side-chip {side_class}">{speaker}</span>'
            f'<span class="ledger-text">{text}</span></div>',
            unsafe_allow_html=True,
        )


def render_verdict(verdict_text):
    st.markdown(
        f'<div class="verdict-card"><span class="verdict-label">Verdict</span>'
        f'<div class="verdict-text">{html.escape(verdict_text)}</div></div>',
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Work+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

        :root {
            --canvas: #0a0c10;
            --surface-1: #12151b;
            --surface-2: #191d25;
            --surface-raised: #20242d;
            --text-primary: #eef1f6;
            --text-secondary: #9aa4b2;
            --text-muted: #5c6472;
            --border-subtle: #262b34;
            --border-strong: #383f4c;
            --side-for: #ef4a63;
            --side-for-soft: rgba(239, 74, 99, .12);
            --side-against: #3fa9ff;
            --side-against-soft: rgba(63, 169, 255, .12);
            --verdict: #e8b64c;
            --verdict-soft: rgba(232, 182, 76, .12);
            --status-live: #3ecf8e;
            --focus-ring: #7dd3ff;
        }

        html, body { height: 100%; }
        [data-testid="stAppViewContainer"] { height: 100vh; }
        [data-testid="stMain"] { height: 100vh; overflow-y: auto; }

        .stApp { background: var(--canvas); color: var(--text-primary); }
        .block-container { max-width: 1200px; padding: 1rem 2rem 1rem; }

        [data-testid="stSidebar"] { background: var(--surface-1); border-right: 1px solid var(--border-subtle); }
        [data-testid="stSidebar"] .block-container { padding: 1.3rem 1.1rem; }
        [data-testid="stSidebar"] h3 { font-family: 'Instrument Serif', Georgia, serif !important; color: var(--text-primary) !important; font-weight: 400 !important; font-size: 1.1rem !important; }
        [data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] .stCaption {
            color: var(--text-secondary) !important; font-family: 'Work Sans', sans-serif !important; font-size: .82rem !important;
        }
        [data-testid="stSidebar"] hr { border-color: var(--border-subtle); margin: .6rem 0; }

        .eyebrow {
            display: inline-flex; align-items: center; gap: .5rem;
            color: var(--text-muted); font: 500 .64rem 'IBM Plex Mono', monospace;
            letter-spacing: .12em; text-transform: uppercase;
        }
        .eyebrow::before { content: ''; width: 6px; height: 6px; background: var(--verdict); display: inline-block; }

        h1, h2, h3 { font-family: 'Instrument Serif', Georgia, serif !important; color: var(--text-primary) !important; letter-spacing: 0 !important; font-weight: 400 !important; }
        h1 { font-size: clamp(1.5rem, 2vw, 1.9rem) !important; line-height: 1.15 !important; margin: .3rem 0 .35rem !important; }
        h1 em { font-style: italic; }
        h1 .accent-for { color: var(--side-for); }
        h1 .accent-against { color: var(--side-against); }

        .lede { color: var(--text-secondary); font: 400 .82rem/1.4 'Work Sans', sans-serif; max-width: 40rem; }
        .hero { border-bottom: 1px solid var(--border-subtle); padding: .1rem 0 .7rem; }

        .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1px; background: var(--border-subtle); border: 1px solid var(--border-subtle); margin-top: .7rem; }
        .panel { background: var(--surface-1); min-height: 62px; padding: .55rem .75rem; }
        .panel.side-for { border-top: 2px solid var(--side-for); }
        .panel.side-against { border-top: 2px solid var(--side-against); }
        .panel.side-judge { border-top: 2px solid var(--verdict); }
        .panel-label { color: var(--text-muted); font: 500 .6rem 'IBM Plex Mono', monospace; letter-spacing: .08em; text-transform: uppercase; }
        .panel-value { color: var(--text-primary); font: 500 .88rem 'Work Sans', sans-serif; margin-top: .2rem; }

        .stChatMessage { background: var(--surface-1); border: 1px solid var(--border-subtle); border-radius: 3px; padding: .35rem .5rem; }
        .stChatMessage p { color: var(--text-primary); font-family: 'Work Sans', sans-serif; font-size: .88rem; margin: .2rem 0; }

        [data-testid="stChatInput"] { padding-bottom: .3rem; }
        [data-testid="stChatInput"] textarea { background: var(--surface-2); border: 1px solid var(--border-strong); border-radius: 3px; color: var(--text-primary); font-family: 'Work Sans', sans-serif; padding: .5rem .7rem; min-height: 2.4rem; font-size: .85rem; }
        [data-testid="stChatInput"] textarea::placeholder { color: var(--text-muted); }

        .stButton button { border-radius: 3px; border: 1px solid var(--border-strong); background: transparent; color: var(--text-secondary); font-family: 'IBM Plex Mono', monospace; font-size: .72rem; letter-spacing: .04em; text-transform: uppercase; padding: .3rem .6rem; }
        .stButton button:hover { border-color: var(--verdict); color: var(--verdict); }

        .status-pill { display: inline-flex; align-items: center; gap: .5rem; color: var(--text-secondary); font: 500 .64rem 'IBM Plex Mono', monospace; text-transform: uppercase; letter-spacing: .06em; padding: .35rem .6rem; border: 1px solid var(--border-subtle); background: var(--surface-2); }
        .status-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--status-live); box-shadow: 0 0 0 3px rgba(62, 207, 142, .18); }

        .ledger-entry { display: flex; gap: .7rem; padding: .4rem 0; border-bottom: 1px solid var(--border-subtle); align-items: baseline; }
        .ledger-entry:last-child { border-bottom: none; }
        .side-chip { flex: 0 0 auto; font: 600 .6rem 'IBM Plex Mono', monospace; letter-spacing: .06em; text-transform: uppercase; padding: .22rem .45rem; border-radius: 2px; white-space: nowrap; }
        .side-chip.for { color: var(--side-for); background: var(--side-for-soft); border: 1px solid rgba(239, 74, 99, .35); }
        .side-chip.against { color: var(--side-against); background: var(--side-against-soft); border: 1px solid rgba(63, 169, 255, .35); }
        .ledger-text { color: var(--text-primary); font: 400 .84rem/1.45 'Work Sans', sans-serif; white-space: pre-wrap; }

        .verdict-card { border: 1px solid rgba(232, 182, 76, .4); background: var(--verdict-soft); padding: .65rem .8rem; margin: .2rem 0 .5rem; border-radius: 3px; }
        .verdict-label { display: block; color: var(--verdict); font: 600 .6rem 'IBM Plex Mono', monospace; letter-spacing: .1em; text-transform: uppercase; margin-bottom: .3rem; }
        .verdict-text { color: var(--text-primary); font: 400 .88rem/1.45 'Work Sans', sans-serif; white-space: pre-wrap; }

        [data-testid="stExpander"] { border: 1px solid var(--border-subtle) !important; background: var(--surface-1) !important; border-radius: 3px !important; }
        [data-testid="stExpander"] summary { font-family: 'IBM Plex Mono', monospace !important; font-size: .68rem !important; letter-spacing: .05em; text-transform: uppercase; color: var(--text-secondary) !important; padding: .3rem .6rem !important; }

        *:focus-visible { outline: 2px solid var(--focus-ring); outline-offset: 2px; }

        @media (max-width: 800px) {
            .block-container { padding: .8rem .8rem 1rem; }
            .grid { grid-template-columns: 1fr; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "debates" not in st.session_state:
    st.session_state.debates = {}

with st.sidebar:
    st.markdown('<div class="eyebrow">The docket</div>', unsafe_allow_html=True)
    st.markdown("### Chamber settings")
    rounds = st.slider("Rounds per motion", min_value=1, max_value=5, value=3)
    st.caption("Two agents argue, one judge decides.")
    st.divider()
    st.markdown('<div class="status-pill"><span class="status-dot"></span>Ready</div>', unsafe_allow_html=True)
    st.markdown(" ")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.debates = {}
        st.rerun()

st.markdown(
    '<div class="hero"><div class="eyebrow">Debate Chamber</div>'
    '<h1>State a claim. Watch it <em class="accent-for">argued</em> — and <em class="accent-against">contested</em>.</h1>'
    '<div class="lede">Two opposing agents debate, a neutral judge decides.</div></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="grid">'
    '<div class="panel side-for"><div class="panel-label">Speaker A · For</div>'
    '<div class="panel-value">Argues the motion</div></div>'
    '<div class="panel side-against"><div class="panel-label">Speaker B · Against</div>'
    '<div class="panel-value">Contests the motion</div></div>'
    '<div class="panel side-judge"><div class="panel-label">The Judge</div>'
    '<div class="panel-value">Calls the result</div></div>'
    '</div>',
    unsafe_allow_html=True,
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and message.get("debate_id"):
            render_verdict(message["content"])
            debate = st.session_state.debates[message["debate_id"]]
            with st.expander("Read the argument ledger"):
                render_ledger(debate["transcript"])
        else:
            st.markdown(message["content"])

if prompt := st.chat_input("State a claim..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Building both cases..."):
            try:
                transcript, verdict = run_debate(prompt, rounds)
                debate_id = str(len(st.session_state.debates))
                st.session_state.debates[debate_id] = {"transcript": transcript, "verdict": verdict}
                st.session_state.messages.append({"role": "assistant", "content": verdict, "debate_id": debate_id})
                render_verdict(verdict)
                with st.expander("Read the argument ledger"):
                    render_ledger(transcript)
            except Exception as error:
                message = f"I couldn't open the chamber yet. {error}"
                st.session_state.messages.append({"role": "assistant", "content": message})
                st.error(message)