import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

model_a = ChatGroq(
    model='openai/gpt-oss-20b',
    api_key=groq_api_key,
    temperature=0.5
)

model_b = ChatGoogleGenerativeAI(
    model='gemini-3.6-flash',
    api_key=google_api_key,
    temperature=0.6
)

model_j = ChatOpenAI(
    model='nvidia/nemotron-3-super-120b-a12b:free',
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key,
    temperature=0.3
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

chain_a = prompt_a | model_a
chain_b = prompt_b | model_b
chain_judge = prompt_judge | model_j

topic = "Social media does more harm than good to society"
transcript = []

def format_transcript(transcript):
    if not transcript:
        return "(no arguments yet)"
    return "\n".join(f"{t['speaker']}: {t['text']}" for t in transcript)

for round_num in range(3):
    current_context = format_transcript(transcript)
    resp_a = chain_a.invoke({"topic": topic, "transcript": current_context, "round_num": round_num + 1})
    transcript.append({"speaker": "Speaker A (FOR)", "text": resp_a.text})

    current_context = format_transcript(transcript)
    resp_b = chain_b.invoke({"topic": topic, "transcript": current_context, "round_num": round_num + 1})
    transcript.append({"speaker": "Speaker B (AGAINST)", "text": resp_b.text})

current_context = format_transcript(transcript)
resp_judge = chain_judge.invoke({"topic": topic, "transcript": current_context})

print("Final Debate Transcript:")
print(current_context)
print("\nJudge's Verdict:")
print(resp_judge.text)