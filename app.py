"""
Agentic RAG — LangChain + OpenRouter + Streamlit (single-file app)

User Query → Streamlit → LangChain Agent (OpenRouter LLM) → Tool Selection
           → Tool Execution → Retrieved Context → LLM → Final Answer

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

import os
import re
from datetime import date
from typing import Any, Literal

import arxiv
import requests
import streamlit as st
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openrouter import ChatOpenRouter
from tavily import TavilyClient

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
APP_TITLE = "Agentic RAG"

# Tool-calling capable models on OpenRouter. Any other ID can be entered as "Custom".
MODEL_OPTIONS = [
    "openai/gpt-4o-mini",
    "openai/gpt-4.1-mini",
    "openai/gpt-4o",
    "anthropic/claude-sonnet-4",
    "anthropic/claude-3.5-haiku",
    "google/gemini-2.5-flash",
    "meta-llama/llama-3.3-70b-instruct",
    "mistralai/mistral-small-3.2-24b-instruct",
    "qwen/qwen-2.5-72b-instruct",
]
CUSTOM_MODEL_LABEL = "Custom model ID…"

DEFAULT_MAX_TOOL_CALLS = 6      # tool executions allowed per question
HISTORY_TURNS = 10              # past user/assistant turns sent to the agent as context
HTTP_TIMEOUT = 15               # seconds for Wikipedia requests
USER_AGENT = "AgenticRAG-Streamlit/1.0 (educational demo)"

TOOL_INFO = {
    "tavily_search": ("Web search (Tavily)", "Recent news, current events, live information"),
    "wikipedia_search": ("Wikipedia", "Definitions, people, history, general knowledge"),
    "arxiv_search": ("arXiv", "Academic research papers, authors, abstracts"),
}

SYSTEM_PROMPT = """You are a helpful research assistant with access to three tools:

- tavily_search: real-time web search. Use for recent news, current events, product \
announcements, prices, or anything that may have changed recently.
- wikipedia_search: encyclopedic knowledge. Use for definitions, biographies, historical \
facts, and well-established concepts.
- arxiv_search: academic papers. Use when the user asks for research, papers, studies, or \
the scientific literature. Use sort_by="date" when the user wants recent/latest papers.

How to work:
1. Decide for yourself whether a tool is needed. For greetings, simple arithmetic, \
rewording, or questions you can answer reliably from general knowledge without \
time-sensitive facts, answer directly WITHOUT calling any tool.
2. When a tool is needed, choose the most appropriate one. If the question has several \
parts (e.g. explain a concept AND find the latest research), call several tools.
3. Examine what each tool returns. If the results are insufficient, irrelevant, or an \
error occurred, refine the query or try another tool — but avoid redundant searches.
4. Base factual claims on retrieved content. Never invent URLs, paper titles, or authors.
5. Cite sources inline as Markdown links, e.g. [Title](https://...), using only URLs that \
appeared in tool results. Finish with a short "Sources" list when tools were used.
6. Use the conversation history to resolve follow-up questions ("he", "that paper", \
"tell me more").
7. Be clear and well-structured; use Markdown headings or bullets when they help.

Today's date is {today}."""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _truncate(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def message_text(msg: Any) -> str:
    """Extract plain text from a message whose content may be a str or a list of blocks."""
    content = getattr(msg, "content", msg)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return ""


def redact(text: str) -> str:
    """Remove any API key that may have leaked into an error message."""
    text = str(text)
    for key in (st.session_state.get("_or_key"), st.session_state.get("_tv_key")):
        if key and len(key) > 6:
            text = text.replace(key, "***")
    # Generic patterns for OpenRouter / Tavily style keys
    text = re.sub(r"sk-or-[A-Za-z0-9\-_]{8,}", "sk-or-***", text)
    text = re.sub(r"tvly-[A-Za-z0-9\-_]{8,}", "tvly-***", text)
    text = re.sub(r"(Bearer\s+)[A-Za-z0-9\-_.]{8,}", r"\1***", text)
    return text


def friendly_error(exc: Exception, model_id: str) -> str:
    """Translate provider / agent exceptions into actionable messages."""
    raw = redact(f"{type(exc).__name__}: {exc}")
    low = raw.lower()
    if "recursion" in low:
        return "The agent hit its execution limit before finishing. Try a more specific question."
    if "401" in low or "unauthorized" in low or "invalid api key" in low or "no auth" in low:
        return "OpenRouter rejected the API key (401). Check the key in the sidebar."
    if "402" in low or "insufficient" in low or "credits" in low:
        return "OpenRouter reports insufficient credits (402) for this model. Add credits or pick a free/cheaper model."
    if "tool" in low and ("support" in low or "no endpoints" in low):
        return (f"The model `{model_id}` does not support tool calling on OpenRouter. "
                "Pick a tool-capable model (e.g. openai/gpt-4o-mini).")
    if "404" in low or "not a valid model" in low or "no endpoints" in low or "not found" in low:
        return f"Model `{model_id}` was not found on OpenRouter. Check the model ID (format: provider/model-name)."
    if "429" in low or "rate limit" in low:
        return "Rate limited by OpenRouter (429). Wait a moment and try again, or switch models."
    if "timeout" in low or "timed out" in low:
        return "The request timed out. Please try again."
    if "connection" in low:
        return "Network error while contacting OpenRouter. Check your internet connection."
    return f"Unexpected error: {_truncate(raw, 400)}"


# ─────────────────────────────────────────────────────────────────────────────
# Tools  (each returns (content_for_llm, sources_artifact))
# ─────────────────────────────────────────────────────────────────────────────
def build_tools(tavily_api_key: str | None) -> list:
    """Create the three LangChain tools. The Tavily key is captured in a closure,
    so it never appears in prompts, tool arguments, or output."""

    @tool(response_format="content_and_artifact")
    def tavily_search(
        query: str,
        topic: Literal["general", "news"] = "general",
        max_results: int = 5,
    ) -> tuple[str, list[dict]]:
        """Search the live web for recent information, news, and current developments.

        Args:
            query: A focused web search query.
            topic: "news" for recent news/announcements, otherwise "general".
            max_results: Number of results to return (1-8).
        """
        if not tavily_api_key:
            return ("ERROR: Tavily web search is unavailable because no Tavily API key was "
                    "provided. Tell the user to add one in the sidebar, or use another tool."), []
        try:
            client = TavilyClient(api_key=tavily_api_key)
            resp = client.search(
                query=query,
                topic=topic,
                max_results=max(1, min(int(max_results), 8)),
                include_answer=True,
                search_depth="basic",
            )
        except Exception as exc:  # noqa: BLE001
            return f"ERROR: Tavily search failed: {_truncate(redact(exc), 300)}", []

        results = resp.get("results") or []
        if not results:
            return f"No web results found for '{query}'. Try rephrasing the query.", []

        lines, sources = [], []
        if resp.get("answer"):
            lines.append(f"Quick answer: {resp['answer']}\n")
        for i, r in enumerate(results, 1):
            title, url = r.get("title") or "Untitled", r.get("url") or ""
            lines.append(f"[{i}] {title}\nURL: {url}\n"
                         f"Published: {r.get('published_date') or 'n/a'}\n"
                         f"Summary: {_truncate(r.get('content', ''), 700)}\n")
            if url:
                sources.append({"title": title, "url": url})
        return "\n".join(lines), sources

    @tool(response_format="content_and_artifact")
    def wikipedia_search(query: str, max_articles: int = 2) -> tuple[str, list[dict]]:
        """Look up general knowledge, definitions, biographies, and historical facts on Wikipedia.
        Returns article summaries with links.

        Args:
            query: The topic, person, or concept to look up.
            max_articles: Number of articles to summarise (1-3).
        """
        headers = {"User-Agent": USER_AGENT}
        try:
            search = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={"action": "query", "list": "search", "srsearch": query,
                        "srlimit": max(1, min(int(max_articles), 3)), "format": "json"},
                headers=headers, timeout=HTTP_TIMEOUT,
            )
            search.raise_for_status()
            hits = search.json().get("query", {}).get("search", [])
        except Exception as exc:  # noqa: BLE001
            return f"ERROR: Wikipedia search failed: {_truncate(str(exc), 300)}", []

        if not hits:
            return f"No Wikipedia articles found for '{query}'.", []

        lines, sources = [], []
        for hit in hits:
            title = hit.get("title", "")
            try:
                summ = requests.get(
                    "https://en.wikipedia.org/api/rest_v1/page/summary/"
                    + requests.utils.quote(title.replace(" ", "_"), safe=""),
                    headers=headers, timeout=HTTP_TIMEOUT,
                )
                summ.raise_for_status()
                data = summ.json()
                extract = data.get("extract") or ""
                url = (data.get("content_urls", {}).get("desktop", {}).get("page")
                       or f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}")
                description = data.get("description") or ""
            except Exception:  # noqa: BLE001 — fall back to the search snippet
                extract = re.sub(r"<[^>]+>", "", hit.get("snippet", ""))
                url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                description = ""
            lines.append(f"Title: {title}" + (f" — {description}" if description else "")
                         + f"\nURL: {url}\nSummary: {_truncate(extract, 1500)}\n")
            sources.append({"title": f"Wikipedia: {title}", "url": url})
        return "\n".join(lines), sources

    @tool(response_format="content_and_artifact")
    def arxiv_search(
        query: str,
        sort_by: Literal["relevance", "date"] = "relevance",
        max_results: int = 5,
    ) -> tuple[str, list[dict]]:
        """Find academic research papers on arXiv. Returns titles, authors, publication
        dates, abstracts, and links.

        Args:
            query: Research topic or keywords, e.g. "retrieval augmented generation".
            sort_by: "date" for the most recent papers, "relevance" for best matches.
            max_results: Number of papers to return (1-8).
        """
        criterion = (arxiv.SortCriterion.SubmittedDate if sort_by == "date"
                     else arxiv.SortCriterion.Relevance)
        try:
            client = arxiv.Client(page_size=10, delay_seconds=1.0, num_retries=2)
            search = arxiv.Search(query=query, max_results=max(1, min(int(max_results), 8)),
                                  sort_by=criterion)
            papers = list(client.results(search))
        except Exception as exc:  # noqa: BLE001
            return f"ERROR: arXiv search failed: {_truncate(str(exc), 300)}", []

        if not papers:
            return f"No arXiv papers found for '{query}'. Try broader keywords.", []

        lines, sources = [], []
        for i, p in enumerate(papers, 1):
            authors = ", ".join(a.name for a in p.authors[:5]) + (" et al." if len(p.authors) > 5 else "")
            published = p.published.strftime("%Y-%m-%d") if p.published else "n/a"
            lines.append(f"[{i}] {p.title}\nAuthors: {authors}\nPublished: {published}\n"
                         f"URL: {p.entry_id}\nPDF: {p.pdf_url or 'n/a'}\n"
                         f"Abstract: {_truncate(p.summary, 900)}\n")
            sources.append({"title": f"arXiv: {p.title}", "url": p.entry_id})
        return "\n".join(lines), sources

    return [tavily_search, wikipedia_search, arxiv_search]


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────
def build_agent(openrouter_key: str, model_id: str, tavily_key: str | None, max_tool_calls: int):
    llm = ChatOpenRouter(
        model=model_id,
        api_key=openrouter_key,
        temperature=0.2,
        max_retries=1,          # SDK backoff: ~150 s ceiling per call
        timeout=90_000,         # milliseconds (langchain-openrouter maps this to timeout_ms)
        app_title=APP_TITLE,
    )
    return create_agent(
        model=llm,
        tools=build_tools(tavily_key),
        system_prompt=SYSTEM_PROMPT.format(today=date.today().strftime("%B %d, %Y")),
        middleware=[
            # Hard cap on tool executions: further calls are blocked and the model
            # is told to answer with what it has.
            ToolCallLimitMiddleware(run_limit=max_tool_calls, exit_behavior="continue"),
            # Hard cap on LLM calls per question — prevents infinite loops.
            ModelCallLimitMiddleware(run_limit=max_tool_calls + 3, exit_behavior="end"),
        ],
    )


def history_as_messages(history: list[dict]) -> list:
    """Convert stored chat history into LangChain messages for follow-up context."""
    msgs = []
    usable = [m for m in history if not m.get("error")]
    for m in usable[-HISTORY_TURNS * 2:]:
        if m["role"] == "user":
            msgs.append(HumanMessage(m["content"]))
        elif m["content"]:
            msgs.append(AIMessage(m["content"]))
    return msgs


def summarise_args(args: dict) -> str:
    query = args.get("query")
    extra = {k: v for k, v in args.items() if k != "query"}
    out = f"`{query}`" if query is not None else ""
    if extra:
        out += ("  ·  " if out else "") + ", ".join(f"{k}={v}" for k, v in extra.items())
    return out or "`(no arguments)`"


def render_step(step: dict) -> None:
    """Render one real execution event."""
    kind = step["type"]
    if kind == "tool_call":
        label = TOOL_INFO.get(step["tool"], (step["tool"], ""))[0]
        st.markdown(f":material/build: **Step {step['n']} — called `{step['tool']}`** ({label})  \n"
                    f"Query: {summarise_args(step['args'])}")
    elif kind == "tool_result":
        if step["status"] == "error":
            st.markdown(f":material/error: `{step['tool']}` returned an error: {step['preview']}")
        else:
            st.markdown(f":material/check_circle: `{step['tool']}` returned "
                        f"{len(step['sources'])} source(s)")
        for s in step["sources"]:
            st.markdown(f"&nbsp;&nbsp;&nbsp;• [{s['title']}]({s['url']})")
        if step.get("raw") and step["status"] != "error":
            st.caption("Retrieved context (excerpt): " + _truncate(step["raw"], 350))
    elif kind == "model_note":
        st.markdown(f":material/chat: **Model message before tool call:** {step['text']}")
    elif kind == "limit":
        st.markdown(f":material/block: {step['text']}")


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    links = "\n".join(f"{i}. [{s['title']}]({s['url']})" for i, s in enumerate(sources, 1))
    st.markdown(f"**Retrieved sources**\n\n{links}")


def dedupe_sources(sources: list[dict]) -> list[dict]:
    seen, out = set(), []
    for s in sources:
        if s["url"] not in seen:
            seen.add(s["url"])
            out.append(s)
    return out


def trace_label(steps: list[dict]) -> str:
    calls = [s for s in steps if s["type"] == "tool_call"]
    if not calls:
        return "Agent execution — answered directly (no tools used)"
    tools = sorted({s["tool"] for s in calls})
    return f"Agent execution — {len(calls)} tool call(s): {', '.join(tools)}"


def run_agent(agent, messages: list, max_tool_calls: int, status, answer_box) -> dict:
    """Stream the agent, render real tool events live, and return the final result."""
    steps: list[dict] = []
    sources: list[dict] = []
    streamed = ""
    final_answer = ""
    call_no = 0

    # Safety net on graph steps (middleware nodes count too); the call-limit middleware
    # normally stops the agent well before this is reached.
    config = {"recursion_limit": 12 * (max_tool_calls + 3) + 20}
    for mode, data in agent.stream({"messages": messages}, config=config,
                                   stream_mode=["updates", "messages"]):
        if mode == "messages":
            chunk, meta = data
            if isinstance(chunk, AIMessageChunk) and meta.get("langgraph_node") == "model":
                text = message_text(chunk)
                if text:
                    streamed += text
                    answer_box.markdown(streamed + " ▌")
            continue

        # mode == "updates": {node_name: state_update}
        for node, update in (data or {}).items():
            if not isinstance(update, dict):
                continue
            for msg in update.get("messages", []) or []:
                if isinstance(msg, AIMessage):
                    text = message_text(msg).strip()
                    if msg.tool_calls:
                        if text:
                            step = {"type": "model_note", "text": text}
                            steps.append(step)
                            with status:
                                render_step(step)
                        for tc in msg.tool_calls:
                            call_no += 1
                            step = {"type": "tool_call", "n": call_no, "tool": tc["name"],
                                    "args": tc.get("args", {}), "id": tc.get("id")}
                            steps.append(step)
                            with status:
                                render_step(step)
                            status.update(label=f"Running `{tc['name']}`…")
                        streamed = ""
                        answer_box.empty()
                    elif text:
                        final_answer = text
                elif isinstance(msg, ToolMessage):
                    content = message_text(msg)
                    found = msg.artifact if isinstance(msg.artifact, list) else []
                    is_err = (getattr(msg, "status", "") == "error"
                              or content.startswith("ERROR")
                              or "limit" in content.lower() and not found)
                    step = {"type": "tool_result", "tool": msg.name or "tool",
                            "status": "error" if is_err else "success",
                            "sources": found, "preview": _truncate(content, 200),
                            "raw": _truncate(content, 4000)}
                    steps.append(step)
                    sources.extend(found)
                    with status:
                        render_step(step)
                    status.update(label="Analysing retrieved context…")

    if not final_answer:
        final_answer = streamed.strip()
    if final_answer.lower().startswith("model call limits exceeded"):
        final_answer = ""  # emitted by ModelCallLimitMiddleware when it stops the run
    if not final_answer:
        final_answer = ("I reached the agent's execution limit before producing a complete "
                        "answer. Please try a more specific question.")
        steps.append({"type": "limit", "text": "Execution limit reached — agent stopped."})
    return {"answer": final_answer, "steps": steps, "sources": dedupe_sources(sources)}


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title=APP_TITLE, page_icon=":material/hub:", layout="centered")

st.markdown("""
<style>
.block-container {max-width: 860px; padding-top: 2rem;}
[data-testid="stChatMessage"] {border-radius: 14px; padding: 0.6rem 0.9rem;}
.tool-card {border: 1px solid rgba(128,128,128,.25); border-radius: 10px;
            padding: .5rem .7rem; margin-bottom: .45rem;}
.tool-card b {font-size: .92rem;} .tool-card span {font-size: .8rem; opacity: .75;}
.hero {text-align:center; margin: 2.5rem 0 1.2rem 0;}
.hero h1 {font-size: 2rem; margin-bottom: .2rem;} .hero p {opacity:.7;}
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    env_or, env_tv = os.environ.get("OPENROUTER_API_KEY", ""), os.environ.get("TAVILY_API_KEY", "")
    or_input = st.text_input("OpenRouter API key", type="password",
                             placeholder="Loaded from environment" if env_or else "sk-or-…",
                             help="Stored only in this browser session; never displayed or logged.")
    tv_input = st.text_input("Tavily API key", type="password",
                             placeholder="Loaded from environment" if env_tv else "tvly-…",
                             help="Required only for web search. Wikipedia and arXiv need no key.")
    openrouter_key = (or_input or env_or).strip()
    tavily_key = (tv_input or env_tv).strip()
    st.session_state["_or_key"], st.session_state["_tv_key"] = openrouter_key, tavily_key

    choice = st.selectbox("Model", MODEL_OPTIONS + [CUSTOM_MODEL_LABEL], index=0,
                          help="Must support tool calling on OpenRouter.")
    if choice == CUSTOM_MODEL_LABEL:
        model_id = st.text_input("Custom OpenRouter model ID",
                                 placeholder="provider/model-name").strip()
    else:
        model_id = choice

    with st.expander("Advanced"):
        max_tool_calls = st.slider("Max tool calls per question", 1, 12, DEFAULT_MAX_TOOL_CALLS,
                                   help="Execution limit that prevents tool-calling loops.")

    st.subheader("Available tools")
    for name, (label, desc) in TOOL_INFO.items():
        badge = ""
        if name == "tavily_search" and not tavily_key:
            badge = " · needs key"
        st.markdown(f"<div class='tool-card'><b>{label}</b>{badge}<br>"
                    f"<code>{name}</code><br><span>{desc}</span></div>",
                    unsafe_allow_html=True)

    st.divider()
    if st.button("Clear conversation", icon=":material/delete:", width="stretch"):
        st.session_state.messages = []
        st.session_state.pop("pending_prompt", None)
        st.rerun()

    st.caption("The agent decides on its own which tools to use. "
               "Execution events shown are the real tool calls made.")

# ── Header / empty state ─────────────────────────────────────────────────────
EXAMPLES = [
    "Who is Alan Turing?",
    "What are the latest AI announcements?",
    "Find recent research papers on RAG.",
    "Explain Agentic RAG and find the latest research.",
    "What is 25 × 4?",
]

if not st.session_state.messages:
    st.markdown("<div class='hero'><h1>Agentic RAG</h1>"
                "<p>An AI agent that chooses between web search, Wikipedia and arXiv "
                "to answer your questions.</p></div>", unsafe_allow_html=True)
    cols = st.columns(2)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 2].button(ex, key=f"ex_{i}", width="stretch"):
            st.session_state.pending_prompt = ex
            st.rerun()

# ── Conversation history ─────────────────────────────────────────────────────
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m["role"] == "assistant" and m.get("steps") is not None:
            with st.expander(trace_label(m["steps"]), icon=":material/account_tree:"):
                if m["steps"]:
                    for s in m["steps"]:
                        render_step(s)
                else:
                    st.markdown("The model answered from its own knowledge — no tool was called.")
        if m.get("error"):
            st.error(m["content"])
        else:
            st.markdown(m["content"])
            render_sources(m.get("sources", []))

# ── New input ────────────────────────────────────────────────────────────────
prompt = st.chat_input("Ask anything…")
if not prompt and st.session_state.get("pending_prompt"):
    prompt = st.session_state.pop("pending_prompt")

if prompt:
    prompt = prompt.strip()
    with st.chat_message("user"):
        st.markdown(prompt)

    # Validate configuration before calling anything
    config_error = None
    if not openrouter_key:
        config_error = "Please enter your OpenRouter API key in the sidebar to start chatting."
    elif not model_id or not re.fullmatch(r"[\w.\-]+/[\w.\-:]+", model_id):
        config_error = "Please provide a valid OpenRouter model ID (format: `provider/model-name`)."

    if config_error:
        with st.chat_message("assistant"):
            st.warning(config_error)
        st.stop()

    history = history_as_messages(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        status = st.status("Agent is thinking…", expanded=True)
        answer_box = st.empty()
        try:
            agent = build_agent(openrouter_key, model_id, tavily_key or None, max_tool_calls)
            result = run_agent(agent, history + [HumanMessage(prompt)],
                               max_tool_calls, status, answer_box)
        except Exception as exc:  # noqa: BLE001
            msg = friendly_error(exc, model_id)
            status.update(label="Agent failed", state="error", expanded=False)
            answer_box.empty()
            st.error(msg)
            with st.expander("Error details"):
                st.code(_truncate(redact(f"{type(exc).__name__}: {exc}"), 1500), language=None)
            st.session_state.messages.append(
                {"role": "assistant", "content": msg, "error": True, "steps": None})
        else:
            if not any(s["type"] == "tool_call" for s in result["steps"]):
                with status:
                    st.markdown("The model answered from its own knowledge — no tool was called.")
            status.update(label=trace_label(result["steps"]), state="complete", expanded=False)
            answer_box.markdown(result["answer"])
            render_sources(result["sources"])
            st.session_state.messages.append({
                "role": "assistant", "content": result["answer"],
                "steps": result["steps"], "sources": result["sources"],
            })
