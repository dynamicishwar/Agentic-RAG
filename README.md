Agentic RAG — LangChain · OpenRouter · Streamlit

A chat app in which an AI agent decides for itself which knowledge source to use. It can search the live web, Wikipedia or arXiv, call several of them in turn, and then write an answer that cites its sources. Simple questions it answers directly, with no tool at all.

The whole app is one Python file (app.py) plus requirements.txt.

User Query → Streamlit → LangChain Agent (OpenRouter LLM) → Tool Selection
           → Tool Execution → Retrieved Context → LLM → Final Answer
Features
Agent-driven tool choice: built on LangChain's create_agent. There are no keyword rules; the model picks tools from their descriptions.
Three tools:
Tool	Source	Used for
tavily_search	Tavily API	Recent news, current events, live information
wikipedia_search	Wikipedia API	Definitions, biographies, history, general knowledge
arxiv_search	arXiv (arxiv package)	Research papers: title, authors, date, abstract, link
Multi-step research: the agent reads what each tool returns and decides whether it needs another search.
Visible execution trace: an expandable panel under each answer lists the tools that were actually called, the query sent to each, and the sources each returned.
Cited answers: answers include clickable source links, followed by a list of the sources retrieved.
Conversation memory: follow-up questions work. The last 10 turns are kept in st.session_state.
Safeguards:
The number of tool calls per question is capped (6 by default, adjustable in the sidebar), and so is the number of model calls, to prevent loops.
Missing keys, bad model IDs, models without tool support and API failures each produce a plain error message.
API keys are entered as password fields and removed from any error text before it is shown.
Any OpenRouter model: pick from a preset list or enter a custom model ID. The model must support tool calling.
Quick start

Requires Python 3.10 or newer.

bash
git clone https://github.com/<your-username>/agentic-rag.git
cd agentic-rag
pip install -r requirements.txt
python -m streamlit run app.py

The app opens at http://localhost:8501. Enter your keys in the sidebar:

OpenRouter API key (required): create a standard API key at openrouter.ai/keys. A management/provisioning key returns 401.
Tavily API key (optional): get one at tavily.com. Without it, web search is turned off and the agent uses Wikipedia and arXiv.

You can also set the keys as environment variables before starting the app:

powershell
# Windows PowerShell
$env:OPENROUTER_API_KEY = "sk-or-v1-..."
$env:TAVILY_API_KEY = "tvly-..."
bash
# macOS / Linux
export OPENROUTER_API_KEY="sk-or-v1-..."
export TAVILY_API_KEY="tvly-..."
Try it
Question	Expected behaviour
Who is Alan Turing?	wikipedia_search
What are the latest AI announcements?	tavily_search (news)
Find recent research papers on RAG.	arxiv_search (sorted by date)
Explain Agentic RAG and find the latest research.	Several tools
What is 25 × 4?	Direct answer, no tools
Tech stack

LangChain (v1 create_agent with middleware call limits) · langchain-openrouter · Streamlit · Tavily · Wikipedia API · arxiv

Troubleshooting
Symptom	Fix
streamlit is not recognized	Run python -m streamlit run app.py instead
"missing ScriptRunContext" warnings, no browser window	The script was run with python app.py. Start it with streamlit run
401 from OpenRouter	The key is invalid or is a management key. Create a standard API key
"does not support tool calling"	Choose a tool-capable model, e.g. openai/gpt-4o-mini
Pydantic V1 warning on Python 3.14	Harmless; you can ignore it
Security

Never commit API keys. Enter them in the sidebar or set them as environment variables. If you add a .env file or .streamlit/secrets.toml, list it in .gitignore.

License

MIT
