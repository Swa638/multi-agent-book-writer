# Multi-Agent Book Writer

AI Engineer take-home assignment: a small, sequential multi-agent system that researches and writes a three-chapter book for Indian small-business owners.

## Assignment Objective

Generate a practical, source-backed book from a brief while making each agent's responsibility and handoff visible. The default brief is **Pay Me on UPI: How Digital Payments Changed Small Business in India**.

## Features

- Exactly three chapters, checked to contain 600–900 words each.
- Direct retrieval of a small, visible catalog of official public sources; no Google Search grounding required.
- Chapter-local numbered citations and references generated from retrieved source pages.
- Editing, fact-checking, and at most one revision for each chapter.
- Final checks for chapter count, length, sections, citation/reference matching, source bounds, and HTTP(S) reference URLs.
- Live agent status, readable trace, source links, and Markdown download in Streamlit.
- Bounded retries for temporary Gemini API failures.

## Architecture

See [architecture.md](architecture.md) for the workflow diagram and handoff details.

## Agent Responsibilities

| Agent | Responsibility |
| --- | --- |
| Planner | Produces a validated outline containing exactly three chapters. |
| Researcher | Fetches the configured public source pages and summarizes only information supported by their retrieved text. |
| Writer | Drafts the chapter using the outline and numbered curated sources. |
| Editor | Improves grammar, clarity, and flow without changing citations or adding facts. |
| Fact-checker | Reviews factual claims and whether citations are supported by retrieved research and sources. |
| Revision | Makes one targeted repair when fact-checking or local checks fail. |
| Final compliance check | Rejects output that misses required length, structure, citation, or source rules. |

## Workflow

For each chapter, the Researcher → Writer → Editor → Fact-checker sequence runs once. A failed fact-check or local chapter check triggers at most one Revision. The final compliance check then validates all three chapters before the book is returned. Fetched source evidence is reused by the fact-checker, avoiding extra search calls.

## Tech Stack

- Python
- Google Gemini API (`google-genai`)
- Python standard-library HTTP/HTML tools for direct source retrieval
- Streamlit
- `python-dotenv`

## Installation

From the project directory in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If the project `.venv` already exists, activate it and install the requirements rather than creating another environment.

## Environment Variables

Copy `.env.example` to `.env`, then set:

```text
GEMINI_API_KEY=your_private_key
GEMINI_MODEL=your_available_gemini_model
```

Keep `.env` local. It is ignored by Git. Never paste the API key into source code, issue reports, or commits.

## Run

```powershell
python -m streamlit run app.py
```

Enter a brief and click **Run Multi-Agent System**. The full workflow is only started by that button; launching Streamlit does not make Gemini calls.

## Example Workflow

1. Planner creates three chapter goals and research questions.
2. Researcher retrieves public-source evidence for Chapter 1.
3. Writer drafts it; Editor improves readability.
4. Fact-checker reviews claims and citations. If necessary, Revision runs once.
5. The same sequence runs for Chapters 2 and 3.
6. Final compliance checks run before the book, trace, and source links are displayed.

## Citation and Fact-Checking Approach

The source catalog is declared as `CURATED_SOURCES` in `agents.py` and displayed in the Streamlit UI. It currently uses RBI's Payment and Settlement Systems overview and Payment System Indicators, MeitY's UPI 123PAY page, and India Stack Global's UPI overview. The Researcher retrieves these pages directly over HTTPS. NPCI currently blocks direct retrieval from this environment, so it is not cited as a fetched source. The writer receives only these source IDs and URLs; the app rebuilds each References list from the catalog, never from model-invented URLs. The Fact-checker receives the retrieved page excerpts. Deterministic checks confirm that citation IDs map to references and retrieved sources and that reference URLs use HTTP(S).

This is an AI-assisted review, not a guarantee of truth or live URL availability. Important claims should still be reviewed by a person before publication.

## Error Handling

The Gemini client is created only when the user starts generation, so the UI can load without making a request. Temporary API errors receive bounded exponential-backoff retries. If the selected model is unavailable, the app checks the configured account's listed generation models and tries a current fallback. API errors appear in a **Debug/Error Details** expander with their exception type and sanitized message; API-key values are redacted and stack traces are not shown. Research retrieval itself does not depend on Gemini Search grounding.

## Project Structure

```text
multi_agent_book_writer/
├── app.py
├── agents.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── architecture.md
```

Generated books are offered as a Markdown download rather than written to the repository.

## Interview Notes and Future Improvements

AI coding tools may have assisted during development; the design stays intentionally small so the responsibilities, prompts, data handoffs, retry policy, and validation can be explained and reviewed directly.

Reasonable future improvements include unit tests with mocked Gemini responses, stronger claim-to-source evaluation, configurable chapter word limits, and optional persistence of generated books.
