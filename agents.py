import json
import os
import re
import time
from functools import lru_cache
from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
_active_model = MODEL
_unavailable_models = set()
_client = None

TONE = """
Write in plain, friendly English for Indian small-business owners.
Avoid jargon. Explain technical terms the first time they appear.
Keep a helpful mentor-like voice.
"""

CURATED_SOURCES = [
    {
        "number": 1,
        "title": "RBI: Payment and Settlement Systems",
        "url": "https://www.rbi.org.in/Scripts/PaymentSystems_UM.aspx",
        "focus": "How India's payment systems are regulated and the safeguards relevant to digital payments.",
    },
    {
        "number": 2,
        "title": "RBI: Payment System Indicators",
        "url": "https://www.rbi.org.in/Scripts/PSIUserView.aspx",
        "focus": "Official payment-system data and indicators; use only values present in the fetched page.",
    },
    {
        "number": 3,
        "title": "Digital India: UPI 123PAY",
        "url": "https://www.digitalindia.gov.in/initiative/upi-123pay/",
        "focus": "UPI access on feature phones and the UPI 123PAY service.",
    },
    {
        "number": 4,
        "title": "India Stack Global: Unified Payments Interface",
        "url": "https://www.indiastack.global/upi/",
        "focus": "UPI design, interoperability, and practical payment capabilities.",
    },
]


class _VisibleTextParser(HTMLParser):
    hidden_tags = {"head", "script", "style", "noscript", "svg", "nav", "footer"}

    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.hidden_tags:
            self.hidden_depth += 1

    def handle_endtag(self, tag):
        if tag in self.hidden_tags and self.hidden_depth:
            self.hidden_depth -= 1

    def handle_data(self, data):
        if not self.hidden_depth:
            text = " ".join(data.split())
            if text:
                self.parts.append(text)


@lru_cache(maxsize=1)
def fetch_curated_sources():
    fetched = []
    failures = []
    for source in CURATED_SOURCES:
        request = Request(
            source["url"],
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; MultiAgentBookWriter/1.0)",
                "Accept": "text/html,application/xhtml+xml",
            },
        )
        try:
            with urlopen(request, timeout=15) as response:
                if response.status != 200 or not response.geturl().startswith("https://"):
                    failures.append(source["title"])
                    continue
                page = response.read(2_000_000).decode("utf-8", errors="replace")
            parser = _VisibleTextParser()
            parser.feed(page)
            text = " ".join(parser.parts)
            if len(text) < 300 or re.search(r"\b404\s+Page Not Found\b", text, re.I):
                failures.append(source["title"])
                continue
            fetched.append({**source, "text": text[:4500]})
        except (HTTPError, URLError, TimeoutError, OSError):
            failures.append(source["title"])

    if len(fetched) < 2:
        failed_list = ", ".join(failures) or "no usable pages"
        raise RuntimeError(
            "The curated public sources could not be retrieved. Check your "
            f"internet connection and try again. Unavailable: {failed_list}."
        )
    return tuple(fetched)


class GeminiAPIError(RuntimeError):
    def __init__(self, original_type, message):
        self.original_type = original_type
        super().__init__(message)


def safe_error_message(error):
    message = str(error)
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        message = message.replace(api_key, "[REDACTED]")
    message = re.sub(r"AIza[0-9A-Za-z_-]{20,}", "[REDACTED]", message)
    message = re.sub(
        r"(?i)(api[_ -]?key\s*[:=]\s*)[^\s,;&]+",
        r"\1[REDACTED]",
        message,
    )
    return message[:1600]


def get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is missing. Add it to your local .env file."
            )
        _client = genai.Client(api_key=api_key)
    return _client


def is_model_unavailable(error):
    status = getattr(error, "status_code", None) or getattr(error, "code", None)
    message = str(error).lower()
    return str(status) == "404" or (
        "model" in message
        and any(
            phrase in message
            for phrase in (
                "not found",
                "not supported",
                "not available",
                "no longer available",
            )
        )
    )


def find_fallback_model(client, current_model):
    try:
        available = set()
        for model in client.models.list():
            name = (getattr(model, "name", "") or "").removeprefix("models/")
            actions = getattr(model, "supported_actions", []) or []
            if (
                name
                and name != current_model
                and name not in _unavailable_models
                and (not actions or any("generateContent" in action for action in actions))
            ):
                available.add(name)
    except Exception:
        return None

    preferences = (
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3-flash-preview",
        "gemini-flash-latest",
        "gemini-2.5-flash-lite",
    )
    return next((name for name in preferences if name in available), None) or next(
        iter(sorted(available)), None
    )


def call(prompt, max_tokens=3000, retries=3):
    """Call Gemini with bounded exponential backoff for transient errors."""
    global _active_model

    attempts_by_model = {}
    quota_fallback_used = False

    while True:
        current_model = _active_model
        attempt = attempts_by_model.get(current_model, 0)
        try:
            response = get_client().models.generate_content(
                model=current_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=max_tokens,
                ),
            )

            return response

        except Exception as e:
            error_text = safe_error_message(e)
            status = getattr(e, "status_code", None) or getattr(e, "code", None)
            error_type = type(e).__name__.lower()
            client = _client
            is_429 = str(status) == "429" or bool(
                re.search(r"\b429\b", error_text)
            )

            if is_model_unavailable(e) and client is not None:
                _unavailable_models.add(_active_model)
                fallback = find_fallback_model(client, _active_model)
                if fallback:
                    _active_model = fallback
                    continue
                raise GeminiAPIError(
                    type(e).__name__,
                    f"Configured Gemini model '{MODEL}' is unavailable and no "
                    "supported fallback model could be listed. Set GEMINI_MODEL "
                    "to an available generation model.",
                ) from None

            quota_message = error_text.lower()
            quota_exhausted = (
                is_429
                and (
                    "exceeded your current quota" in quota_message
                    or "resource_exhausted" in quota_message
                )
            )
            if quota_exhausted and not quota_fallback_used and client is not None:
                fallback = find_fallback_model(client, current_model)
                if fallback:
                    quota_fallback_used = True
                    _active_model = fallback
                    continue

            is_temporary = str(status) in {"429", "500", "502", "503", "504"} or re.search(
                r"\b(429|500|502|503|504)\b", error_text
            ) or any(
                marker in error_type for marker in ("timeout", "connection", "tempor")
            )

            if is_temporary:
                if attempt < retries:
                    attempts_by_model[current_model] = attempt + 1
                    time.sleep(min(2 ** attempt, 8))
                    continue
                raise GeminiAPIError(
                    type(e).__name__,
                    f"Gemini API remained unavailable after {retries + 1} "
                    f"attempts using '{current_model}'. {error_text}",
                ) from None

            if isinstance(e, RuntimeError) and "GEMINI_API_KEY is missing" in str(e):
                raise
            raise GeminiAPIError(type(e).__name__, error_text) from None


def clean_json(text):
    text = text.strip()

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    return text.strip()


def chapter_word_count(text):
    body = text.split("References:", 1)[0]
    body = re.sub(r"^#\s+.*$", "", body, count=1, flags=re.MULTILINE)
    return len(re.findall(r"\b[\w’'-]+\b", body))


def attach_curated_references(text, sources):
    body = text.split("References:", 1)[0].rstrip()
    citation_ids = sorted(
        {int(number) for number in re.findall(r"\[(\d+)\]", body)}
    )
    sources_by_number = {source["number"]: source for source in sources}
    reference_lines = []
    for number in citation_ids:
        source = sources_by_number.get(number)
        if source:
            url = source.get("url", "")
            if url.startswith(("https://", "http://")):
                title = source.get("title") or url
                reference_lines.append(f"[{number}] {title} — {url}")
    return body + "\n\nReferences:\n" + "\n".join(reference_lines)


def chapter_issues(text, sources):
    issues = []
    word_count = chapter_word_count(text)
    if not 600 <= word_count <= 900:
        issues.append(
            f"Chapter has {word_count} words; revise it to 600–900 words."
        )

    takeaway_matches = re.findall(r"(?m)^Takeaway:[ \t]+\S.*$", text)
    takeaway_match = re.search(r"(?m)^Takeaway:[ \t]+\S.*$", text)
    references_match = re.search(r"(?m)^References:\s*$", text)
    if len(takeaway_matches) != 1:
        issues.append("Use exactly one single-line 'Takeaway: <text>' ending.")
    if not references_match:
        issues.append("Add the required References: section at the end.")
        return issues
    if takeaway_match and takeaway_match.start() > references_match.start():
        issues.append("Place Takeaway: before References: at the end.")

    body, references = text.split("References:", 1)
    body_lines = [line.strip() for line in body.splitlines() if line.strip()]
    if takeaway_match and body_lines and not re.fullmatch(
        r"Takeaway:[ \t]+\S.*", body_lines[-1]
    ):
        issues.append("Takeaway must be the final one-line paragraph before References.")
    if re.search(r"(?m)^\s*(?:[-*+]\s+|\d+[.)]\s+)", body):
        issues.append("Remove bullet or numbered-list formatting from the chapter prose.")

    citation_ids = set(re.findall(r"\[(\d+)\]", body))
    reference_ids = set(re.findall(r"\[(\d+)\]", references))
    reference_lines = re.findall(
        r"(?m)^\[(\d+)\]\s+(.+?)\s+—\s+(https?://\S+)\s*$", references
    )

    if not citation_ids:
        issues.append("Add numbered citations for factual claims.")
    if citation_ids != reference_ids:
        issues.append("Make the References list match the citations used.")
    if len(reference_lines) != len(reference_ids):
        issues.append("References must include a source title and an HTTP(S) URL.")
    sources_by_number = {str(source["number"]): source for source in sources}
    invalid_ids = sorted(
        int(number)
        for number in citation_ids
        if str(int(number)) not in sources_by_number
    )
    if invalid_ids:
        issues.append(
            "Remove citation numbers without a supplied source: "
            + ", ".join(f"[{number}]" for number in invalid_ids)
            + "."
        )
    for number, title, url in reference_lines:
        source = sources_by_number.get(str(int(number)))
        if source:
            if title != source["title"] or url != source["url"]:
                issues.append(
                    f"Reference [{number}] does not match its retrieved catalog source."
                )
    return issues


# ---------------------------------------------------------
# PLANNER
# ---------------------------------------------------------

def planner(topic):
    response = call(
        f"""
You are the PLANNER agent.

Create exactly 3 chapters for this book:

{topic}

Audience:
Small-business owners in India.

Each chapter should support approximately 600–900 words.

Return ONLY valid JSON:

{{
  "chapters": [
    {{
      "title": "...",
      "goal": "...",
      "questions": ["...", "..."]
    }},
    {{
      "title": "...",
      "goal": "...",
      "questions": ["...", "..."]
    }},
    {{
      "title": "...",
      "goal": "...",
      "questions": ["...", "..."]
    }}
  ]
}}
""",
        max_tokens=1800
    )

    try:
        data = json.loads(clean_json(response.text))
        chapters = data["chapters"]

        if not isinstance(chapters, list) or len(chapters) != 3:
            raise ValueError("Planner did not create exactly 3 chapters.")
        if any(
            not isinstance(chapter, dict)
            or not all(chapter.get(key) for key in ("title", "goal", "questions"))
            for chapter in chapters
        ):
            raise ValueError("Planner returned an incomplete chapter outline.")

        return chapters

    except Exception:
        raise RuntimeError(
            "Planner returned an invalid outline. Try again with a simpler brief."
        ) from None


# ---------------------------------------------------------
# RESEARCHER
# ---------------------------------------------------------

def source_evidence_text(sources):
    return "\n\n".join(
        f"[{source['number']}] {source['title']}\n"
        f"URL: {source['url']}\n"
        f"Research focus: {source['focus']}\n"
        f"Retrieved page text:\n{source['text']}"
        for source in sources
    )


def research(topic, chapter):
    found_sources = [
        source
        for source in fetch_curated_sources()
        if re.search(r"\bUPI\b", source["text"], re.IGNORECASE)
    ]
    if not found_sources:
        raise RuntimeError(
            "None of the retrieved curated sources contains UPI-specific text. "
            "The Researcher will not invent evidence."
        )
    source_evidence = source_evidence_text(found_sources)

    response = call(
        f"""
You are the RESEARCHER agent.

Research this chapter using ONLY the retrieved source excerpts below. These
pages were fetched directly from the listed public URLs; Google Search was not
used. Cite each factual statement with the source number in square brackets.
Do not add outside knowledge, statistics, dates, or URLs. If the source text
does not support a useful claim, leave that claim out.

Book:
{topic}

Chapter:
{chapter["title"]}

Goal:
{chapter["goal"]}

Questions:
{chapter["questions"]}

Prepare a concise research brief of 4–6 useful, source-backed claims for an
Indian small-business owner. Put a [n] citation at the end of every claim.
Return plain text, not JSON.

CURATED SOURCES AND RETRIEVED PAGE TEXT:
{source_evidence}
""",
        max_tokens=1500
    )

    research_text = (response.text or "").strip()
    if not research_text:
        raise RuntimeError(f"Researcher returned no usable evidence for '{chapter['title']}'.")
    return research_text, found_sources, source_evidence


# ---------------------------------------------------------
# WRITER
# ---------------------------------------------------------

def write_chapter(topic, chapter, research_text, sources):

    if not sources:
        source_text = "No curated public sources were retrieved."
    else:
        source_text = "\n".join(
            f"[{source['number']}] {source['title']} — {source['url']}"
            for source in sources
        )

    response = call(
        TONE
        + f"""
You are the WRITER agent.

Book:
{topic}

Chapter:
{chapter["title"]}

Goal:
{chapter["goal"]}

Questions:
{chapter["questions"]}

RESEARCH:
{research_text}

ALLOWED SOURCES:
{source_text}

Write 750–800 words (aim for about 775). Do not stop before 700 words.

Requirements:

1. Use flowing prose.
2. Do NOT use bullet points inside the chapter.
3. Every factual or data claim must have a numbered citation such as [1].
4. Use ONLY the supplied source numbers, and only for claims supported in the research brief.
5. Never invent a source.
6. Keep the tone friendly and practical.
7. Explain technical terms the first time they appear.
8. Do not claim that UPI is zero-fee, prevents theft/reduces security risks,
   improves loan approval/access, automatically connects sales to khata or
   ledger apps, reveals customer loyalty/behavior, or makes lenders able to
   verify turnover unless the supplied excerpts directly establish that claim.
   If unsupported, omit it or frame it as a question for the owner to check.
9. End the chapter prose with exactly one line in this format:
Takeaway: <one short practical takeaway>

After that write:

References:

Only list sources actually cited in the chapter.

Return only the chapter.
""",
        max_tokens=4200
    )

    return response.text or ""


# ---------------------------------------------------------
# EDITOR
# ---------------------------------------------------------

def edit(text):

    response = call(
        TONE
        + f"""
You are the EDITOR agent.

Edit the chapter for:

- grammar
- spelling
- punctuation
- readability
- flow
- consistent tone

IMPORTANT:

Do NOT add new facts.
Do NOT add new citations.
Do NOT remove existing citations.
Keep all [n] citation numbers unchanged.
Keep exactly one final line formatted as `Takeaway: <text>` immediately before References.
Keep References.
Preserve the chapter's length; do not compress it below 700 words.
Do not use bullet points.

Return ONLY the edited chapter.

CHAPTER:

{text}
""",
        max_tokens=4200
    )

    return response.text or ""


# ---------------------------------------------------------
# FACT CHECKER
# ---------------------------------------------------------

def fact_check(text, research_text, sources, source_evidence):

    if sources:
        source_text = "\n".join(
            f"[{source['number']}] {source['title']} — {source['url']}"
            for source in sources
        )
    else:
        source_text = "No curated public sources were retrieved."

    response = call(
        f"""
You are the FACT-CHECKER agent.

Check the chapter against the research brief and retrieved public source
excerpts. Verify each citation against the actual excerpt, not just the title.

Check:

1. Important factual claims.
2. Numbers and statistics.
3. Dates.
4. Whether citations actually support the claims.
5. Whether citations refer to real sources.
6. Unsupported claims about fees, theft/security, credit/loans, automatic
    bookkeeping links, customer behavior, or lender verification of turnover.

Do not flag opinions or general advice.
Treat a factual claim as unsupported unless the attached retrieved excerpt
directly states or clearly entails it. Do not infer support from a source title.

Return ONLY valid JSON:

{{
  "status": "PASS",
  "issues": []
}}

OR:

{{
  "status": "REVISE",
  "issues": [
    {{
      "claim": "...",
      "problem": "...",
      "suggestion": "..."
    }}
  ]
}}

Use PASS only when important factual claims are adequately supported.

SOURCES:

{source_text}

RETRIEVED SOURCE EXCERPTS:

{source_evidence}

RESEARCHER BRIEF:

{research_text}

CHAPTER:

{text}
""",
        max_tokens=2200
    )

    return response.text or ""


# ---------------------------------------------------------
# REVISION
# ---------------------------------------------------------

def revise(text, feedback):

    response = call(
        TONE
        + f"""
You are the WRITER performing a final revision.

Fix every factual, citation, structure, and word-count problem identified by
the fact-checker or final checks.

Do not invent sources.
Do not invent statistics.
Preserve the chapter structure.
Preserve valid citations.
End the prose with exactly one line formatted as `Takeaway: <text>` immediately before References.
Preserve References.
If the chapter is under 600 words, expand it with practical explanation and
examples supported by the supplied research and sources. Put citations on all
factual claims. Aim for 750–800 words without adding filler. If it is over 900
words, trim repetition without dropping supported claims or citations.

Return ONLY the revised chapter.

CHAPTER:

{text}

FACT-CHECKER FEEDBACK:

{feedback}
""",
        max_tokens=4200
    )

    return response.text or ""


def fact_check_status(feedback):
    try:
        data = json.loads(clean_json(feedback))
        if data.get("status") == "PASS" and data.get("issues") == []:
            return "PASS"
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass
    return "REVISE"


# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------

def report_progress(callback, agent, message):
    if callback:
        callback(agent, message)


def run_pipeline(topic, progress_callback=None):
    report_progress(progress_callback, "Planner", "Creating the three-chapter outline.")
    outline = planner(topic)

    trace = [{"agent": "Planner", "summary": "Created the 3-chapter outline."}]
    chapters = []
    chapter_texts = []
    all_sources = []

    for number, chapter in enumerate(outline, 1):
        report_progress(
            progress_callback,
            "Researcher",
            f"Fetching curated public sources for chapter {number}.",
        )
        research_text, found_sources, source_evidence = research(topic, chapter)

        report_progress(progress_callback, "Writer", f"Drafting chapter {number}.")
        draft = write_chapter(topic, chapter, research_text, found_sources)

        report_progress(progress_callback, "Editor", f"Editing chapter {number}.")
        edited = attach_curated_references(edit(draft), found_sources)

        report_progress(
            progress_callback,
            "Fact-checker",
            f"Verifying claims and citations in chapter {number}.",
        )
        feedback = fact_check(
            edited,
            research_text,
            found_sources,
            source_evidence,
        )
        local_issues = chapter_issues(edited, found_sources)
        status = fact_check_status(feedback)
        if local_issues:
            status = "REVISE"
            feedback += "\n\nSTRUCTURE AND CITATION CHECKS:\n- " + "\n- ".join(
                local_issues
            )

        revision_used = status == "REVISE"
        if revision_used:
            report_progress(
                progress_callback,
                "Revision",
                f"Applying one revision to chapter {number}.",
            )
            edited = attach_curated_references(
                revise(edited, feedback), found_sources
            )
            final_feedback = fact_check(
                edited,
                research_text,
                found_sources,
                source_evidence,
            )
            final_status = fact_check_status(final_feedback)
            if final_status != "PASS":
                raise RuntimeError(
                    f"Fact-checker did not approve chapter {number} after its "
                    "single revision: " + final_feedback[:1000]
                )
        else:
            final_status = status

        remaining_issues = chapter_issues(edited, found_sources)
        if remaining_issues:
            raise RuntimeError(
                f"Chapter {number} did not meet the requirements after one "
                "revision: " + "; ".join(remaining_issues)
            )

        chapters.append(f"## Chapter {number}: {chapter['title']}\n\n{edited}")
        chapter_texts.append(edited)
        all_sources.extend(
            {
                "chapter": number,
                "number": source["number"],
                "title": source["title"],
                "url": source["url"],
            }
            for source in found_sources
        )

        trace.extend(
            [
                {"agent": "Researcher", "summary": f"Chapter {number}: fetched curated public sources."},
                {"agent": "Writer", "summary": f"Chapter {number}: wrote the chapter with citations."},
                {"agent": "Editor", "summary": f"Chapter {number}: edited grammar and readability."},
                {"agent": "Fact-checker", "summary": f"Chapter {number}: {status}; final result {final_status}."},
            ]
        )
        if revision_used:
            trace.append(
                {"agent": "Revision", "summary": f"Chapter {number}: completed one revision round."}
            )

    report_progress(
        progress_callback,
        "Final compliance check",
        "Checking chapter count, lengths, citations, references, and URLs.",
    )
    if len(chapter_texts) != 3:
        raise RuntimeError("Final compliance check failed: expected exactly 3 chapters.")
    for number, chapter_text in enumerate(chapter_texts, 1):
        chapter_sources = [
            {
                "number": source["number"],
                "title": source["title"],
                "url": source["url"],
            }
            for source in all_sources
            if source["chapter"] == number
        ]
        issues = chapter_issues(chapter_text, chapter_sources)
        if issues:
            raise RuntimeError(
                f"Final compliance check failed for chapter {number}: "
                + "; ".join(issues)
            )

    trace.append(
        {
            "agent": "Final compliance check",
            "summary": "Confirmed exactly 3 compliant chapters with curated public references.",
        }
    )
    book = f"# {topic}\n\n" + "\n\n".join(chapters)
    return {"book": book, "sources": all_sources, "trace": trace}
