"""
Transcript parser for earnings call PDFs.

Identifies the most senior executive from the participant listing
(first listed under Management = highest in corporate hierarchy),
detects speaker turns, collects all speech by that executive,
and scrubs financial-specific text.
"""

import re
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import fitz

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import MIN_PAGE_TEXT_LENGTH, MAX_HEADER_LINES, MAX_FOOTER_LINES

logger = logging.getLogger(__name__)


@dataclass
class SpeakerTurn:
    speaker_name: str
    speaker_title: str
    text: str
    word_count: int
    turn_index: int = 0


@dataclass
class ParseResult:
    filepath: str
    company_name: str
    executive_name: str
    executive_title: str
    speech_text: str
    word_count: int
    total_speakers: int
    total_turns: int
    all_speakers: List[str] = field(default_factory=list)


def extract_text_from_pdf(filepath, skip_first_page=False):
    """Extract text from a PDF, stripping headers/footers per page."""
    filepath = Path(filepath)
    if not filepath.exists():
        logger.error("PDF not found: %s", filepath)
        return None

    try:
        doc = fitz.open(str(filepath))
    except Exception as e:
        logger.error("Cannot open PDF %s: %s", filepath, e)
        return None

    pages = []
    first_page = 1 if skip_first_page else 0

    for page_num in range(first_page, len(doc)):
        page = doc[page_num]
        blocks = page.get_text("blocks", sort=True)
        lines = [b[4].strip() for b in blocks if b[6] == 0 and b[4].strip()]
        if not lines:
            continue

        if len(lines) > MAX_HEADER_LINES + MAX_FOOTER_LINES + 1:
            top = 0
            for i in range(min(MAX_HEADER_LINES, len(lines))):
                if _looks_like_boilerplate(lines[i]):
                    top = i + 1
                else:
                    break
            bottom = len(lines)
            for i in range(len(lines) - 1, max(len(lines) - MAX_FOOTER_LINES - 1, top), -1):
                if _looks_like_boilerplate(lines[i]):
                    bottom = i
                else:
                    break
            lines = lines[top:bottom]

        page_text = "\n".join(lines)
        if len(page_text) >= MIN_PAGE_TEXT_LENGTH:
            pages.append(page_text)

    doc.close()
    full_text = "\n\n".join(pages)
    if not full_text.strip():
        logger.warning("No extractable text in %s", filepath)
        return None

    logger.info("Extracted %d words from %s", len(full_text.split()), filepath.name)
    return full_text


def _looks_like_boilerplate(line):
    """Short lines, page numbers, copyright notices, URLs."""
    line = line.strip()
    if len(line) < 10:
        return True
    if line.replace(" ", "").replace("-", "").replace(".", "").isdigit():
        return True
    lower = line.lower()
    giveaways = [
        "page ", "confidential", "for private circulation",
        "not for publication", "all rights reserved",
        "www.", "http", "\u00a9", "copyright",
    ]
    return any(g in lower for g in giveaways)


# Regex patterns for locating the Management/Company participant section
# and stopping at the Analyst section.

_MANAGEMENT_HEADER = re.compile(
    r'^[ \t]*(?:From\s+(?:the\s+)?)?'
    r'(?:Management|Company|Corporate)\s*'
    r'(?:Participants?|Representatives?|Team|Side|Speakers?)?'
    r'\s*[:\-\u2013\u2014]?\s*$',
    re.MULTILINE | re.IGNORECASE,
)

_ANALYST_HEADER = re.compile(
    r'^[ \t]*(?:From\s+(?:the\s+)?)?'
    r'(?:Analyst|Moderator|Operator|Investor|Media|Press|'
    r'Questioner|Conference|Call|Other)\s*'
    r'(?:Participants?|Representatives?|Team|Side|Speakers?)?'
    r'\s*[:\-\u2013\u2014]?\s*$',
    re.MULTILINE | re.IGNORECASE,
)

_HONORIFIC = re.compile(
    r'^(?:[*\-\d.]+\s*)?'
    r'(?:Mrs\.?|Ms\.?|Mr\.?|Dr\.?|Shri\.?|Sri\.?|Sh\.?|Smt\.?)\s*',
    re.IGNORECASE,
)

_NAME_TITLE_SPLIT = re.compile(r'\s*(?:[,(]|[-\u2013\u2014]+)\s*(.+)$')

# Words that disqualify a line from being a person's name
_NOT_A_NAME = {
    "limited", "ltd", "pvt", "inc", "corp", "corporation",
    "industries", "enterprises", "technologies", "solutions",
    "services", "group", "holdings", "international",
    "quarter", "quarterly", "annual", "results", "report",
    "transcript", "conference", "earnings", "fiscal",
    "management", "participants", "analysts", "moderator",
    "operator", "presentation", "discussion", "overview",
}


def _parse_participant_line(raw):
    """Split 'Mr. Rohit Jawa - CEO & MD, HUL' into ('Rohit Jawa', 'CEO & MD, HUL')."""
    name = _HONORIFIC.sub("", raw.strip()).strip()
    title = ""
    m = _NAME_TITLE_SPLIT.search(name)
    if m:
        title = m.group(1).strip().rstrip(")")
        name = name[:m.start()].strip()
    if name.isupper():
        name = name.title()
    return name.strip(" .,:;"), title


def _looks_like_person_name(text):
    if not text:
        return False
    words = text.split()
    if len(words) < 2 or len(words) > 5:
        return False
    if any(w.lower() in _NOT_A_NAME for w in words):
        return False
    return all(w[0].isupper() or len(w) == 1 for w in words)


def find_senior_executive(text):
    """
    Return (name, title) of the most senior executive.

    Scans the first ~6000 chars for a participant listing. Indian
    earnings calls list management in strict corporate hierarchy,
    so the first person under Management is the most senior.

    Falls back to the first name-with-title line if no explicit
    Management section header is found.
    """
    scan = text[:6000]
    lines = scan.split("\n")

    mgmt = _MANAGEMENT_HEADER.search(scan)
    if mgmt:
        start = scan[:mgmt.start()].count("\n")
        for j in range(start + 1, min(start + 25, len(lines))):
            line = lines[j].strip()
            if not line:
                continue
            if _ANALYST_HEADER.match(line):
                break
            name, title = _parse_participant_line(line)
            if name and _looks_like_person_name(name):
                logger.info("Senior executive (mgmt section): '%s' | '%s'", name, title)
                return name, title

    # Fallback: first name-with-title before the analyst section
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if _ANALYST_HEADER.match(stripped):
            break
        name, title = _parse_participant_line(stripped)
        if name and _looks_like_person_name(name) and title:
            logger.info("Senior executive (first w/ title): '%s' | '%s'", name, title)
            return name, title

    logger.warning("Could not find senior executive in participant listing")
    return "", ""


# Speaker turn patterns, ordered from most specific to least.

_TITLED_TURN = re.compile(
    r'^[ \t]*'
    r'(?:Mrs\.?|Ms\.?|Mr\.?|Dr\.?|Shri\.?|Sri\.?|Sh\.?|Smt\.?)?\s*'
    r'([A-Z][a-zA-Z.\'""-]*(?:\s+[A-Z][a-zA-Z.\'""-]*){0,4})'
    r'\s*[-\u2013\u2014]+\s*'
    r'((?:(?:Managing\s+Director|CEO|Chairman|CFO|COO|MD|CMD|CTO|'
    r'Chief\s+(?:Executive|Financial|Operating)\s+Officer|'
    r'Executive\s+Director|Whole[\s-]?Time\s+Director|'
    r'Analyst|Moderator|Operator|'
    r'Head\s+[-\u2013\u2014]?\s*(?:Investor\s+Relations|IR|Finance|Treasury)|'
    r'Company\s+Secretary|President|Vice\s+President|VP)'
    r'(?:\s*(?:&|and)\s*(?:CEO|MD|Managing\s+Director|CFO))?'
    r'[^:\n]{0,80})?)'
    r'\s*[:]\s*$',
    re.MULTILINE | re.IGNORECASE,
)

_SIMPLE_TURN = re.compile(
    r'^[ \t]*'
    r'(?:Mrs\.?|Ms\.?|Mr\.?|Dr\.?|Shri\.?|Sri\.?|Sh\.?|Smt\.?)?\s*'
    r'([A-Z][a-zA-Z.\'""-]*(?:\s+[A-Z][a-zA-Z.\'""-]*){0,3})'
    r'\s*:\s*$',
    re.MULTILINE,
)

_INLINE_TURN = re.compile(
    r'^[ \t]*'
    r'(?:Mrs\.?|Ms\.?|Mr\.?|Dr\.?|Shri\.?|Sri\.?|Sh\.?|Smt\.?)?\s*'
    r'([A-Z][a-zA-Z.\'""-]*(?:\s+[A-Z][a-zA-Z.\'""-]*){1,3})'
    r'\s*[-\u2013\u2014:]?\s*[-\u2013\u2014:]\s+'
    r'(.+)',
    re.MULTILINE,
)

_SECTION_BREAK = re.compile(
    r'^\s*(?:Q\s*&\s*A\s*(?:Session|Round)?|'
    r'Question\s*(?:and|&)\s*Answer|'
    r'Presentation|Opening\s+Remarks?|'
    r'Management\s+Discussion|Forward[\s-]?Looking|'
    r'Safe\s+Harbor|Disclaimer|'
    r'Operator\s+Instructions?)\s*$',
    re.MULTILINE | re.IGNORECASE,
)


def identify_speakers(text):
    """Parse transcript into ordered speaker turns."""
    lines = text.split("\n")
    markers = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or _SECTION_BREAK.match(stripped):
            continue

        m = _TITLED_TURN.match(line)
        if m:
            markers.append((i, m.group(1).strip(), m.group(2).strip(), "titled"))
            continue

        m = _SIMPLE_TURN.match(line)
        if m:
            name = m.group(1).strip()
            if _looks_like_person_name(name) or name.lower() in ("operator", "moderator"):
                markers.append((i, name, "", "simple"))
                continue

        m = _INLINE_TURN.match(line)
        if m:
            name = m.group(1).strip()
            if _looks_like_person_name(name) or name.lower() in ("operator", "moderator"):
                markers.append((i, name, "", "inline"))

    turns = []
    for idx, (line_idx, name, title, kind) in enumerate(markers):
        next_start = markers[idx + 1][0] if idx + 1 < len(markers) else len(lines)

        if kind == "inline":
            m = _INLINE_TURN.match(lines[line_idx])
            speech_lines = [m.group(2).strip()] if m else []
            for j in range(line_idx + 1, next_start):
                if lines[j].strip():
                    speech_lines.append(lines[j].strip())
        else:
            speech_lines = [lines[j].strip() for j in range(line_idx + 1, next_start)
                            if lines[j].strip()]

        speech = " ".join(speech_lines)
        if not speech:
            continue

        turns.append(SpeakerTurn(
            speaker_name=name, speaker_title=title,
            text=speech, word_count=len(speech.split()),
            turn_index=len(turns),
        ))

    logger.debug("Identified %d speaker turns", len(turns))
    return turns


def _name_matches(name_a, name_b):
    """Two names match if at least 2 cleaned parts overlap."""
    def tokenize(n):
        honorifics = {"mr", "mrs", "ms", "dr", "shri", "smt", "sri"}
        return {p.lower().strip() for p in n.replace(".", " ").replace("-", " ").split()} - honorifics

    a, b = tokenize(name_a), tokenize(name_b)
    return len(a & b) >= 2 if a and b else False


# Patterns to strip financial content while preserving natural language.
# Order matters: currency patterns must fire before bare-number patterns.
_FINANCIAL_NOISE = [
    re.compile(r'(?:\u20b9|Rs\.?|INR|USD|\$|\u20ac|\u00a3)\s*[\d,.]+\s*(?:crore|crores|lakh|lakhs|billion|million|mn|bn|cr|lac|lacs)?', re.I),
    re.compile(r'\b[\d,.]+\s+(?:crore|crores|lakh|lakhs|billion|million|mn|bn|cr|lac|lacs)\b', re.I),
    re.compile(r'\b[\d,.]+\s*(?:%|percent|percentage|basis\s+points?|bps)\b', re.I),
    re.compile(r'\b(?:Q[1-4]|H[12])\s*(?:FY\s*\'?\d{2,4})\b', re.I),
    re.compile(r'\bFY\s*\'?\d{2,4}(?:\s*[-\u2013]\s*\d{2,4})?\b', re.I),
    re.compile(r'\b(?:YoY|QoQ|MoM|Y-o-Y|Q-o-Q|year[- ]on[- ]year|quarter[- ]on[- ]quarter)\b', re.I),
    re.compile(r'\b(?:EBITDA|EBIT|PAT|PBT|ROCE|ROE|ROA|ROIC|EPS|P/E|PE|NPA|GNPA|NNPA|NIM|AUM|NAV|EMI|CAGR|IRR|NPV|WACC)\b'),
    re.compile(r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b'),
    re.compile(r'\b\d+(?:\.\d+)?\b'),
]


def clean_financial_text(text):
    """Strip currency amounts, percentages, financial acronyms, and bare numbers."""
    for pattern in _FINANCIAL_NOISE:
        text = pattern.sub(" ", text)
    return re.sub(r'\s+', ' ', text).strip()


def _company_name_from_filename(filename):
    """'Infosys_Ltd_2024_01_Jan.pdf' -> 'Infosys Ltd.'"""
    stem = Path(filename).stem
    parts = []
    for token in stem.split("_"):
        if re.match(r'^\d{4}$', token) and 2020 <= int(token) <= 2030:
            break
        parts.append(token)
    return re.sub(r'\bLtd\b', 'Ltd.', " ".join(parts))


def parse_transcript(filepath):
    """
    Parse a single earnings-call PDF and return a ParseResult
    with the senior executive's cleaned speech, or None on failure.
    """
    filepath = Path(filepath)

    full_text = extract_text_from_pdf(filepath, skip_first_page=False)
    if not full_text:
        return None

    company = _company_name_from_filename(filepath.name)
    exec_name, exec_title = find_senior_executive(full_text)

    logger.info("Parsing: %s | Company: '%s' | Executive: '%s'",
                filepath.name, company, exec_name or "NOT FOUND")

    turns = identify_speakers(full_text)
    if not turns:
        logger.warning("No speakers found in %s", filepath.name)
        return None

    speakers = list({t.speaker_name for t in turns})

    if not exec_name:
        return ParseResult(
            filepath=str(filepath), company_name=company,
            executive_name="", executive_title="", speech_text="",
            word_count=0, total_speakers=len(speakers),
            total_turns=len(turns), all_speakers=speakers,
        )

    matched_turns = [t for t in turns if _name_matches(exec_name, t.speaker_name)]
    if not matched_turns:
        logger.warning("Executive '%s' not matched to any speaker in %s", exec_name, filepath.name)
        return ParseResult(
            filepath=str(filepath), company_name=company,
            executive_name=exec_name, executive_title=exec_title,
            speech_text="", word_count=0,
            total_speakers=len(speakers), total_turns=len(turns),
            all_speakers=speakers,
        )

    raw_speech = " ".join(t.text for t in matched_turns)
    cleaned = clean_financial_text(raw_speech)
    wc = len(cleaned.split())

    logger.info("Extracted %d words from '%s' (%d turns, %d raw -> %d cleaned)",
                wc, exec_name, len(matched_turns), len(raw_speech.split()), wc)

    return ParseResult(
        filepath=str(filepath), company_name=company,
        executive_name=exec_name, executive_title=exec_title,
        speech_text=cleaned, word_count=wc,
        total_speakers=len(speakers), total_turns=len(turns),
        all_speakers=speakers,
    )
