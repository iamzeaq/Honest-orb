#!/usr/bin/env python3
"""Build the Honest Orb constitution data layer.

Reads the raw PDFs in data/raw/ and writes data/sections.json, an array of
objects like::

    {"id": "CFRN-s35", "source": "Constitution 1999", "chapter": "IV",
     "section": "35", "title": "Right to personal liberty",
     "text": "<full section text>",
     "keywords": ["arrest", "detention", "bail", "lawyer"]}

Scope (deliberate, not exhaustive):
  * Constitution 1999: Chapter IV only (sections 33-46). This copy is the
    unamended 1999 text, so no other chapters are used.
  * Police Act 2020 (OCR'd Gazette scan): only the sections about arrest,
    search, bail, detention and use of force.

Usage:
    python3 api/build_sections.py

Requires:
    pip install pymupdf
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_PATH = ROOT / "data" / "sections.json"


def require_fitz():
    try:
        import fitz  # PyMuPDF
    except ImportError:
        sys.exit("ERROR: PyMuPDF is required: pip install pymupdf")
    return fitz


def find_input(*candidates: str, hint: str) -> Path:
    for name in candidates:
        path = RAW_DIR / name
        if path.is_file():
            return path
    lowered = hint.lower()
    for path in sorted(RAW_DIR.glob("*.pdf")):
        if lowered in path.name.lower():
            return path
    sys.exit(f"ERROR: could not find input PDF {candidates[0]!r} in {RAW_DIR}")


def squash(text: str) -> str:
    """Collapse all whitespace runs to single spaces."""
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Constitution 1999 (Chapter IV, sections 33-46)
# ---------------------------------------------------------------------------

CONSTITUTION_TITLES = {
    "33": "Right to life",
    "34": "Right to dignity of human persons",
    "35": "Right to personal liberty",
    "36": "Right to fair hearing",
    "37": "Right to private and family life",
    "38": "Right to freedom of thought, conscience and religion",
    "39": "Right to freedom of expression and the press",
    "40": "Right to peaceful assembly and association",
    "41": "Right to freedom of movement",
    "42": "Right to freedom from discrimination",
    "43": "Right to acquire and own immovable property",
    "44": "Compulsory acquisition of property",
    "45": "Restriction on and derogation from fundamental human rights",
    "46": "Special jurisdiction of High Court and Legal aid",
}

CONSTITUTION_KEYWORDS = {
    "33": ["life", "death penalty", "court sentence", "killing",
           "self-defence", "lawful arrest", "riot", "right to life"],
    "34": ["dignity", "torture", "inhuman treatment", "slavery",
           "forced labour", "degrading", "human dignity"],
    "35": ["arrest", "detention", "bail", "lawyer", "court",
           "remain silent", "24 hours", "carry go station", "police cell",
           "release"],
    "36": ["fair hearing", "court", "trial", "lawyer", "innocent",
           "defence", "interpreter", "judgment"],
    "37": ["privacy", "home", "telephone", "search phone",
           "correspondence", "family life"],
    "38": ["religion", "thought", "conscience", "worship", "belief",
           "prayer", "freedom of religion"],
    "39": ["expression", "speech", "press", "media", "opinion",
           "protest", "free speech"],
    "40": ["assembly", "association", "protest", "rally", "union",
           "gather", "meeting", "demonstration"],
    "41": ["movement", "travel", "reside", "relocate", "waka", "japa",
           "freedom of movement"],
    "42": ["discrimination", "equality", "ethnic group", "gender",
           "tribe", "religion", "equal rights"],
    "43": ["property", "land", "own", "acquire", "house", "inherit"],
    "44": ["compulsory acquisition", "compensation", "land",
           "seize property", "government take land", "prompt payment"],
    "45": ["restrictions", "derogation", "emergency", "public order",
           "security", "morality"],
    "46": ["enforce rights", "high court", "legal aid", "sue",
           "human rights case", "claim", "court order"],
}


def extract_constitution(fitz, path: Path) -> list[dict]:
    doc = fitz.open(path)
    lines: list[str] = []
    for page in doc:
        for block in page.get_text("blocks"):
            for raw in block[4].splitlines():
                line = raw.strip()
                if not line:
                    continue
                # Strip the repeated "Back to top / Nigerian Constitution" strip
                # and its page number.
                if re.fullmatch(r"(?i)back\s*to\s*top", line):
                    continue
                if re.fullmatch(r"(?i)nigerian constitution", line):
                    continue
                if re.fullmatch(r"\d{1,3}", line):
                    continue
                lines.append(line)
    joined = squash(" ".join(lines))

    start_key = "33. (1) Every person has a right to life"
    end_key = "47. There shall be a National Assembly"
    start = joined.find(start_key)
    end = joined.find(end_key)
    if start < 0 or end < 0 or end <= start:
        sys.exit("ERROR: Chapter IV boundaries not found in constitution PDF")
    chapter = joined[start:end]

    # Split on section headers "33." ... "46." at the start of a section.
    # In-text references ("sections 37, 38", "section 305") never carry a
    # trailing period directly after the number, so require one.
    parts = re.split(r"(?<!\b[Ss]ection )(?<!\b[Ss]ections )"
                     r"(?:(?<=\s)|^)\b(3[3-9]|4[0-6])\.(?!\d)", chapter)
    # re.split with a group yields [pre, num, body, num, body, ...]; the text
    # before the first header is empty because the chapter starts at "33.".
    sections: list[dict] = []
    for i in range(1, len(parts), 2):
        num = parts[i]
        body = squash(parts[i + 1]) if i + 1 < len(parts) else ""
        text = squash(f"{num}. {body}")
        sections.append({
            "id": f"CFRN-s{num}",
            "source": "Constitution 1999",
            "chapter": "IV",
            "section": num,
            "title": CONSTITUTION_TITLES[num],
            "text": text,
            "keywords": CONSTITUTION_KEYWORDS[num],
        })

    want = [str(n) for n in range(33, 47)]
    got = [s["section"] for s in sections]
    if got != want:
        sys.exit(f"ERROR: expected constitution sections {want}, got {got}")
    # The slice runs to section 47; drop the Chapter V matter that trails
    # the final section ("Chapter V The Legislature ...").
    sections[-1]["text"] = re.sub(r"\s*Chapter\s+V\b.*$", "",
                                  sections[-1]["text"], flags=re.S)
    sections[-1]["text"] = squash(sections[-1]["text"])
    return sections


# ---------------------------------------------------------------------------
# Police Act 2020 (arrest, search, bail, detention, use of force)
# ---------------------------------------------------------------------------

# (section number, official marginal title, keywords)
POLICE_WANTED: list[tuple[str, str, list[str]]] = [
    ("32", "Arrest generally",
     ["arrest", "investigation", "charge", "court", "civil case",
      "carry go station", "suspect"]),
    ("33", "Mode of arrest",
     ["arrest", "touch", "confine", "custody", "submit",
      "carry go station", "how police arrest"]),
    ("34", "No unnecessary restraint",
     ["handcuff", "restraint", "force", "violence", "escape",
      "court order", "use of force", "no beating"]),
    ("35", "Notification of cause of arrest and rights of suspect",
     ["arrest", "rights", "remain silent", "lawyer", "legal aid",
      "inform family", "reason for arrest", "carry go station"]),
    ("36", "Arrest in lieu prohibited",
     ["arrest in lieu", "family", "relative", "innocent person",
      "carry another person", "no proxy arrest"]),
    ("37", "Humane treatment of arrested suspect",
     ["humane treatment", "torture", "dignity", "no beating",
      "detention", "rights of suspect", "use of force"]),
    ("38", "Arrest by police officer without warrant",
     ["arrest without warrant", "reasonable suspicion", "steal",
      "deserter", "escape", "offensive weapon", "carry go station",
      "commit offence"]),
    ("43", "Arrested suspect to be taken immediately to police station",
     ["police station", "carry go station", "inform allegation",
      "lawyer", "bail", "facilities", "detention"]),
    ("44", "Recording of arrests",
     ["record", "arrest register", "photograph", "fingerprints",
      "48 hours", "confessional statement", "documentation"]),
    ("48", "Power to search",
     ["search", "seize", "stolen property", "premises",
      "search power", "human rights"]),
    ("49", "Power to stop and search",
     ["stop and search", "search vehicle", "stolen",
      "offensive weapon", "reasonable suspicion", "search phone",
      "roadblock"]),
    ("50", "Action before a search takes place",
     ["search procedure", "identify", "uniform", "identity card",
      "grounds of search", "question before search"]),
    ("51", "General conduct of the search",
     ["search conduct", "forcible search", "last resort",
      "detention time", "same sex search", "use of force",
      "embarrassment"]),
    ("52", "Search of arrested suspect",
     ["search arrested suspect", "danger", "escape", "evidence",
      "bail", "custody search"]),
    ("53", "Examination of arrested suspect",
     ["intimate search", "body search", "detention",
      "medical practitioner", "consent", "custody record",
      "strip search"]),
    ("58", "Search warrant safeguards",
     ["search warrant", "court order", "oath", "premises",
      "one occasion", "warrant requirement"]),
    ("62", "Release on bail of a suspect arrested without warrant",
     ["bail", "release", "recognizance", "sureties", "24 hours",
      "bail na free", "police bail"]),
    ("63", "Power to release on bail before charge is accepted",
     ["bail", "release", "recognizance", "inquiry", "police station",
      "report back"]),
    ("64", "Remedy of suspect detained in custody",
     ["detention", "bail", "court", "24 hours", "custody", "remedy",
      "unlawful detention", "overstay"]),
    ("69", "Police to report to supervising magistrates",
     ["report", "magistrate", "arrest record", "monthly report",
      "oversight", "attorney-general", "bail"]),
    ("70", "Chief magistrate to visit police stations every month",
     ["magistrate visit", "inspection", "police station", "detention",
      "bail", "arraignment", "oversight", "cell visit"]),
]

# Literal OCR fixes for this Gazette scan, applied in order. Entries are
# (old, new) plain-string replacements.
POLICE_LITERAL_FIXES: list[tuple[str, str]] = [
    ("sea,:)wariant", "search warrant"),
    ("ifcollll)itted", "if committed"),
    ("Nigeriawhich,", "Nigeria which,"),
    ("anoffen" + chr(0x20F) + "e", "an offence"),
    (",andJor", ", and for"),
    ("Nigeri'l,)iabieto", "Nigeria, liable to"),
    ("ofPolice", "of Police"),
    ("implement'ofho" + chr(0x5C) + "isebreaking", "implement of housebreaking"),
    ("byll<;OUrt,of", "by a court of"),
    ("o...ltvoid", "or to avoid"),
    ("Wlier" + chr(0x296) + "", "Where"),
    ("arr" + chr(0x211) + "st", "arrest"),
    ("arrrst", "arrest"),
    ("instru" + chr(0x297) + "ent " + chr(0x298) + "f" + chr(0x299) + "iolenc" + chr(0x29A) + ".orp" + chr(0x29B) + "isonous", "instrument of violence or poisonous"),
    ("instru" + chr(0x297) + "ent", "instrument"),
    ("ofviolenc" + chr(0x29A) + ".orppisonous", "of violence or poisonous"),
    ("inity':fbtrlish", "might furnish"),
    ("his.possession", "his possession"),
    (chr(0x5C) + "Yithout", "Without"),
    (chr(0x5C) + "Yarrant", "Warrant"),
    ("offensiveor", "offensive or"),
    ("offence,,", "offence,"),
    ("restrain'-", ""),
    ("E-on of", ""),
    ('""""-', ""),
    (chr(0xA0), " "),
    ("-" + chr(0xAD), "-"),
    (chr(0xAD), ""),
    ("next-ofkin", "next-of-kin"),
    ("next-of kin", "next-of-kin"),
    ("InspectorGeneral", "Inspector-General"),
    ("-1' Nig" + chr(0xBF) + "ria Polic;e Act, 2020 taken immecliat" + chr(0x149) + "ly,", "taken immediately,"),
    ("warrant, Recording and taken", "warrant, and taken"),
    ("the of arrests. police officer", "the police officer"),
    ("suspect- without (a)", "suspect- (a)"),
    ("committed an warrant. offence", "committed an offence"),
    (". ;_,liich", " which"),
    ("as the case may,", "as the case may be,"),
    ("search m any", "search in any"),
    ("course. of", "course of"),
    ("only'.if", "only if"),
    ("shall .be", "shall be"),
    ("the. opposite", "the opposite"),
    ("his.full", "his full"),
    ("section'which", "section which"),
    ("state'--", "state -"),
    ("specifY the--", "specify the -"),
    ("subsection 13", "subsection (13)"),
    ("arrests,'", "arrests,"),
    ("repOiis", "reports"),
    ("contain" + chr(0x5C), "contain"),
    ("inNigeria", "in Nigeria"),
    ("and .detained", "and detained"),
    ("afford. reas" + chr(0x210) + "n", "afford reason"),
    ("arrest .has", "arrest has"),
    ("wammt", "warrant"),
    ("infonnation", "information"),
    ("-1" + chr(39), ""),
    ("immecliatnly", "immediately"),
    ("immecliat" + chr(0x149) + "ly", "immediately"),
    ("contait" + chr(0x5C), "contain"),
    ("havingjurisdiction", "having jurisdiction"),
    ("{ I}", "(1)"),
    ("( 1 5)", "(15)"),
    ("which.he is alleged", "which he is alleged"),
    ("making the arrest. arrest shall", "making the arrest shall"),
    ("subjected unnecessary to restraint", "subjected to unnecessary restraint"),
    (chr(0x5C) + "vithout", "without"),
    (chr(0x5C) + "VOtild", "would"),
]

# Word-boundary regex fixes (case-sensitive; each tuple is pattern, repl).
POLICE_WORD_FIXES: list[tuple[str, str]] = [
    (r"\bfonn\b", "form"),
    (r"\bwilh\b", "with"),
    (r"\bTeasonable\b", "reasonable"),
    (r"\bce11ain\b", "certain"),
    (r"\bprool'", "proof"),
    (r"\bl\.1e\b", "he"),
    (r"\bbyajudge\b", "by a judge"),
    (r"\bpu11ished\b", "punished"),
    (r"\bhejs\b", "he is"),
    (r"\bnotifY\b", "notify"),
    (r"\bbetaken\b", "be taken"),
    (r"\botl1erwise\b", "otherwise"),
    (r"\btlle\b", "the"),
    (r"\btl;le\b", "the"),
    (r"\btiHl\b", "the"),
    (r"\burider\b", "under"),
    (r"\bshill!\b", "shall"),
    (r"\bshail\b", "shall"),
    (r"\btD\b", "to"),
    (r"\bFe\(leral\b", "Federal"),
    (r"\bFedecal\b", "Federal"),
    (r"\bleveL\b", "level"),
    (r"\brep011\b", "report"),
    (r"\brepOii\b", "report"),
    (r"\bcl1ief\b", "chief"),
    (r"\bofficia\b", "official"),
    (r"\bihay\b", "may"),
    (r"\bpers:m\b", "person"),
    (r"\bperSon\b", "person"),
    (r"\bcomi\b", "court"),
    (r"\bcontait\b", "contain"),
    (r"\bembarras ment\b", "embarrassment"),
    (r"\bexa!;iination\b", "examination"),
    (r"\bcc\)\.pperate\b", "cooperate"),
    (r"\bsus\.picion\b", "suspicion"),
    (r"\bcarrie!", "carried"),
    (r"\bstoleii\b", "stolen"),
    (r"\bartideswhich\b", "articles which"),
    (r"\bag\u029cin?st\b", "against"),
    (r"\bconnec!ed\b", "connected"),
    (r"\bwiththi\b", "with this"),
    (r"\bki\*d\b", "kind"),
    (r"\b4he power\b", "The power"),
    (r"\buhder\b", "under"),
    (r"\be:vidence\b", "evidence"),
    (r"\bfo\u029e\b", "for"),
    (r"\bid\u029fntity\b", "identity"),
    (r"\bdaniage\b", "damage"),
    (r"\bvitiue\b", "virtue"),
    (r"\bititerfere\b", "interfere"),
    (r"\bs" + chr(0x179) + "tatement\b", "statement"),
    (r"\boffic\.er\b", "officer"),
    (r"\b\u017ascertain\b", "ascertain"),
    (r"\bcatmot\b", "cannot"),
    (r"\bAlly summons\b", "Any summons"),
    (r"\bha is\b", "he is"),
    (r"\bofthe\b", "of the"),
    (r"\bsaf\u0323\b", "safe"),
    (r"\bsl\)ch\b", "such"),
    (r"\bini,ist\b", "interest"),
    (r"\bsubjectto\b", "subject to"),
    (r"\bin\. the actual\b", "in the actual"),
    (r"\bto6pm\b", "to 6pm"),
    (r"\btrnnsmit\b", "transmit"),
    (r"\btliat\b", "that"),
    (r"\bi11\b", "in"),
    (r"\bmightbe\b", "might be"),
    (r"\btothe\b", "to the"),
    (r"\bextentthat\b", "extent that"),
    (r"\brequiredfor\b", "required for"),
    (r"\bofNigeria\b", "of Nigeria"),
    (r"\bthisAct\b", "this Act"),
    (r"\bitljury\b", "injury"),
    (r"\bprecatitions\b", "precautions"),
    (r"\bidentifY\b", "identify"),
    (r"\bspecifY\b", "specify"),
    (r"\bsl\)bstance\b", "substance"),
    (r"\binNigeria\b", "in Nigeria"),
    (r"\bspecifYing\b", "specifying"),
]


def fix_police_ocr(text: str) -> str:
    for old, new in POLICE_LITERAL_FIXES:
        text = text.replace(old, new)
    for pattern, repl in POLICE_WORD_FIXES:
        text = re.sub(pattern, repl, text)
    # Backstop for stray OCR glyphs (codepoints verified from the scan).
    text = (text.replace(chr(0x211), "e").replace(chr(0x210), "a")
                .replace(chr(0x28E), "s").replace(chr(0x21C), "i")
                .replace(chr(0x21D), "i").replace(chr(0x179), "t")
                .replace(chr(0x17A), "a").replace(chr(0x29C), "a")
                .replace(chr(0x29D), "m").replace(chr(0x298), "o")
                .replace(chr(0x299), "v").replace(chr(0x25A), "e")
                .replace(chr(0x29B), "p").replace(chr(0x283), "o")
                .replace(chr(0x149), "n").replace(chr(0x29A), "e").replace(chr(0xB7), ""))
    # Numbered cross-references split by the scan ("section 4 7").
    text = re.sub(r"\b[Ss]ection\s+(\d)\s+(\d)\b", r"section \1\2", text)
    # Page-header debris that survived line filtering ("2020 No. 2").
    for debris in ["2020 N+. 2 .", "2020 No. 2", "2020 No, 2",
                   "2020 No.2", "2020 :Nu.", "2020 No 2"]:
        text = text.replace(debris, " ")
    # Subsection-marker artefacts peculiar to this scan.
    for old, new in [("{J}", "(3)"), ("(2}", "(2)"), ("(1}", "(1)"),
                     ("{l)", "(1)"), ("{1)", "(1)"), ("{ 1)", "(1)"),
                     ("{ I)", "(1)"), ("( I)", "(1)"), ("(I)", "(1)"),
                     ("(I 0)", "(10)"), ("(II)", "(11)"), ("(1 5)", "(15)"),
                     ("(;)", "(j)"), (" < ", " "),
                     ("subsection (!)", "subsection (1)"),
                     ("; or (!) required", "; or (l) required")]:
        text = text.replace(old, new)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    # Words split or joined by line breaks and the tight scan spacing.
    text = re.sub(r"(\([a-z0-9]+\))([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"([a-z])\.([A-Z])", r"\1. \2", text)
    # The scan puts a space before every semicolon ("offence ;").
    text = text.replace(" ;", ";")
    return squash(text)


def extract_police(fitz, path: Path) -> list[dict]:
    doc = fitz.open(path)
    lines: list[str] = []
    for page in doc:
        for block in page.get_text("blocks"):
            x0, _y0, x1, _y1, text = block[0], block[1], block[2], block[3], block[4]
            # Margin side-notes (section titles in the outer margin) and
            # folio numbers live in narrow blocks at the page edges; the
            # main column always spans most of the page width.
            if x0 >= 400 or x1 <= 105:
                continue
            for raw in text.splitlines():
                line = raw.strip()
                if not line:
                    continue
                # Page headers ("Nigeria Police Act, 2020 ..." / "A 47").
                if re.search(r"Nigeria\s+\S+\s+Act,?\s*2020", line):
                    continue
                if "2020" in line and len(line) < 30:
                    continue
                if re.fullmatch(r"A\s*\d{1,3}", line):
                    continue
                # Part / sub-part headings ("PART VII-...", "B. Search").
                if re.match(r"PART\s+[IVX]", line):
                    continue
                if re.match(r"^[A-E]\.\s+[A-Z][a-z]", line):
                    continue
                # Lines that are only punctuation are OCR debris.
                if not re.search(r"[A-Za-z0-9]", line):
                    continue
                # The scan splits some section numbers ("4 8.", "5 1.").
                line = re.sub(r"^(\d)\s+(\d)\s*\.", r"\1\2.", line)
                lines.append(line)

    wanted = {num for num, _, _ in POLICE_WANTED}
    raw_sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in lines:
        head = re.match(r"^(\d{1,3})\.(?!\d)", line)
        if head:
            if head.group(1) in wanted:
                current = head.group(1)
                raw_sections.setdefault(current, []).append(line)
            else:
                # A section outside our arrest/search/bail/detention/
                # use-of-force scope closes the current one.
                current = None
        elif current is not None:
            raw_sections[current].append(line)

    sections: list[dict] = []
    for num, title, keywords in POLICE_WANTED:
        if num not in raw_sections:
            sys.exit(f"ERROR: Police Act section {num} not found in {path.name}")
        text = fix_police_ocr(" ".join(raw_sections[num]))
        # Strip the duplicated margin side-note heading the section
        # ("48. Power to search. 48.-(1) ...").
        text = re.sub(r"^" + num + r"\.\s+.*?(?=" + num + r"\.)", "", text)
        # Strip margin side-note titles that leaked into the body text.
        for _title in [t for _, t, _ in POLICE_WANTED] + [
                "Form and requisites of warrant of arrest"]:
            text = re.sub(r" " + re.escape(_title) + r"\.",
                          " ", text, flags=re.IGNORECASE)
        # Strip a trailing margin-note fragment ("...action. No").
        _lower = text.lower()
        _frags = ({t.split()[0].lower() for _, t, _ in POLICE_WANTED}
                  | {" ".join(t.split()[:2]).lower()
                     for _, t, _ in POLICE_WANTED})
        for _frag in sorted(_frags, key=len, reverse=True):
            if re.search(r"[.?!]\s+" + re.escape(_frag) + r"\.?$",
                         _lower):
                text = re.sub(r"[.?!]\s+" + re.escape(_frag) + r"\.?$",
                              ".", text, flags=re.IGNORECASE)
                break
        text = squash(text)
        sections.append({
            "id": f"NPA2020-s{num}",
            "source": "Police Act 2020",
            "chapter": "VII",
            "section": num,
            "title": title,
            "text": text,
            "keywords": keywords,
        })
    return sections


# ---------------------------------------------------------------------------
# New Acts (spec step 1): generic ingestor + per-law config.
#
# Each entry: file candidates in data/raw/, id prefix, display source,
# jurisdiction ("federal" or a lowercase state name), domain, the section
# numbers to extract, titles, per-section keywords, and plain-string OCR
# fixes applied in order. Laws whose PDF is absent from data/raw/ are
# SKIPPED with a report line (never fatal): the existing sections are kept
# byte-identical. Laws whose PDF yields no usable text layer (photographed
# scan, e.g. the FCT Recovery of Premises Act) are also skipped and
# reported, per the "only if clean and verifiable" rule.
# ---------------------------------------------------------------------------

NEW_LAWS: list[dict] = [
    {
        "files": ["acja_2015.pdf"],
        "id_prefix": "ACJA2015",
        "source": "Administration of Criminal Justice Act 2015",
        "jurisdiction": "federal",
        "domain": "criminal_procedure",
        "note": "Federal criminal procedure: arrest, bail, remand and trial.",
        "fixes": [],
        "region_end": r"FIRST SCHEDULE",
        "sections": {
            # number: (title, keywords) — titles verified against the gazette body
            "4": ("Mode of arrest",
                  ["arrest", "how police arrest", "touch", "confine",
                   "carry go station", "warrant"]),
            "5": ("No unnecessary restraint",
                  ["handcuff", "restraint", "no beating", "tie", "force"]),
            "7": ("Arrest in lieu prohibited",
                  ["arrest in lieu", "family", "relative", "proxy arrest",
                   "carry another person", "dem no see the person"]),
            "158": ("Bail: general entitlement",
                    ["bail", "release", "court", "bail na free", "police bail",
                     "apply for bail"]),
            "164": ("Court bail in other cases",
                    ["bail", "court", "discretion", "release", "remand"]),
            "293": ("Remand procedure after arrest",
                    ["remand", "magistrate", "brought to court", "detention",
                     "reasonable time", "custody"]),
            "295": ("Court consideration of remand application",
                    ["remand", "probable cause", "magistrate", "detention",
                     "release", "bail"]),
        },
    },
    {
        "files": ["lagos_tenancy_law_2011.pdf"],
        "id_prefix": "LAGTEN2011",
        "source": "Lagos Tenancy Law 2011",
        "jurisdiction": "lagos",
        "domain": "tenancy",
        "note": ("Lagos State ONLY (see s.1 Application for the premises this Law "
                 "covers and its exclusions). Tenancy notice and eviction rules "
                 "differ by state."),
        "fixes": [
            ("faci1ities", "facilities"),
        ],
        "sections": {
            # Titles verified against the Law's headings; s.1 captured whole
            "1": ("Application of the Law",
                  ["lagos", "apply", "covers", "exempt", "excluded",
                   "residential", "business premises", "tenancy"]),
            "2": ("Jurisdiction of the Courts",
                  ["court", "jurisdiction", "magistrate", "high court",
                   "tenancy", "lagos"]),
            "4": ("Advance Rent",
                  ["advance rent", "one year", "three months", "pay rent",
                   "landlord collect", "upfront"]),
            "6": ("Rights of a Tenant",
                  ["tenant rights", "enjoy", "rent", "receipt", "lagos"]),
            "8": ("Obligations of the Landlord",
                  ["landlord", "repair", "obligations", "maintain",
                   "rates", "tenant"]),
            "13": ("Length of Notice",
                   ["notice to quit", "quit notice", "length of notice",
                    "one month", "three months", "six months", "yearly tenant",
                    "eviction", "landlord", "leave", "eject"]),
            "16": ("Tenant refusing to give up possession",
                   ["refuse", "give up possession", "overstay", "hold over",
                    "mesne profits", "landlord", "evict"]),
            "25": ("Grounds for Possession",
                   ["grounds", "possession", "recover premises", "breach",
                    "arrears", "landlord", "evict"]),
        },
    },
    {
        "files": ["foi_act_2011.pdf"],
        "id_prefix": "FOIA2011",
        "source": "Freedom of Information Act 2011",
        "jurisdiction": "federal",
        "domain": "civic_foi",
        "note": "Right to request records from public institutions; request procedure and timelines.",
        "fixes": [
            ("or a apart thereof", "or a part thereof"),
        ],
        "sections": {
            "1": ("Rights of access to records",
                  ["information", "records", "request", "government agency",
                   "ministry", "access", "right to know"]),
            "2": ("Information about public institution",
                  ["contact", "register", "list", "records", "agency",
                   "publish", "information desk"]),
            "3": ("Request for access to records",
                  ["apply", "request procedure", "write", "application",
                   "describe", "agency", "how to request"]),
            "4": ("Time for granting or refusing application",
                  ["days", "7 days", "timeline", "grant", "refuse", "refusal",
                   "deemed refused", "transfer"]),
            "6": ("Extension of time limit",
                  ["extend", "extension", "more time", "large records",
                   "consultation", "notice"]),
            "8": ("Fees",
                  ["fees", "charges", "pay", "duplication", "transcription",
                   "cost", "free"]),
            "20": ("Judicial review",
                   ["court", "judge", "review", "refused", "denied",
                   "sue", "order", "lawyer"]),
        },
    },
    {
        "files": ["land_use_act.pdf"],
        "id_prefix": "LUA1978",
        "source": "Land Use Act (Cap L5)",
        "jurisdiction": "federal",
        "domain": "land",
        "note": "Vests land in the Governor; rights of occupancy; consent and compensation rules.",
        "fixes": [],
        "region_end": r"SUBSIDIARY LEGISLATION",
        "sections": {
            "1": ("Vesting of all land in the State",
                  ["land", "governor", "vested", "own land", "all land",
                   "family land", "ancestral"]),
            "2": ("Control and management of land",
                  ["manage", "control", "governor", "advisory", "land use",
                   "committee"]),
            "5": ("Powers of the Governor in relation to land",
                  ["governor", "grant", "statutory right of occupancy",
                   "urban land", "certificate", "C of O", "consent"]),
            "6": ("Powers of local government in non-urban land",
                  ["local government", "customary", "right of occupancy",
                   "rural land", "village", "chairman"]),
            "9": ("Certificates of occupancy",
                  ["certificate", "C of O", "evidence", "title", "land",
                   "right of occupancy"]),
            "28": ("Power of Governor to revoke rights of occupancy",
                   ["revoke", "revocation", "take land", "government take land",
                   "override", "public purpose"]),
            "29": ("Compensation on revocation",
                   ["compensation", "pay", "unexhausted improvements",
                    "crops", "buildings", "taken", "prompt payment"]),
        },
    },
    {
        "files": ["labour_act.pdf"],
        "id_prefix": "LABOUR1971",
        "source": "Labour Act",
        "jurisdiction": "federal",
        "domain": "employment",
        "note": ("Contracts, wages, hours, termination and redundancy for workers. "
                 "s.7 (written particulars) deliberately excluded: this copy numbers "
                 "that provision '3.' in the body while its own arrangement calls it "
                 "'7.', colliding with another s.3 — unverifiable as s.7, so skipped."),
        "fixes": [],
        "region_start": r"LABOUR ACT\s+An Act to repeal",
        "region_end": r"SUBSIDIARY LEGISLATION",
        "sections": {
            "11": ("Termination of contracts by notice",
                   ["sack", "sacked", "terminate", "dismiss", "notice",
                    "fired", "one week", "two weeks", "one month",
                    "length of service", "wetin I fit do"]),
            "13": ("Hours of work and overtime",
                   ["hours", "overtime", "8 hours", "work time", "rest",
                    "shift", "overwork"]),
            "15": ("Periodicity of payment of wages",
                   ["salary", "wages", "pay", "boss", "not paid", "owe",
                    "monthly", "when wages due", "no pay", "two months"]),
            "18": ("Annual holidays with pay",
                   ["leave", "annual leave", "holiday", "vacation", "paid leave",
                    "rest days"]),
            "20": ("Redundancy",
                   ["redundant", "redundancy", "retrench", "lay off",
                    "compensation", "declare redundant"]),
            "54": ("Maternity protection",
                   ["maternity", "pregnant", "nursing", "maternity leave",
                    "woman", "sack pregnant"]),
        },
    },
    {
        "files": ["fccpa_2018.pdf"],
        "id_prefix": "FCCPA2018",
        "source": "Federal Competition and Consumer Protection Act 2018",
        "jurisdiction": "federal",
        "domain": "consumer",
        "note": "Gazette OCR scan: verified plain-string typo fixes in 'fixes', applied in order.",
        "fixes": [
            ("Cornrnission", "Commission"),
            ("Corrmission", "Commission"),
            ("Comnlission", "Commission"),
            ("Conrmission", "Commission"),
            ("Conqumer", "Consumer"),
            ("Consunrer", "Consumer"),
            ("Consurner", "Consumer"),
            ("Contpetition", "Competition"),
            ("Coypeteria", "Competition"),
            ("Coyp etiou", "Competition"),
            ("Coryrpelition", "Competition"),
            ("Establishmenl", "Establishment"),
            ("Enforcernent", "Enforcement"),
            ("Prohibitiorr", "Prohibition"),
            ("Managenrent", "Management"),
            ("Payrnent", "Payment"),
            ("Cornposition", "Composition"),
            ("Sunrmons", "Summons"),
            ("Panr", "Part"),
            ("Penr", "Part"),
            ("Prrnr", "Part"),
            ("Prnr", "Part"),
            ("Porver", "Power"),
            ("shal I", "shall"),
            ("requ ire", "require"),
            ("cond ition", "condition"),
            ("conditiou", "condition"),
            ("thal", "that"),
            ("hranufacturer", "manufacturer"),
            ("impofier", "importer"),
            ("ofgoods", "of goods"),
            ("rvhich", "which"),
            ("tlie", "the"),
            (" ol ", " of "),
            ("consumeE", "consumer"),
            ("men.rbers", "members"),
            ("tlre", "the"),
            ("oftl", "of the"),
            ("Prolection", "Protection"),
            ("Acl,", "Act,"),
            ("20l8", "2018"),
            ("tlr", "the"),
            ("rranner", "manner"),
            ("r.vill", "will"),
            ("inrporter", "importer"),
            ("d istributor", "distributor"),
            ("beconring", "becoming"),
            ("ofany", "of any"),
            ("unloreseen", "unforeseen"),
            ("urarket", "market"),
            ("rranufacturer", "manufacturer"),
            ("ofsuch", "of such"),
            ("imnrediately", "immediately"),
            ("suclr", "such"),
            ("rvithdrarvn", "withdrawn"),
            ("( l)", "(1)"),
            ("I34", "134"),
            ("conrnrits", "commits"),
            ("u rrdertaking", "undertaking"),
            ("v io lates", "violates"),
            ("provis ions", "provisions"),
            ("th is", "this"),
            ("whiclr", "which"),
            ("rvas", "was"),
            ("r.nade", "made"),
            ("rnade", "made"),
            ("m isrepresentation", "misrepresentation"),
            ("nr isl'epresentation", "misrepresentation"),
            ("pafty", "party"),
            ("terr n", "term"),
            ("arrother's", "another's"),
            (",vhich", "which"),
            ("Linbilir]", "Liability"),
            ("obli-eations", "obligations"),
            ("br lau", "by law"),
            ("u hich", "which"),
            ("misreprcsenta-tion", "misrepresentation"),
            ("Er asion", "Evasion"),
            ("b1 means", "by means"),
            ("secondarr", "secondary"),
            ("Conryetition ond Consunter", "Competition and Consumer"),
            ("Liabilitl fbr defectir e", "Liability for defective"),
            ("tiorn", "from"),
            ("suppll", "supply"),
            ("Sat'et1 moniloring", "Safety monitoring"),
            ("Dut)", "Duty"),
            ("soods propelll", "goods properly"),
            ("rl ithclrarr", "withdraw"),
            ("L'ontpctitiotl", "Competition"),
            ("on.sumer", "Consumer"),
            (",4ct", "Act"),
            ("tiom", "from"),
            ("fronr", "from"),
            ("uiidertaking", "undertaking"),
            ("th ird", "third"),
            ("uuless", "unless"),
            ("theat the", "that the"),
            ("linritation", "limitation"),
            ("ofthe", "of the"),
            ("ofthose", "of those"),
            ("120.{l", "120. (1)"),
            ("135.--{ | )", "135. (1)"),
            ("146.-{", "146. ("),
            ("lor cance", "for cance"),
            ("cance llation", "cancellation"),
            ("reservat ion", "reservation"),
            ("su pp lier", "supplier"),
            ("prov ider", "provider"),
            ("Folthe", "For the"),
            ("ofthis", "of this"),
            ("sectiolr", "section"),
            ("unleasouable", "unreasonable"),
            ("il it exceeds", "if it exceeds"),
            ("lair amount", "fair amount"),
            ("circumstarrces", "circumstances"),
            ("ofcancellation", "of cancellation"),
            ("sdrvice", "service"),
            ("betweerl", "between"),
            ("tirne", "time"),
            ("(/)", "(d)"),
            ("(D) ", "(b) "),
            (" (l) ", " (1) "),
            ("tinre", "time"),
            ("rlanner", "manner"),
            ("delermined", "determined"),
            ("rvithin", "within"),
            ("consurner", "consumer"),
            ("thejurisdiction", "the jurisdiction"),
            ("withe appropriate", "with the appropriate"),
            ("j urisd iction", "jurisdiction"),
            ("beerr", "been"),
            ("cornmitted", "committed"),
            ("rvay", "way"),
            ("oftrade", "of trade"),
            ("ofservices", "of services"),
            ("olinfonnation", "or information"),
            ("tothe", "to the"),
            ("irnpose", "impose"),
            ("cornpensation", "compensation"),
            (" Part XVII-ETFoRcEMENT or CoNsuurns' Rrcuts", ""),
        ],
        "region_start": r"EHacreo by the National Assembly|Enacted by the National Assembly",
        "region_end": r"FIRST SCHEDULE",
        "sections": {
            "119": ("Prohibition of tied supply conditions",
                    ["tie", "tied", "bundle", "condition", "must buy",
                     "another goods", "cheat", "consumer", "seller"]),
            "134": ("Duty to label goods properly",
                    ["label", "describe", "goods", "fake", "counterfeit",
                     "package", "manufacturer", "importer"]),
            "140": ("Exclusion of liability for defective goods",
                    ["exclude", "liability", "defective", "contract term",
                     "unfair term", "void", "consumer right"]),
            "143": ("Implied terms in service contracts",
                    ["service", "implied", "repair", "artisan", "mechanic",
                     "tailor", "work", "quality", "time"]),
            "145": ("Defective goods: onus of proof",
                    ["defective", "proof", "prove", "burden", "faulty",
                     "spoilt", "fake", "refund"]),
            "152": ("Investigation of consumer complaints",
                    ["complaint", "commission", "investigate", "report seller",
                     "cheat", "consumer", "where to complain"]),
        },
    },
    {
        "files": ["corrupt_practices_act_2000.pdf"],
        "id_prefix": "CPA2000",
        "source": "Corrupt Practices and Other Related Offences Act 2000",
        "jurisdiction": "federal",
        "domain": "anti_corruption",
        "note": "ICPC offences: bribery, gratification, and how to report.",
        "fixes": [],
        "sections": {
            "8": ("Offence of accepting gratification",
                  ["bribe", "bribery", "corruption", "gratification",
                   "accept", "official", "collect money", "roadblock"]),
            "9": ("Giving or accepting gratification through agent",
                  ["agent", "middleman", "bribe", "gratification",
                   "through", "corruption"]),
            "10": ("Guilty notwithstanding purpose not carried out",
                   ["bribe", "purpose", "not carried out", "guilty",
                   "gratification", "corruption"]),
            "18": ("Bribery of public officer",
                   ["bribe", "public officer", "offer", "solicit",
                   "government official", "corruption"]),
            "19": ("Using office for gratification",
                   ["office", "position", "gratify", "confer", "favour",
                   "unfair advantage", "corruption"]),
            "23": ("Duty to report bribery transactions",
                   ["report", "duty", "inform", "ICPC", "bribe",
                    "how to report", "complain"]),
            "64": ("Protection of informers",
                   ["informer", "protect", "identity", "secret",
                    "whistleblower", "report safely"]),
        },
    },
    {
        "files": ["legal_aid_act_2011.pdf"],
        "id_prefix": "LAA2011",
        "source": "Legal Aid Act 2011",
        "jurisdiction": "federal",
        "domain": "legal_aid",
        "note": "Who qualifies for free legal help and how to apply to the Legal Aid Council.",
        "fixes": [],
        "region_end": r"FIRST SCHEDULE",
        "sections": {
            "8": ("Scope of legal aid",
                  ["scope", "cover", "criminal", "civil", "legal aid",
                   "what cases", "free lawyer"]),
            "10": ("Persons entitled to legal aid",
                   ["free lawyer", "cannot afford", "legal aid", "qualify",
                    "poor", "indigent", "no money", "entitled"]),
            "11": ("Ascertainment of means",
                   ["means", "income", "test", "ascertain", "poor",
                    "qualify", "eligibility", "salary"]),
        },
    },
    {
        "files": ["anti_torture_act_2017.pdf"],
        "id_prefix": "ATA2017",
        "source": "Anti-Torture Act 2017",
        "jurisdiction": "federal",
        "domain": "police_rights",
        "note": ("Scanned but clean typed text, tesseract-OCR'd. 13 sections: "
                 "prohibition, right to complain, right to examination, penalties. "
                 "s.5's opening subject fragment was lost where a margin note "
                 "merged into the line; the operative right text is verbatim."),
        "ocr": True,
        "body_start": r"Commencement",
        "body_end": r"Penalties,\s*Regulatory agency",
        "min_len": 40,
        "fixes": [
            ("1is Act", "This Act"),
            ("of such acts,", "of such acts."),
            ("cruel, human and degrading", "cruel, inhuman and degrading"),
            ("\\CTED", "ENACTED"),
            ("\u201che Government", "The Government"),
            ("cts of torture", "Acts of torture"),
            ("Hed y p & pay", ""),
            ("(J)", "(1)"),
            ("\u20181)", "(1)"),
            ("\u20182)", "(2)"),
            ("(1i1)", "(iii)"),
            ("Cv)", "(iv)"),
            ("yunishment", "punishment"),
            ("vandling", "handling"),
            ("rovernmental", "governmental"),
            ("igencies", "agencies"),
            ("shal] 2e", "shall be"),
            ("raining of law", "training of law"),
            ("ind other persons", "and other persons"),
            ("ndividual", "individual"),
            ("official!", "official"),
            ("~All laws", "All laws"),
            ("emer pency", "emergency"),
        ],
        "anchors": [
            ("The Government shall", "Duty of Government"),
            ("1) Torture is deemed committed", "Acts of torture"),
            ("(1) No exceptional circumstances", "No justification for torture"),
            ("(1) A person who has suffered or alleges", "Right to complain"),
            ("or any interested party on his behalf, may seek legal assistance",
             "Assistance in filing complaint"),
            ("1) A person arrested, detained or under custodial investigation",
             "Right to examination"),
            ("1) A person who actually participates in the infliction of torture",
             "Liability"),
            ("A person who contravenes section 2", "Penalties"),
            ("shall ensure that the function of overseeing", "Regulatory agency"),
            ("other concerned parties shall ensure that education",
             "Education campaign"),
            ("with the approval of the President, make rules and regulations",
             "Rules and regulations"),
            ("All laws, rules and regulations that are contrary to", "Repeal"),
            ("This Act may be cited as the Anti-Torture Act, 2017.", "Citation"),
        ],
        "sections": {
            # number: (unused title slot, keywords) — display titles come
            # from anchors above in arrangement order
            "1": ("Duty of Government",
                  ["government", "duty", "respect rights", "suspects",
                   "detainees", "prisoners", "no harm"]),
            "2": ("Acts of torture",
                  ["torture", "beat", "beaten", "confess", "pain",
                   "suffering", "prohibit", "cell", "electric", "define"]),
            "3": ("No justification for torture",
                  ["justify", "no excuse", "war", "emergency", "secret",
                   "detention", "evidence", "inadmissible"]),
            "4": ("Right to complain",
                  ["complain", "complaint", "report", "competent authority",
                   "protected", "torture", "examine case"]),
            "5": ("Assistance in filing complaint",
                  ["assistance", "help", "file", "Human Rights Commission",
                   "NGO", "complaint", "lawyer"]),
            "6": ("Right to examination",
                  ["examination", "doctor", "medical", "physical",
                   "psychological", "injury", "report", "custody"]),
            "7": ("Liability",
                  ["liable", "principal", "superior", "order", "commanding",
                   "officer", "accessory", "subordinates"]),
            "8": ("Penalties",
                  ["penalty", "punishment", "25 years", "imprisonment",
                   "murder", "death", "compensation", "jail"]),
            "9": ("Regulatory agency",
                  ["attorney-general", "oversee", "unit", "office",
                   "implement", "agency"]),
            "10": ("Education campaign",
                   ["education", "training", "law enforcement", "personnel",
                    "medical", "prohibition"]),
            "11": ("Rules and regulations",
                   ["rules", "regulations", "president", "approval",
                    "implement", "attorney-general"]),
            "12": ("Repeal",
                   ["repeal", "inconsistent", "contrary", "modified"]),
            "13": ("Citation",
                   ["citation", "cited", "Anti-Torture Act"]),
        },
    },
    {
        "files": ["fct_recovery_of_premises_act.pdf"],
        "id_prefix": "FCTRPA",
        "source": "Recovery of Premises Act (Cap 544, FCT Abuja)",
        "jurisdiction": "fct",
        "domain": "tenancy",
        "note": ("Photographed, skewed, watermarked scan: tesseract output is "
                 "garbled on operative sections (e.g. notices to quit), so notice "
                 "periods cannot be verified — SKIPPED per the clean-and-verifiable "
                 "rule. Revisit only from a clean copy of Cap 544."),
        "fixes": [],
        "sections": {
            "3": ("Notice to quit",
                  ["notice to quit", "quit notice", "abuja", "eviction"]),
            "7": ("Recovery of possession",
                  ["recover possession", "abuja", "court", "eviction"]),
        },
    },
]


def _nonws_len(text: str) -> int:
    return len(re.sub(r"\s", "", text))


def _law_text_quality(text: str) -> float:
    """Fraction of characters that are plain word/space/punctuation text.

    Photographed scans come back from PyMuPDF as near-empty or symbol soup;
    gate them out so we never ship unverifiable sections.
    """
    if not text or _nonws_len(text) < 100:
        return 0.0
    good = sum(1 for ch in text if ch.isalnum() or ch.isspace() or ch in ".,;:()[]-–—'\"/§")
    return good / max(len(text), 1)


def _find_law_pdf(filenames: list[str]) -> Path | None:
    # Exact filenames only: a fuzzy match once pulled sections out of
    # Police-Act-2020.pdf for unrelated Acts sharing words like "act".
    for name in filenames:
        path = RAW_DIR / name
        if path.is_file():
            return path
    return None


def _cut_region(raw: str, cfg: dict) -> str:
    """Cut the raw text to the main Act body, dropping front/back matter.

    Arrangements of sections, preceding instruments, schedules and forms
    reuse section numbers (Labour Regulations, ACJA forms, FCCPA
    arrangement, Legal Aid offence schedules), so matching is bounded to
    [region_start, region_end) before section lookup.
    """
    start_pat = cfg.get("region_start")
    if start_pat:
        match = re.search(start_pat, raw)
        if match:
            raw = raw[match.start():]
    end_pat = cfg.get("region_end")
    if end_pat:
        match = re.search(end_pat, raw)
        if match:
            raw = raw[: match.start()]
    return raw


def _pick_section(raw: str, num: str, min_len: int = 300) -> tuple[int, int] | None:
    """Locate one section, robust to arrangement/schedule duplicates.

    Arrangement-of-sections entries and schedule duplicates share numbers
    with real sections but are short. Take the first candidate whose text
    (up to the next header) reaches min_len; else the longest candidate.
    Headers may use the "1.-(1)" dash form (Lagos Tenancy Law).
    Returns (start, end) or None.
    """
    pattern = re.compile(
        r"(?:\b[Ss]ection\s+)?\b" + re.escape(num) + r"\.\s*-?\s*\(?[A-Z0-9(]",
    )
    cands: list[tuple[int, int]] = []
    for match in pattern.finditer(raw):
        start = match.start()
        next_match = re.search(r"\s\b\d{1,3}[A-Z]?\.\s*-?\s*\(?[A-Z0-9(]",
                              raw[match.end():])
        end = match.end() + next_match.start() if next_match else len(raw)
        cands.append((start, end))
    if not cands:
        return None
    for start, end in cands:
        if end - start >= min_len:
            return (start, end)
    return max(cands, key=lambda p: p[1] - p[0])


def extract_generic_law(fitz, cfg: dict, ocr_text: str | None = None) -> tuple[list[dict], str]:
    """Extract configured sections from one Act PDF.

    Returns (sections, report_line). Missing file or unusable text layer
    yields ([], reason) — never fatal. Pass ocr_text for scanned laws
    (tesseract output) instead of reading a text layer.
    """
    if ocr_text is not None:
        raw = ocr_text
        origin = "tesseract OCR"
    else:
        path = _find_law_pdf(cfg["files"])
        if path is None:
            return [], f"SKIP {cfg['source']}: PDF not in data/raw/ ({'/'.join(cfg['files'])})"
        doc = fitz.open(path)
        pages_text = [squash(page.get_text()) for page in doc]
        raw = "\n".join(pages_text)
        if _law_text_quality(raw) < 0.85 or _nonws_len(raw) < 2000:
            return [], (f"SKIP {cfg['source']}: {path.name} photographed scan, "
                         f"no clean verifiable text layer "
                         f"(quality={_law_text_quality(raw):.2f}, "
                         f"non-space chars={_nonws_len(raw)}); not included")
        origin = path.name
    # Squash first so fixes match across line breaks in the raw text.
    raw = squash(raw)
    for old, new in cfg.get("fixes", []):
        raw = raw.replace(old, new)
    raw = _cut_region(raw, cfg)

    sections: list[dict] = []
    missing: list[str] = []
    for num, (title, keywords) in cfg["sections"].items():
        picked = _pick_section(raw, num)
        if picked is None:
            missing.append(num)
            continue
        start, end = picked
        text = squash(raw[start:end])[:4000]
        item = {
            "id": f"{cfg['id_prefix']}-s{num}",
            "source": cfg["source"],
            "section": num,
            "title": title,
            "text": text,
            "keywords": keywords,
            "jurisdiction": cfg["jurisdiction"],
            "domain": cfg["domain"],
        }
        if cfg.get("note"):
            item["applicability"] = cfg["note"]
        sections.append(item)
    report = f"{cfg['source']}: {len(sections)} sections ({origin})"
    if missing:
        report += f" (sections not found, skipped: {', '.join(missing)})"
    return sections, report


def _find_tesseract() -> str | None:
    """Locate a tesseract binary (PATH, TESSERACT_BIN, or user-space build)."""
    import os
    import shutil

    explicit = os.environ.get("TESSERACT_BIN", "")
    if explicit and Path(explicit).is_file():
        return explicit
    found = shutil.which("tesseract")
    if found:
        return found
    fallback = Path("/tmp/tessroot/usr/bin/tesseract")
    if fallback.is_file():
        return str(fallback)
    return None


def ocr_pdf(fitz, path: Path, dpi: int = 300) -> str:
    """OCR a scanned PDF page by page; returns joined text."""
    import os
    import subprocess
    import tempfile

    tess = _find_tesseract()
    if tess is None:
        raise RuntimeError("no tesseract binary found (PATH/TESSERACT_BIN)")
    env = dict(os.environ)
    tessdata = Path(tess).parent.parent / "share" / "tesseract-ocr"
    for cand in (tessdata / "5" / "tessdata", tessdata / "4.00" / "tessdata",
                 tessdata / "tessdata"):
        if (cand / "eng.traineddata").is_file():
            env["TESSDATA_PREFIX"] = str(cand)
            break
    doc = fitz.open(path)
    out: list[str] = []
    with tempfile.TemporaryDirectory(prefix="orb_ocr_") as tmpdir:
        for i, page in enumerate(doc):
            png = str(Path(tmpdir) / f"p{i:03d}.png")
            page.get_pixmap(dpi=dpi).save(png)
            proc = subprocess.run([tess, png, "stdout", "-l", "eng"],
                                  capture_output=True, text=True, env=env,
                                  timeout=300)
            if proc.returncode != 0:
                raise RuntimeError(f"tesseract failed on {path.name} p{i}: "
                                   f"{proc.stderr[:200]}")
            out.append(proc.stdout)
    return "\n".join(out)


def extract_by_anchors(raw: str, anchors: list[tuple[str, str]],
                       cfg: dict, min_len: int = 200) -> tuple[list[dict], list[str]]:
    """Slice a law by ordered body-start anchors (Anti-Torture Act).

    The scan lost section numbers into the margin, so each section is cut
    from its verified opening words to the next section's opening words.
    Anchors are (opening-words, display-title) in arrangement order, so
    numbering follows the arrangement (1..13). Returns (sections, missing).
    """
    body_start = cfg.get("body_start")
    if body_start:
        match = re.search(body_start, raw)
        if match:
            raw = raw[match.start():]
    body_end = cfg.get("body_end")
    if body_end:
        match = re.search(body_end, raw)
        if match:
            raw = raw[: match.start()]
    sections: list[dict] = []
    missing: list[str] = []
    min_len = int(cfg.get("min_len", 200))
    found: list[tuple[int, str, str]] = []  # (arr_idx, pos, title)
    cursor = 0
    for arr_idx, (opening, title) in enumerate(anchors):
        match = re.search(re.escape(opening), raw[cursor:])
        if match is None:
            missing.append(title)
            continue
        found.append((arr_idx, cursor + match.start(), title))
        cursor += match.end()
    for k, (arr_idx, start, title) in enumerate(found):
        num = str(arr_idx + 1)
        end = found[k + 1][1] if k + 1 < len(found) else len(raw)
        text = squash(raw[start:end])[:4000]
        if len(text) < min_len:
            missing.append(f"{title} (too short)")
            continue
        keywords = cfg["sections"].get(num, (None, []))[1]
        item = {
            "id": f"{cfg['id_prefix']}-s{num}",
            "source": cfg["source"],
            "section": num,
            "title": title,
            "text": text,
            "keywords": keywords or [title.lower()],
            "jurisdiction": cfg["jurisdiction"],
            "domain": cfg["domain"],
        }
        if cfg.get("note"):
            item["applicability"] = cfg["note"]
        sections.append(item)
    return sections, missing


def main() -> None:
    fitz = require_fitz()
    constitution_path = find_input("constitution_1999.pdf",
                                   hint="constitution")
    police_path = find_input("police_act_2020.pdf", "Police-Act-2020.pdf",
                             hint="police")

    constitution = extract_constitution(fitz, constitution_path)
    police = extract_police(fitz, police_path)
    sections = constitution + police
    reports: list[str] = []
    for cfg in NEW_LAWS:
        try:
            if cfg.get("ocr"):
                # Scanned law: OCR the PDF, cache the text for audit, then
                # slice by verified body anchors (section numbers live in
                # the margin, lost to OCR).
                path = _find_law_pdf(cfg["files"])
                if path is None:
                    reports.append(
                        f"SKIP {cfg['source']}: PDF not in data/raw/ "
                        f"({'/'.join(cfg['files'])})")
                    continue
                ocr_text = ocr_pdf(fitz, path)
                cache = RAW_DIR / (path.stem + ".ocr.txt")
                cache.write_text(ocr_text, encoding="utf-8")
                ocr_text = squash(ocr_text)
                for old, new in cfg.get("fixes", []):
                    ocr_text = ocr_text.replace(old, new)
                extra, missing = extract_by_anchors(
                    ocr_text, cfg["anchors"], cfg)
                report = (f"{cfg['source']}: {len(extra)} sections "
                          f"(tesseract OCR of {path.name})")
                if missing:
                    report += f" (missing: {', '.join(missing)})"
                reports.append(report)
                sections.extend(extra)
                continue
            extra, report = extract_generic_law(fitz, cfg)
        except Exception as exc:  # noqa: BLE001 - one bad PDF must not kill the build
            reports.append(f"SKIP {cfg['source']}: extractor error: {exc}")
            continue
        reports.append(report)
        sections.extend(extra)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(sections, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    counts: dict[str, int] = {}
    for item in sections:
        counts[item["source"]] = counts.get(item["source"], 0) + 1
    for source in sorted(counts):
        print(f"{source}: {len([s for s in sections if s['source'] == source])} sections")
    for report in reports:
        print(report)
    print(f"Total: {len(sections)} sections -> {OUT_PATH}")


if __name__ == "__main__":
    main()
