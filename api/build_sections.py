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


def main() -> None:
    fitz = require_fitz()
    constitution_path = find_input("constitution_1999.pdf",
                                   hint="constitution")
    police_path = find_input("police_act_2020.pdf", "Police-Act-2020.pdf",
                             hint="police")

    constitution = extract_constitution(fitz, constitution_path)
    police = extract_police(fitz, police_path)
    sections = constitution + police

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(sections, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    counts: dict[str, int] = {}
    for item in sections:
        counts[item["source"]] = counts.get(item["source"], 0) + 1
    for source in sorted(counts):
        print(f"{source}: {len([s for s in sections if s['source'] == source])} sections")
    print(f"Total: {len(sections)} sections -> {OUT_PATH}")


if __name__ == "__main__":
    main()
