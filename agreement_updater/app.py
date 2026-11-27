from __future__ import annotations

import base64
import copy
import csv
import io
import json
import os
import re
import shutil
import sys
import tempfile
import unicodedata
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pymupdf as fitz
from passport_ocr import MAX_SCAN_FILE_BYTES, extract_passport_data, ocr_status
from flask import Flask, abort, flash, jsonify, redirect, render_template, request, send_file, url_for
from werkzeug.exceptions import RequestEntityTooLarge

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "data" / "documents.json"
STATE_PATH = BASE_DIR / "data" / "state.json"
SOURCE_DIR = BASE_DIR / "source_pdfs"
OUTPUT_DIR = BASE_DIR / "generated"
BATCH_DIR = OUTPUT_DIR / "applicant_batches"
ASSET_DIR = BASE_DIR / "assets"
FONT_REGULAR = ASSET_DIR / "DejaVuSans.ttf"
FONT_BOLD = ASSET_DIR / "DejaVuSans-Bold.ttf"
MIN_FONT_SIZE = 7.25
MAX_APPLICANTS_PER_BATCH = 50
MAX_APPLICANT_NAME_LENGTH = 160
MAX_PASSPORT_LENGTH = 40
MAX_OCR_FILES = 20
MAX_OCR_PDF_PAGES = 5
MAX_SCAN_FILE_BYTES = 10 * 1024 * 1024
ALLOWED_SCAN_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".heic", ".heif"}

with CONFIG_PATH.open("r", encoding="utf-8") as f:
    CONFIG = json.load(f)
DOCS = CONFIG["documents"]
DOCS_BY_ID = {doc["id"]: doc for doc in DOCS}

# Page numbers are zero-based. Repeated occurrences are intentionally all updated,
# except for fields whose source text is also a substring of another value.
TARGET_PAGES = {
    "employer_name": [0],
    "signature_employer_name": [3],
    "employer_address": [0],
    "registration_number": [0],
    "company_code": [0],
    "phone": [0],
    "representative_name": [0, 3],
    "employee_name": [3],
    "contract_years": [0],
    "start_date": [0],
    "gross_salary": [1],
    "net_salary": [1],
    "meal_amount": [1],
    "meal_basis": [1],
    "work_hours_day": [1],
    "work_hours_week": [1],
    "annual_leave_days": [1],
    "trial_days": [2],
    "dismissal_notice_days": [2],
    "resignation_notice_days": [2],
    "salary_payment_day": [1],
}

# Each tuple gives the source page and the selected index in search_for() results.
# AFGO's "500Lei" also occurs as the tail of "2500Lei" on the salary line.
MATCH_INDEX = {
    "afgo_bym": {
        "meal_amount": {1: [1]},
        "dismissal_notice_days": {2: [0]},
        "resignation_notice_days": {2: [1]},
    },
    "credit": {
        "dismissal_notice_days": {2: [0]},
        "resignation_notice_days": {2: [1]},
    },
    "bym": {
        "dismissal_notice_days": {2: [0]},
        "resignation_notice_days": {2: [1]},
    },
    "asia": {
        "dismissal_notice_days": {2: [0]},
        "resignation_notice_days": {2: [1]},
    },
    "dnb": {
        "dismissal_notice_days": {2: [0]},
        "resignation_notice_days": {2: [1]},
    },
}

# The AFGO source has actual yellow highlight rectangles for these values.
YELLOW_FIELDS = {
    "afgo_bym": {
        "employee_name",
        "passport_number",
        "start_date",
        "gross_salary",
        "net_salary",
        "meal_amount",
    }
}

FIELD_FORMATS = {
    "employer_name": "plain",
    "signature_employer_name": "plain",
    "employer_address": "plain",
    "registration_number": "plain",
    "company_code": "plain",
    "phone": "plain",
    "representative_name": "plain",
    "employee_name": "plain",
    "passport_number": "plain",
    "workpermit_number": "plain",
    "workpermit_date": "plain",
}

FIELD_LABELS = {
    "employer_name": "employer name",
    "signature_employer_name": "signature employer name",
    "employer_address": "employer/workplace address",
    "registration_number": "trade-register number",
    "company_code": "company code",
    "phone": "employer phone",
    "representative_name": "representative name",
    "employee_name": "employee name",
    "passport_number": "passport number",
    "workpermit_number": "work-permit number",
    "workpermit_date": "work-permit date",
    "contract_years": "contract duration",
    "start_date": "start date",
    "gross_salary": "gross salary",
    "net_salary": "net salary",
    "meal_amount": "meal allowance",
    "meal_basis": "meal frequency",
    "work_hours_day": "daily work hours",
    "work_hours_week": "weekly work hours",
    "annual_leave_days": "annual leave days",
    "trial_days": "trial period",
    "dismissal_notice_days": "dismissal notice",
    "resignation_notice_days": "resignation notice",
    "salary_payment_day": "salary payment timing",
    "job_line": "job category line",
}

app = Flask(__name__)
app.secret_key = os.environ.get("AGREEMENT_UPDATER_SECRET", "local-agreement-editor")
# Allow multipart headers/boundaries on top of the 25 MiB of scan data.
app.config["MAX_CONTENT_LENGTH"] = 26 * 1024 * 1024
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@app.errorhandler(413)
def upload_too_large(_error):
    flash("The upload exceeded the 25 MB combined limit. Reduce the scan sizes or split them into smaller uploads.", "error")
    return redirect(url_for("index"))


def initial_state() -> dict[str, Any]:
    state = {"documents": {}, "last_generated": None}
    for doc in DOCS:
        state["documents"][doc["id"]] = {
            "values": copy.deepcopy(doc["initial"]),
            "sync_signature_employer": bool(doc.get("sync_signature_employer_default", False)),
        }
    return state


def save_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, STATE_PATH)


def load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        state = initial_state()
        save_state(state)
        return state
    try:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        state = initial_state()
        save_state(state)
        return state

    # Add any newly introduced defaults while retaining values already edited.
    for doc in DOCS:
        record = state.setdefault("documents", {}).setdefault(doc["id"], {})
        values = record.setdefault("values", {})
        for key, default in doc["initial"].items():
            values.setdefault(key, default)
        record.setdefault("sync_signature_employer", bool(doc.get("sync_signature_employer_default", False)))
    return state


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_") or "agreement"


def make_display(doc: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for cfg in DOCS:
        current = state["documents"][cfg["id"]]
        initial = cfg["initial"]
        values = current["values"]
        result.append({
            **cfg,
            "values": values,
            "sync_signature_employer": current.get("sync_signature_employer", False),
            "signature_mismatch": initial.get("employer_name") != initial.get("signature_employer_name"),
            "fields_present": {
                field["key"]: bool(initial.get(field["key"]))
                or field["key"] in {"signature_employer_name", "cor_code"}
                for group in CONFIG["field_groups"]
                for field in group["fields"]
            },
            "generated_exists": (OUTPUT_DIR / f"{cfg['id']}_updated.pdf").exists(),
            "output_name": f"{cfg['id']}_updated.pdf",
        })
    return result


def validate_value(field: str, value: str, doc_name: str) -> None:
    if field in {"cor_code", "signature_employer_name"}:
        return
    if not value.strip():
        raise ValueError(f"{doc_name}: {FIELD_LABELS.get(field, field)} cannot be blank.")
    if field == "start_date":
        try:
            datetime.strptime(value.strip(), "%d.%m.%Y")
        except ValueError as exc:
            raise ValueError(f"{doc_name}: enter the start date as DD.MM.YYYY.") from exc
    if field in {"contract_years", "work_hours_day", "work_hours_week", "annual_leave_days", "trial_days", "dismissal_notice_days", "resignation_notice_days"}:
        if not re.fullmatch(r"\d+", value.strip()):
            raise ValueError(f"{doc_name}: {FIELD_LABELS[field]} must be a whole number.")
    if field in {"gross_salary", "net_salary", "meal_amount"} and value.strip():
        if not re.fullmatch(r"\d+(?:[.,]\d{1,2})?", value.strip()):
            raise ValueError(f"{doc_name}: {FIELD_LABELS[field]} must be a number (amount only).")


def validate_state(state: dict[str, Any]) -> None:
    for cfg in DOCS:
        record = state["documents"][cfg["id"]]
        values = record["values"]
        for group in CONFIG["field_groups"]:
            for field_cfg in group["fields"]:
                key = field_cfg["key"]
                # A field blank in the source has no replacement location, except COR and
                # the optional signature line field, which can be added/edited deliberately.
                if not cfg["initial"].get(key) and key not in {"cor_code", "signature_employer_name"}:
                    continue
                val = str(values.get(key, ""))
                if key == "signature_employer_name" and record.get("sync_signature_employer"):
                    continue
                validate_value(key, val, cfg["name"])
        if not values.get("job_title", "").strip():
            raise ValueError(f"{cfg['name']}: job category / title cannot be blank.")
        if not record.get("sync_signature_employer") and cfg["initial"].get("signature_employer_name") and not values.get("signature_employer_name", "").strip():
            raise ValueError(f"{cfg['name']}: fill in the signature employer name or link it to the main employer name.")


def _all_spans(page: fitz.Page) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    data = page.get_text("dict")
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                rows.append(span)
    return rows


def _style_for_rect(page: fitz.Page, rect: fitz.Rect) -> dict[str, Any]:
    best = None
    best_area = -1.0
    for span in _all_spans(page):
        sb = fitz.Rect(span["bbox"])
        inter = sb & rect
        area = max(0.0, inter.width) * max(0.0, inter.height)
        if area > best_area:
            best, best_area = span, area
    if not best or best_area <= 0:
        return {"size": 11.0, "bold": False, "baseline": rect.y1 - 2.1}
    flags = int(best.get("flags", 0))
    font_name = str(best.get("font", ""))
    return {
        "size": float(best.get("size", 11.0)),
        "bold": bool(flags & 16 or "bold" in font_name.lower()),
        "baseline": float(best.get("origin", (rect.x0, rect.y1 - 2.1))[1]),
    }


def _group_rows(rects: list[fitz.Rect]) -> list[list[fitz.Rect]]:
    """Group search rectangles into contiguous multi-line occurrences."""
    if not rects:
        return []
    ordered = sorted(rects, key=lambda r: (round(r.y0, 1), r.x0))
    groups: list[list[fitz.Rect]] = []
    current: list[fitz.Rect] = [ordered[0]]
    last_y = ordered[0].y0
    for rect in ordered[1:]:
        # PDF text line leading in these agreements is approximately 14-17 points.
        if rect.y0 - last_y <= 18.5:
            current.append(rect)
        else:
            groups.append(current)
            current = [rect]
        last_y = rect.y0
    groups.append(current)
    return groups


def _merge_same_line_rects(rects: list[fitz.Rect]) -> list[fitz.Rect]:
    """Join adjacent fragments that PyMuPDF returns for one visually continuous match."""
    rows: list[list[fitz.Rect]] = []
    for rect in sorted(rects, key=lambda r: (r.y0 + r.y1, r.x0)):
        center_y = (rect.y0 + rect.y1) / 2
        target_row = None
        for row in rows:
            row_center = sum((r.y0 + r.y1) / 2 for r in row) / len(row)
            if abs(center_y - row_center) <= 2.2:
                target_row = row
                break
        if target_row is None:
            rows.append([fitz.Rect(rect)])
        else:
            target_row.append(fitz.Rect(rect))

    merged_rows: list[fitz.Rect] = []
    for row in rows:
        row = sorted(row, key=lambda r: r.x0)
        current = fitz.Rect(row[0])
        for rect in row[1:]:
            if rect.x0 <= current.x1 + 1.2:
                current |= rect
            else:
                merged_rows.append(current)
                current = fitz.Rect(rect)
        merged_rows.append(current)
    return sorted(merged_rows, key=lambda r: (round(r.y0, 1), r.x0))


def _selected_matches(page: fitz.Page, query: str, indices: list[int] | None) -> list[fitz.Rect]:
    found = list(page.search_for(query))
    if not found:
        return []
    if indices is not None:
        selected = []
        for idx in indices:
            if idx < 0 or idx >= len(found):
                raise ValueError(f"PDF text locator failed for {query!r}: expected match index {idx}, found {len(found)}.")
            selected.append(found[idx])
        found = selected
    return _merge_same_line_rects(found)


def _font_path(bold: bool) -> str:
    return str(FONT_BOLD if bold else FONT_REGULAR)


def _font_resource_name(bold: bool) -> str:
    # A non-Base14 resource name is required so PyMuPDF embeds the bundled TTF.
    return "AgreementUpdaterBold" if bold else "AgreementUpdaterRegular"


def _fit_size(text: str, font: fitz.Font, base_size: float, width: float) -> float:
    if not text:
        return base_size
    measured = font.text_length(text, fontsize=base_size)
    if measured <= max(width - 0.4, 1.0):
        return base_size
    return base_size * max(width - 0.6, 1.0) / measured


def _split_to_slots(text: str, widths: list[float], font: fitz.Font, fontsize: float) -> list[str] | None:
    """Wrap text across the original matched line widths without touching adjacent text."""
    text = text.strip()
    if not text:
        return [""] * len(widths)
    start = 0
    chunks: list[str] = []
    for width in widths:
        while start < len(text) and text[start].isspace():
            start += 1
        if start >= len(text):
            chunks.append("")
            continue
        remaining = text[start:]
        if font.text_length(remaining, fontsize=fontsize) <= max(width - 0.6, 1.0):
            chunks.append(remaining)
            start = len(text)
            continue

        fit_end = start
        for end in range(start + 1, len(text) + 1):
            candidate = text[start:end]
            if font.text_length(candidate, fontsize=fontsize) <= max(width - 0.6, 1.0):
                fit_end = end
            else:
                break
        if fit_end == start:
            return None

        # Prefer a clean word boundary, then a comma/hyphen boundary. If the input is
        # one long unspaced address token, fall back to a safe character break.
        cut = fit_end
        good = [i for i in range(start + 1, fit_end + 1) if text[i - 1].isspace()]
        if good:
            cut = good[-1]
        else:
            punctuation = [i for i in range(start + 1, fit_end + 1) if text[i - 1] in ",;/-"]
            if punctuation:
                cut = punctuation[-1]
        chunk = text[start:cut].strip()
        if not chunk:
            chunk = text[start:fit_end].strip()
            cut = fit_end
        chunks.append(chunk)
        start = cut
    while start < len(text) and text[start].isspace():
        start += 1
    if start < len(text):
        return None
    return chunks


def _add_white_redaction(page: fitz.Page, rect: fitz.Rect) -> None:
    r = fitz.Rect(rect)
    # A sub-point pad catches antialiased text but avoids consuming the adjacent run.
    r.x0 -= 0.22
    r.y0 -= 0.18
    r.x1 += 0.22
    r.y1 += 0.18
    page.add_redact_annot(r, fill=(1, 1, 1), cross_out=False)


def _draw_slot_group(
    page: fitz.Page,
    rects: list[fitz.Rect],
    styles: list[dict[str, Any]],
    text: str,
    bold: bool,
    highlight: bool,
    field_label: str,
    doc_name: str,
) -> None:
    paired = sorted(zip(rects, styles), key=lambda pair: (round(pair[0].y0, 1), pair[0].x0))
    rects = [pair[0] for pair in paired]
    styles = [pair[1] for pair in paired]
    font_path = _font_path(any(s["bold"] for s in styles) if styles else bold)
    font = fitz.Font(fontfile=font_path)
    base = min((s["size"] for s in styles), default=11.0)
    widths = [max(1.0, r.width) for r in rects]

    # Prefer the original type size; step down only as needed to keep the value inside
    # its original field footprint and preserve the surrounding printed text.
    fontsize = base
    chunks = _split_to_slots(text, widths, font, fontsize)
    while chunks is None and fontsize > MIN_FONT_SIZE:
        fontsize = max(MIN_FONT_SIZE, fontsize - 0.25)
        chunks = _split_to_slots(text, widths, font, fontsize)
    if chunks is None:
        raise ValueError(
            f"{doc_name}: {field_label} is too long for the original space. Shorten it or split it differently."
        )

    for rect, style, chunk in zip(rects, styles, chunks):
        if not chunk:
            continue
        x = rect.x0
        baseline = style["baseline"]
        width = font.text_length(chunk, fontsize=fontsize)
        if highlight:
            highlight_rect = fitz.Rect(
                x - 1.0,
                baseline - fontsize * 0.90,
                min(rect.x1 + 1.0, x + width + 1.0),
                baseline + fontsize * 0.22,
            )
            page.draw_rect(highlight_rect, color=None, fill=(1, 1, 0), overlay=True, width=0)
        page.insert_text(
            (x, baseline),
            chunk,
            fontfile=font_path,
            fontname=_font_resource_name(font_path == str(FONT_BOLD)),
            fontsize=fontsize,
            color=(0, 0, 0),
            overlay=True,
        )


def _replacement_value(field: str, values: dict[str, str], cfg: dict[str, Any]) -> str:
    value = str(values.get(field, "")).strip()
    if field in {"gross_salary", "net_salary", "meal_amount"}:
        return value + cfg.get("money_suffixes", {}).get(field, "")
    if field == "contract_years":
        years = int(value)
        return f"{years:02d} year" if years == 1 else f"{years:02d} years"
    if field in {"work_hours_day", "work_hours_week"}:
        unit = "hours/day" if field == "work_hours_day" else "hours/week"
        return f"{value} {unit}"
    if field in {"dismissal_notice_days", "resignation_notice_days"}:
        return f"{value} business days"
    return value


def _job_line_text(cfg: dict[str, Any], values: dict[str, str]) -> tuple[str, str]:
    layout = cfg["job_layout"]
    title = str(values.get("job_title", "")).strip()
    cor = str(values.get("cor_code", "")).strip()
    if cor:
        dynamic = f"{title}{layout['cor_prefix']}{cor}{layout['cor_suffix']}"
        static = layout["with_cor_static"]
    else:
        dynamic = title
        static = layout["without_cor_static"]
    return dynamic + static, dynamic


def _find_job_line(page: fitz.Page, title: str) -> tuple[fitz.Rect, float, float, bool]:
    title_hits = page.search_for(title)
    if not title_hits:
        raise ValueError(f"Could not locate the source job title {title!r} on page 1.")
    hit = title_hits[0]
    best_line = None
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = "".join(s.get("text", "") for s in spans).strip()
            lb = fitz.Rect(line["bbox"])
            if title.lower() in text.lower() and lb.intersects(hit):
                best_line = line
                break
        if best_line:
            break
    if not best_line:
        raise ValueError(f"Could not identify the full job category line containing {title!r}.")
    line_rect = fitz.Rect(best_line["bbox"])
    spans = best_line.get("spans", [])
    first = spans[0] if spans else {}
    origin = first.get("origin", (line_rect.x0, line_rect.y1 - 2.1))
    size = float(first.get("size", 11.0))
    flags = int(first.get("flags", 0))
    font_name = str(first.get("font", ""))
    bold = bool(flags & 16 or "bold" in font_name.lower())
    return line_rect, float(origin[1]), size, bold


def _register_job_line(
    page: fitz.Page,
    cfg: dict[str, Any],
    values: dict[str, str],
    job_draws: list[dict[str, Any]],
) -> None:
    source_title = cfg["initial"]["job_title"]
    rect, baseline, base_size, bold = _find_job_line(page, source_title)
    new_line, dynamic = _job_line_text(cfg, values)
    job_draws.append({
        "rect": rect,
        "baseline": baseline,
        "base_size": base_size,
        "bold": bold,
        "text": new_line,
        "dynamic": dynamic,
        "highlight": bool(cfg["job_layout"].get("highlight_dynamic", False)),
        "doc_name": cfg["name"],
    })


def _person_tokens(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tokens: list[dict[str, Any]] = []
    for segment in segments:
        for match in re.finditer(r"(\s*)(\S+)", segment["text"]):
            tokens.append({
                "space": match.group(1),
                "text": match.group(2),
                "bold": bool(segment.get("bold", False)),
                "highlight": bool(segment.get("highlight", False)),
            })
    return tokens


def _wrap_person_tokens(
    tokens: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    fonts: dict[bool, fitz.Font],
    fontsize: float,
) -> list[list[dict[str, Any]]] | None:
    lines: list[list[dict[str, Any]]] = [[] for _ in rows]
    row_index = 0
    used = 0.0
    for token in tokens:
        word = token["text"]
        space = token["space"]
        font = fonts[bool(token["bold"])]
        width = font.text_length(space + word, fontsize=fontsize)
        available = rows[row_index]["x1"] - rows[row_index]["x0"]
        if used + width <= available - 0.6:
            lines[row_index].append({**token, "space": space})
            used += width
            continue
        if lines[row_index] and row_index + 1 < len(rows):
            row_index += 1
            used = 0.0
            width = font.text_length(word, fontsize=fontsize)
            available = rows[row_index]["x1"] - rows[row_index]["x0"]
            if width > available - 0.6:
                return None
            lines[row_index].append({**token, "space": ""})
            used = width
            continue
        return None
    return lines


def _register_person_block(page: fitz.Page, cfg: dict[str, Any], values: dict[str, str]) -> dict[str, Any]:
    name_rects = _selected_matches(page, cfg["initial"]["employee_name"], None)
    if not name_rects:
        raise ValueError(f"{cfg['name']}: could not locate the employee name in Clause A.")
    first_name_rect = min(name_rects, key=lambda r: (r.y0, r.x0))
    name_style = _style_for_rect(page, first_name_rect)

    passport_rects = _selected_matches(page, cfg["initial"]["passport_number"], None)
    if not passport_rects:
        raise ValueError(f"{cfg['name']}: could not locate the passport number in Clause A.")
    passport_style = _style_for_rect(page, passport_rects[0])

    end_rects = _selected_matches(page, "which we agree", None)
    if not end_rects:
        raise ValueError(f"{cfg['name']}: could not locate the end of the Clause A identity paragraph.")
    end_y = max(r.y1 for r in end_rects)
    start_y = first_name_rect.y0 - 2.0

    # Find the original text baselines from the PDF and make safe horizontal rows:
    # the first starts immediately after "and the employee"; later rows start at the
    # left body margin. This lets long replacement names flow without covering labels.
    row_groups: list[dict[str, Any]] = []
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            line_rect = fitz.Rect(line["bbox"])
            if line_rect.y1 < start_y or line_rect.y0 > end_y + 1.0:
                continue
            spans = line.get("spans", [])
            baselines = [float(span.get("origin", (0, line_rect.y1 - 2.1))[1]) for span in spans]
            baseline = sum(baselines) / len(baselines) if baselines else line_rect.y1 - 2.1
            existing = next((row for row in row_groups if abs(row["baseline"] - baseline) <= 1.6), None)
            if existing is None:
                row_groups.append({"top": line_rect.y0, "bottom": line_rect.y1, "baseline": baseline})
            else:
                existing["top"] = min(existing["top"], line_rect.y0)
                existing["bottom"] = max(existing["bottom"], line_rect.y1)
                existing["baseline"] = (existing["baseline"] + baseline) / 2

    row_groups.sort(key=lambda row: row["baseline"])
    if not row_groups:
        raise ValueError(f"{cfg['name']}: could not map the employee identity paragraph rows.")
    # Use the exact employee baseline for the first row, avoiding small mixed-font offsets.
    first_row_index = min(range(len(row_groups)), key=lambda i: abs(row_groups[i]["baseline"] - name_style["baseline"]))
    row_groups = row_groups[first_row_index:]
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(row_groups):
        rows.append({
            "x0": first_name_rect.x0 if index == 0 else 72.0,
            "x1": page.rect.width - 72.0,
            "top": row["top"],
            "bottom": row["bottom"],
            "baseline": name_style["baseline"] if index == 0 else row["baseline"],
        })
    if len(rows) < 2:
        raise ValueError(f"{cfg['name']}: not enough mapped lines for the Clause A identity paragraph.")

    afgo_highlight = cfg["id"] == "afgo_bym"
    segments = [
        {"text": values["employee_name"].strip(), "bold": name_style["bold"], "highlight": afgo_highlight},
        {"text": ", identified by passport number", "bold": False},
        {"text": " " + values["passport_number"].strip(), "bold": passport_style["bold"], "highlight": afgo_highlight},
    ]
    if cfg["initial"].get("workpermit_number"):
        segments.extend([
            {"text": ", work permit no", "bold": False},
            {"text": " " + values.get("workpermit_number", "").strip(), "bold": False},
            {"text": " of", "bold": False},
            {"text": " " + values.get("workpermit_date", "").strip(), "bold": False},
        ])
    segments.append({
        "text": ", I compile this individual agreement under the following conditions to which we agree:",
        "bold": False,
    })
    tokens = _person_tokens(segments)
    fonts = {False: fitz.Font(fontfile=str(FONT_REGULAR)), True: fitz.Font(fontfile=str(FONT_BOLD))}
    base_size = min(11.0, float(name_style["size"]))
    fontsize = base_size
    line_tokens = _wrap_person_tokens(tokens, rows, fonts, fontsize)
    while line_tokens is None and fontsize > MIN_FONT_SIZE:
        fontsize = max(MIN_FONT_SIZE, fontsize - 0.25)
        line_tokens = _wrap_person_tokens(tokens, rows, fonts, fontsize)
    if line_tokens is None:
        raise ValueError(f"{cfg['name']}: the employee identity text is too long for the available Clause A lines.")

    # Clear the original identity paragraph from the employee-name start through its
    # closing words. The preceding employer and representative text is left untouched.
    for row in rows:
        _add_white_redaction(page, fitz.Rect(row["x0"], row["top"], row["x1"], row["bottom"]))

    return {
        "rows": rows,
        "line_tokens": line_tokens,
        "fontsize": fontsize,
        "doc_name": cfg["name"],
    }


def _draw_person_block(page: fitz.Page, item: dict[str, Any]) -> None:
    fontsize = item["fontsize"]
    fonts = {False: fitz.Font(fontfile=str(FONT_REGULAR)), True: fitz.Font(fontfile=str(FONT_BOLD))}
    for row, tokens in zip(item["rows"], item["line_tokens"]):
        x = row["x0"]
        for token in tokens:
            bold = bool(token["bold"])
            font = fonts[bold]
            font_path = _font_path(bold)
            space_width = font.text_length(token["space"], fontsize=fontsize)
            x += space_width
            word_width = font.text_length(token["text"], fontsize=fontsize)
            if token["highlight"]:
                highlight_rect = fitz.Rect(
                    x - 0.6,
                    row["baseline"] - fontsize * 0.90,
                    x + word_width + 0.6,
                    row["baseline"] + fontsize * 0.22,
                )
                page.draw_rect(highlight_rect, color=None, fill=(1, 1, 0), overlay=True, width=0)
            page.insert_text(
                (x, row["baseline"]),
                token["text"],
                fontfile=font_path,
                fontname=_font_resource_name(bold),
                fontsize=fontsize,
                color=(0, 0, 0),
                overlay=True,
            )
            x += word_width


def _build_one(cfg: dict[str, Any], record: dict[str, Any], destination: Path) -> None:
    source = SOURCE_DIR / cfg["source_filename"]
    if not source.exists():
        raise FileNotFoundError(f"Missing source PDF: {source}")
    values = {k: str(v) for k, v in record["values"].items()}
    doc = fitz.open(source)
    placements: list[dict[str, Any]] = []
    job_draws: list[dict[str, Any]] = []

    for field, pages in TARGET_PAGES.items():
        source_query = cfg.get("match_texts", {}).get(field, cfg["initial"].get(field, ""))
        if not source_query:
            continue
        replacement = _replacement_value(field, values, cfg)
        if field == "signature_employer_name" and record.get("sync_signature_employer"):
            replacement = str(values.get("employer_name", "")).strip()
        if not replacement:
            continue
        for page_index in pages:
            if page_index >= len(doc):
                continue
            page = doc[page_index]
            indices = MATCH_INDEX.get(cfg["id"], {}).get(field, {}).get(page_index)
            rects = _selected_matches(page, source_query, indices)
            if not rects:
                raise ValueError(
                    f"{cfg['name']}: could not locate source {FIELD_LABELS.get(field, field)} "
                    f"({source_query!r}) on page {page_index + 1}."
                )
            highlight = field in YELLOW_FIELDS.get(cfg["id"], set())
            for group in _group_rows(rects):
                # Capture source styles and baselines before redaction; the text layer is
                # intentionally removed before the replacement text is drawn.
                styles = [_style_for_rect(page, rect) for rect in group]
                for rect in group:
                    _add_white_redaction(page, rect)
                draw_rects = [fitz.Rect(rect) for rect in group]
                draw_styles = [dict(style) for style in styles]
                if field == "employee_name" and page_index == 3:
                    # The signature's employee column is otherwise blank to the right
                    # and below, so allow a long name to wrap there safely.
                    right_edge = page.rect.width - 18.0
                    for rect in draw_rects:
                        rect.x1 = right_edge
                    gap = 14.4
                    for _ in range(2):
                        last_rect = draw_rects[-1]
                        last_style = draw_styles[-1]
                        extra_rect = fitz.Rect(last_rect.x0, last_rect.y0 + gap, right_edge, last_rect.y1 + gap)
                        extra_style = dict(last_style)
                        extra_style["baseline"] = last_style["baseline"] + gap
                        draw_rects.append(extra_rect)
                        draw_styles.append(extra_style)
                elif field in {"signature_employer_name", "representative_name"} and page_index == 3:
                    # Keep clear of the employee signature column at x≈333.
                    for rect in draw_rects:
                        rect.x1 = min(page.rect.width - 18.0, 314.0)
                elif field == "meal_basis" and page_index == 1:
                    # The meal frequency ends its source line; unused whitespace follows.
                    for rect in draw_rects:
                        rect.x1 = page.rect.width - 72.0
                style = styles[0]
                placements.append({
                    "page_index": page_index,
                    "rects": draw_rects,
                    "styles": draw_styles,
                    "text": replacement,
                    "bold": style["bold"],
                    "highlight": highlight,
                    "field": field,
                    "doc_name": cfg["name"],
                })

    # Clause A's identity sentence is a reflowable block, so longer names do not
    # collide with the passport/work-permit wording that follows them.
    person_block = _register_person_block(doc[0], cfg, values)

    # Clause E is rebuilt as one clean line so a changed job title and COR code stay
    # together without colliding with the fixed classification wording.
    _register_job_line(doc[0], cfg, values, job_draws)
    job_rect = job_draws[0]["rect"]
    _add_white_redaction(doc[0], job_rect)

    for page in doc:
        if page.first_annot:
            page.apply_redactions(images=0, graphics=2, text=0)

    # Text overlays are drawn only after every redaction on each page has been applied.
    for placement in placements:
        page = doc[placement["page_index"]]
        _draw_slot_group(
            page,
            placement["rects"],
            placement["styles"],
            placement["text"],
            placement["bold"],
            placement["highlight"],
            FIELD_LABELS.get(placement["field"], placement["field"]),
            placement["doc_name"],
        )

    _draw_person_block(doc[0], person_block)

    for item in job_draws:
        page = doc[0]
        font_path = _font_path(item["bold"])
        font = fitz.Font(fontfile=font_path)
        right_edge = page.rect.width - 72
        available = max(10.0, right_edge - item["rect"].x0)
        font_size = _fit_size(item["text"], font, item["base_size"], available)
        if font_size < MIN_FONT_SIZE:
            raise ValueError(f"{cfg['name']}: job category line is too long for the page.")
        if item["highlight"]:
            dynamic_width = font.text_length(item["dynamic"], fontsize=font_size)
            highlight_rect = fitz.Rect(
                item["rect"].x0 - 1.0,
                item["baseline"] - font_size * 0.90,
                item["rect"].x0 + dynamic_width + 1.0,
                item["baseline"] + font_size * 0.22,
            )
            page.draw_rect(highlight_rect, color=None, fill=(1, 1, 0), overlay=True, width=0)
        page.insert_text(
            (item["rect"].x0, item["baseline"]),
            item["text"],
            fontfile=font_path,
            fontname=_font_resource_name(item["bold"]),
            fontsize=font_size,
            color=(0, 0, 0),
            overlay=True,
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    doc.save(destination, garbage=4, deflate=True)
    doc.close()


def _build_all_temp(state: dict[str, Any]) -> Path:
    temp_root = Path(tempfile.mkdtemp(prefix="agreement_build_", dir=OUTPUT_DIR))
    try:
        for cfg in DOCS:
            output = temp_root / f"{cfg['id']}_updated.pdf"
            _build_one(cfg, state["documents"][cfg["id"]], output)
        return temp_root
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise


def _promote_outputs(temp_root: Path) -> None:
    for cfg in DOCS:
        src = temp_root / f"{cfg['id']}_updated.pdf"
        dest = OUTPUT_DIR / src.name
        os.replace(src, dest)
    shutil.rmtree(temp_root, ignore_errors=True)


def rebuild_and_save(candidate_state: dict[str, Any]) -> None:
    validate_state(candidate_state)
    temp_root = _build_all_temp(candidate_state)
    candidate_state["last_generated"] = datetime.now().astimezone().isoformat(timespec="minutes")
    try:
        save_state(candidate_state)
        _promote_outputs(temp_root)
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise


def _candidate_from_form(form) -> dict[str, Any]:
    current = load_state()
    candidate = copy.deepcopy(current)
    for cfg in DOCS:
        doc_id = cfg["id"]
        record = candidate["documents"][doc_id]
        old_values = record["values"]
        for group in CONFIG["field_groups"]:
            for field_cfg in group["fields"]:
                key = field_cfg["key"]
                name = f"v__{doc_id}__{key}"
                if name in form:
                    old_values[key] = form.get(name, "").strip()
        record["sync_signature_employer"] = form.get(f"sync__{doc_id}") == "on"
    return candidate


def _short_bulk_fields() -> list[dict[str, str]]:
    # Bulk options include only fields with a real template location in at least one PDF.
    fields = []
    for group in CONFIG["field_groups"]:
        for field in group["fields"]:
            fields.append({"key": field["key"], "label": field["label"]})
    return fields


def _bulk_state(field: str, value: str) -> dict[str, Any]:
    current = load_state()
    candidate = copy.deepcopy(current)
    eligible = 0
    for cfg in DOCS:
        initial = cfg["initial"]
        if field not in initial:
            continue
        if not initial.get(field) and field not in {"cor_code", "signature_employer_name"}:
            continue
        candidate["documents"][cfg["id"]]["values"][field] = value.strip()
        if field == "employer_name":
            # A deliberate cross-document employer-name broadcast also links signature blocks.
            candidate["documents"][cfg["id"]]["sync_signature_employer"] = True
        eligible += 1
    if not eligible:
        raise ValueError("That field has no editable location in the agreements.")
    return candidate


def _clean_applicant_details(full_name: str, passport_number: str, source_label: str) -> dict[str, str]:
    name = " ".join(str(full_name or "").split())
    passport = " ".join(str(passport_number or "").split())
    if not name or not passport:
        raise ValueError(f"{source_label}: enter both the applicant full name and passport number.")
    if len(name) > MAX_APPLICANT_NAME_LENGTH:
        raise ValueError(f"{source_label}: applicant full name is too long (maximum {MAX_APPLICANT_NAME_LENGTH} characters).")
    if len(passport) > MAX_PASSPORT_LENGTH:
        raise ValueError(f"{source_label}: passport number is too long (maximum {MAX_PASSPORT_LENGTH} characters).")
    if not any(char.isalpha() for char in name):
        raise ValueError(f"{source_label}: enter a valid full name.")
    if not any(char.isalnum() for char in passport):
        raise ValueError(f"{source_label}: enter a valid passport number.")
    return {"employee_name": name, "passport_number": passport}


def _normalize_csv_header(header: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(header or "").strip().casefold()).strip("_")


def _applicants_from_csv(upload) -> list[dict[str, str]]:
    if upload is None or not upload.filename:
        return []
    raw = upload.read()
    if not raw.strip():
        raise ValueError("The uploaded CSV is empty.")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = raw.decode("cp1252")
        except UnicodeDecodeError as exc:
            raise ValueError("The CSV must use UTF-8 or Windows-1252 text encoding.") from exc

    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    if not reader.fieldnames:
        raise ValueError("The CSV needs a header row with full_name and passport_number columns.")

    name_aliases = {"full_name", "applicant_full_name", "employee_full_name", "applicant_name", "employee_name", "name"}
    passport_aliases = {"passport_number", "passport_no", "passport", "passport_id"}
    normalized = [(header, _normalize_csv_header(header)) for header in reader.fieldnames if header]
    name_column = next((header for header, key in normalized if key in name_aliases), None)
    passport_column = next((header for header, key in normalized if key in passport_aliases), None)
    if not name_column or not passport_column:
        raise ValueError("CSV headers must include full_name and passport_number (employee_name and passport_no are also accepted).")

    applicants: list[dict[str, str]] = []
    for row_number, row in enumerate(reader, start=2):
        name = str(row.get(name_column) or "").strip()
        passport = str(row.get(passport_column) or "").strip()
        if not name and not passport:
            continue
        details = _clean_applicant_details(name, passport, f"CSV row {row_number}")
        details["source_label"] = f"CSV row {row_number}"
        applicants.append(details)
    return applicants


def _collect_applicants(form, upload) -> list[dict[str, str]]:
    applicants: list[dict[str, str]] = []
    names = form.getlist("manual_full_name")
    passports = form.getlist("manual_passport_number")
    for index in range(max(len(names), len(passports))):
        name = names[index] if index < len(names) else ""
        passport = passports[index] if index < len(passports) else ""
        if not str(name or "").strip() and not str(passport or "").strip():
            continue
        details = _clean_applicant_details(name, passport, f"Manual applicant {index + 1}")
        details["source_label"] = f"Manual applicant {index + 1}"
        applicants.append(details)

    applicants.extend(_applicants_from_csv(upload))
    if not applicants:
        raise ValueError("Add at least one applicant manually or upload a CSV.")
    if len(applicants) > MAX_APPLICANTS_PER_BATCH:
        raise ValueError(f"A batch can contain at most {MAX_APPLICANTS_PER_BATCH} applicants.")

    seen_passports: dict[str, int] = {}
    for index, applicant in enumerate(applicants, start=1):
        normalized_passport = "".join(char for char in applicant["passport_number"].casefold() if char.isalnum())
        if normalized_passport in seen_passports:
            first = seen_passports[normalized_passport]
            raise ValueError(f"Applicants {first} and {index} have the same passport number; check the entries.")
        seen_passports[normalized_passport] = index
    return applicants


def _applicant_filename_slug(full_name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", full_name).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9]+", "_", ascii_name).strip("_")[:42].strip("_")
    return slug or "applicant"


def _build_applicant_batch(template_id: str, applicants: list[dict[str, str]]) -> dict[str, Any]:
    if template_id not in DOCS_BY_ID:
        raise ValueError("Choose one of the available agreement templates.")
    if not applicants or len(applicants) > MAX_APPLICANTS_PER_BATCH:
        raise ValueError(f"A batch must contain between 1 and {MAX_APPLICANTS_PER_BATCH} applicants.")

    state = load_state()
    validate_state(state)
    cfg = DOCS_BY_ID[template_id]
    template_record = state["documents"][template_id]
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    batch_id = uuid.uuid4().hex
    stage = Path(tempfile.mkdtemp(prefix=f".{batch_id}-", dir=BATCH_DIR))
    final_dir = BATCH_DIR / batch_id
    files: list[dict[str, Any]] = []
    try:
        source_doc = fitz.open(SOURCE_DIR / cfg["source_filename"])
        expected_pages = len(source_doc)
        source_doc.close()
        for index, raw_applicant in enumerate(applicants, start=1):
            source_label = raw_applicant.get("source_label", f"Applicant {index}")
            applicant = _clean_applicant_details(
                raw_applicant.get("employee_name", ""),
                raw_applicant.get("passport_number", ""),
                source_label,
            )
            record = copy.deepcopy(template_record)
            record["values"]["employee_name"] = applicant["employee_name"]
            record["values"]["passport_number"] = applicant["passport_number"]
            filename = f"{index:03d}_{_applicant_filename_slug(applicant['employee_name'])}_{template_id}.pdf"
            destination = stage / filename
            _build_one(cfg, record, destination)

            generated = fitz.open(destination)
            page_count = len(generated)
            page_texts = [page.get_text().strip() for page in generated]
            readable_pages = all(bool(text) for text in page_texts)
            normalized_pdf_text = " ".join(" ".join(page_texts).split()).casefold()
            required_values = (applicant["employee_name"], applicant["passport_number"])
            values_present = all(" ".join(value.split()).casefold() in normalized_pdf_text for value in required_values)
            generated.close()
            if page_count != expected_pages or not readable_pages or not values_present:
                raise ValueError(f"Applicant {index}: generated PDF is incomplete or is missing applicant details; no batch files were published.")
            files.append({
                "filename": filename,
                "employee_name": applicant["employee_name"],
                "passport_number": applicant["passport_number"],
                "page_count": page_count,
            })

        zip_filename = "applicant_agreements.zip"
        with zipfile.ZipFile(stage / zip_filename, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for item in files:
                archive.write(stage / item["filename"], arcname=item["filename"])

        manifest = {
            "id": batch_id,
            "created_at": datetime.now().astimezone().isoformat(timespec="minutes"),
            "template_id": template_id,
            "template_name": cfg["name"],
            "applicant_count": len(files),
            "files": files,
            "zip_filename": zip_filename,
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(stage, final_dir)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _load_applicant_batch(batch_id: str) -> dict[str, Any] | None:
    if not re.fullmatch(r"[0-9a-f]{32}", batch_id):
        return None
    manifest_path = BATCH_DIR / batch_id / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if manifest.get("id") != batch_id or not isinstance(manifest.get("files"), list):
        return None
    return manifest


def _recent_applicant_batches(limit: int = 5) -> list[dict[str, Any]]:
    if not BATCH_DIR.exists():
        return []
    batches = []
    for folder in BATCH_DIR.iterdir():
        if not folder.is_dir():
            continue
        manifest = _load_applicant_batch(folder.name)
        if manifest:
            batches.append(manifest)
    batches.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return batches[:limit]


def _batch_pdf_path(batch_id: str, filename: str) -> Path | None:
    if not re.fullmatch(r"[A-Za-z0-9_-]+\.pdf", filename):
        return None
    manifest = _load_applicant_batch(batch_id)
    if not manifest or not any(item.get("filename") == filename for item in manifest["files"]):
        return None
    path = BATCH_DIR / batch_id / filename
    return path if path.is_file() else None


def _render_visual_pdf(path: Path, title: str, download_url: str, back_url: str):
    document = fitz.open(path)
    pages = []
    try:
        for index, page in enumerate(document):
            max_dimension = max(float(page.rect.width), float(page.rect.height), 1.0)
            scale = min(1.55, 1500.0 / max_dimension)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            image_data = base64.b64encode(pixmap.tobytes("png")).decode("ascii")
            pages.append({"number": index + 1, "data_uri": f"data:image/png;base64,{image_data}"})
    finally:
        document.close()
    return render_template(
        "pdf_visual_preview.html",
        title=title,
        pages=pages,
        download_url=download_url,
        back_url=back_url,
    )


def _outputs_ready() -> bool:
    return all((OUTPUT_DIR / f"{cfg['id']}_updated.pdf").exists() for cfg in DOCS)


@app.get("/")
def index():
    state = load_state()
    docs = make_display(CONFIG, state)
    ocr_ready, ocr_message = ocr_status()
    return render_template(
        "index.html",
        docs=docs,
        field_groups=CONFIG["field_groups"],
        bulk_fields=_short_bulk_fields(),
        last_generated=state.get("last_generated"),
        outputs_ready=_outputs_ready(),
        recent_batches=_recent_applicant_batches(),
        ocr_ready=ocr_ready,
        ocr_message=ocr_message,
    )


@app.post("/generate-applicant-batch")
def generate_applicant_batch():
    try:
        template_id = request.form.get("template_id", "").strip()
        applicants = _collect_applicants(request.form, request.files.get("applicants_csv"))
        batch = _build_applicant_batch(template_id, applicants)
        flash(f"Generated and checked {batch['applicant_count']} complete PDF(s). Preview them below before downloading.", "success")
        return redirect(url_for("applicant_batch_review", batch_id=batch["id"]))
    except Exception as exc:
        flash(f"Applicant PDFs were not generated: {exc}", "error")
        return redirect(url_for("index"))


@app.get("/applicant-csv-template")
def applicant_csv_template():
    content = "full_name,passport_number" + chr(10) + "Example Applicant,P1234567" + chr(10)
    return app.response_class(
        content,
        mimetype="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=applicants_template.csv"},
    )


@app.post("/extract-passports")
def extract_passports():
    try:
        template_id = request.form.get("template_id", "").strip()
        if template_id not in DOCS_BY_ID:
            raise ValueError("Choose an agreement template before uploading passport scans.")
        ready, message = ocr_status()
        if not ready:
            raise ValueError(message)
        uploads = [item for item in request.files.getlist("passport_scans") if item and item.filename]
        if not uploads:
            raise ValueError("Choose at least one passport scan to extract.")
        if len(uploads) > MAX_OCR_FILES:
            raise ValueError(f"Upload no more than {MAX_OCR_FILES} scans at one time.")

        candidates = []
        for index, upload in enumerate(uploads, start=1):
            filename = Path(upload.filename.replace("\\", "/")).name.strip() or f"Passport scan {index}"
            raw = upload.read()
            if len(raw) > MAX_SCAN_FILE_BYTES:
                raise ValueError(f"{filename}: each scan must be no larger than {MAX_SCAN_FILE_BYTES // (1024 * 1024)} MB.")
            try:
                candidate = extract_passport_data(raw, filename)
            except ValueError as exc:
                raise ValueError(f"{filename}: {exc}") from exc
            except Exception as exc:
                candidate = {
                    "employee_name": "",
                    "passport_number": "",
                    "mrz_detected": False,
                    "passport_checksum_valid": False,
                    "composite_checksum_valid": False,
                    "status": f"OCR could not read this file: {exc}. Enter the fields manually.",
                    "status_kind": "warning",
                }
            if candidate.get("status_kind") == "warning":
                app.logger.warning(
                    "Passport OCR needs manual review for scan %d (format=%s; name_candidate=%s; passport_candidate=%s; MRZ=%s)",
                    index,
                    Path(filename).suffix.casefold() or "unknown",
                    bool(candidate.get("employee_name")),
                    bool(candidate.get("passport_number")),
                    bool(candidate.get("mrz_detected")),
                )
            candidate["scan_name"] = filename
            candidate["scan_index"] = index
            candidates.append(candidate)

        return render_template(
            "ocr_review.html",
            candidates=candidates,
            template_id=template_id,
            template_name=DOCS_BY_ID[template_id]["name"],
            max_applicants=MAX_APPLICANTS_PER_BATCH,
        )
    except RequestEntityTooLarge:
        flash("The upload exceeded the 25 MB combined limit. Reduce the scan sizes or split them into smaller uploads.", "error")
        return redirect(url_for("index"))
    except Exception as exc:
        flash(f"Passport extraction could not be completed: {exc}", "error")
        return redirect(url_for("index"))


@app.get("/batches/<batch_id>")
def applicant_batch_review(batch_id: str):
    batch = _load_applicant_batch(batch_id)
    if not batch:
        abort(404)
    for item in batch["files"]:
        if not _batch_pdf_path(batch_id, item.get("filename", "")):
            abort(404)
    return render_template("batch_review.html", batch=batch)


@app.get("/batches/<batch_id>/preview/<filename>")
def applicant_batch_preview(batch_id: str, filename: str):
    path = _batch_pdf_path(batch_id, filename)
    if not path:
        abort(404)
    return send_file(path, as_attachment=False, download_name=filename, mimetype="application/pdf", conditional=True)


@app.get("/batches/<batch_id>/visual/<filename>")
def applicant_batch_visual_preview(batch_id: str, filename: str):
    path = _batch_pdf_path(batch_id, filename)
    if not path:
        abort(404)
    manifest = _load_applicant_batch(batch_id)
    item = next((value for value in manifest["files"] if value.get("filename") == filename), None)
    if not item:
        abort(404)
    return _render_visual_pdf(
        path,
        f"{item['employee_name']} · {manifest['template_name']}",
        url_for("applicant_batch_download", batch_id=batch_id, filename=filename),
        url_for("applicant_batch_review", batch_id=batch_id),
    )


@app.get("/batches/<batch_id>/download/<filename>")
def applicant_batch_download(batch_id: str, filename: str):
    path = _batch_pdf_path(batch_id, filename)
    if not path:
        abort(404)
    return send_file(path, as_attachment=True, download_name=filename, mimetype="application/pdf", conditional=True)


@app.get("/batches/<batch_id>/download-zip")
def applicant_batch_download_zip(batch_id: str):
    batch = _load_applicant_batch(batch_id)
    if not batch:
        abort(404)
    filename = batch.get("zip_filename", "")
    if filename != "applicant_agreements.zip":
        abort(404)
    path = BATCH_DIR / batch_id / filename
    if not path.is_file():
        abort(404)
    return send_file(
        path,
        as_attachment=True,
        download_name=f"applicant_agreements_{batch_id[:8]}.zip",
        mimetype="application/zip",
        conditional=True,
    )


@app.post("/save")
def save():
    try:
        candidate = _candidate_from_form(request.form)
        rebuild_and_save(candidate)
        flash("Saved. All five agreements have been rebuilt from the original PDFs.", "success")
    except Exception as exc:
        flash(f"No PDFs were replaced: {exc}", "error")
    return redirect(url_for("index"))


@app.post("/bulk")
def bulk():
    field = request.form.get("field", "").strip()
    value = request.form.get("value", "").strip()
    known = {f["key"] for group in CONFIG["field_groups"] for f in group["fields"]}
    if field not in known:
        flash("Choose a valid field for the bulk update.", "error")
        return redirect(url_for("index"))
    try:
        candidate = _bulk_state(field, value)
        rebuild_and_save(candidate)
        flash(f"Bulk update applied to all agreements that contain {FIELD_LABELS.get(field, field)}; PDFs regenerated.", "success")
    except Exception as exc:
        flash(f"Bulk update was not applied: {exc}", "error")
    return redirect(url_for("index"))


@app.post("/reset")
def reset():
    try:
        candidate = initial_state()
        rebuild_and_save(candidate)
        flash("Restored the five source values and regenerated clean updated copies.", "success")
    except Exception as exc:
        flash(f"Could not reset the agreements: {exc}", "error")
    return redirect(url_for("index"))


@app.get("/download/<doc_id>")
def download(doc_id: str):
    if doc_id not in DOCS_BY_ID:
        abort(404)
    path = OUTPUT_DIR / f"{doc_id}_updated.pdf"
    if not path.exists():
        abort(404)
    return send_file(path, as_attachment=True, download_name=f"{safe_name(DOCS_BY_ID[doc_id]['name'])}_updated.pdf", mimetype="application/pdf")


@app.get("/preview/<doc_id>")
def preview(doc_id: str):
    if doc_id not in DOCS_BY_ID:
        abort(404)
    path = OUTPUT_DIR / f"{doc_id}_updated.pdf"
    if not path.exists():
        abort(404)
    return send_file(path, as_attachment=False, download_name=f"{doc_id}_updated.pdf", mimetype="application/pdf")


@app.get("/preview-pages/<doc_id>")
def visual_document_preview(doc_id: str):
    if doc_id not in DOCS_BY_ID:
        abort(404)
    path = OUTPUT_DIR / f"{doc_id}_updated.pdf"
    if not path.is_file():
        abort(404)
    return _render_visual_pdf(
        path,
        f"{DOCS_BY_ID[doc_id]['name']} agreement",
        url_for("download", doc_id=doc_id),
        url_for("index"),
    )


@app.get("/download-all")
def download_all():
    if not _outputs_ready():
        abort(404)
    zip_path = OUTPUT_DIR / "all_updated_agreements.zip"
    import zipfile
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for cfg in DOCS:
            p = OUTPUT_DIR / f"{cfg['id']}_updated.pdf"
            archive.write(p, arcname=f"{safe_name(cfg['name'])}_updated.pdf")
    return send_file(zip_path, as_attachment=True, download_name="all_updated_agreements.zip", mimetype="application/zip")


@app.get("/health")
def health():
    ocr_ready, _ = ocr_status()
    return jsonify({"status": "ok", "documents": len(DOCS), "outputs_ready": _outputs_ready(), "ocr_available": ocr_ready})


def ensure_initial_outputs() -> None:
    state = load_state()
    if not _outputs_ready():
        rebuild_and_save(state)


if __name__ == "__main__":
    if "--check" in sys.argv:
        candidate = load_state()
        validate_state(candidate)
        temp_root = _build_all_temp(candidate)
        print("Preflight passed; PDFs built in:", temp_root)
        for p in sorted(temp_root.glob("*.pdf")):
            print(f"  {p.name}: {p.stat().st_size:,} bytes")
        shutil.rmtree(temp_root, ignore_errors=True)
    else:
        ensure_initial_outputs()
        app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
