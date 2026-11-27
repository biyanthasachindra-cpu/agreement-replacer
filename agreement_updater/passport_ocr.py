"""Local passport OCR with MRZ checksum validation and editable candidates.

Uploaded scans are processed transiently and are not retained by the app. OCR results
are suggestions, never accepted silently: the caller must present them for review.
"""
from __future__ import annotations

import io
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import pymupdf as fitz
from PIL import Image, ImageEnhance, ImageOps

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:  # HEIC support is optional; common image formats still work.
    pass

try:
    import pytesseract
except ImportError:  # Keep the agreement editor usable without the optional OCR extra.
    pytesseract = None

MAX_SCAN_FILE_BYTES = 10 * 1024 * 1024
MAX_OCR_PDF_PAGES = 5
MAX_IMAGE_PIXELS = 40_000_000
MRZ_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
MRZ_CONFUSIONS = {
    "O": ("0",), "0": ("O",),
    "I": ("1",), "1": ("I", "L"), "L": ("1",),
    "Z": ("2",), "2": ("Z",),
    "S": ("5",), "5": ("S",),
    "B": ("8",), "8": ("B",),
    "G": ("6",), "6": ("G",),
}
DIGIT_CONFUSIONS = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6"}


@lru_cache(maxsize=1)
def ocr_status() -> tuple[bool, str]:
    if pytesseract is None:
        return False, "Install the pytesseract Python package and the Tesseract OCR engine."
    try:
        pytesseract.get_tesseract_version()
        languages = pytesseract.get_languages(config="")
        if "eng" not in languages:
            return False, "Tesseract is installed, but its English language data is missing."
    except Exception as exc:  # Missing executable, trained data, or platform setup.
        return False, f"Tesseract OCR is not ready: {exc}"
    return True, "Local Tesseract OCR is ready. Uploaded scans are not saved by the OCR step."


def _clean_mrz_line(value: str) -> str:
    value = value.upper().replace("«", "<").replace("‹", "<").replace("〈", "<").replace("＜", "<")
    return re.sub(r"[^A-Z0-9<]", "", value)


def _as_rgb(image: Image.Image) -> Image.Image:
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        background.alpha_composite(rgba)
        return background.convert("RGB")
    return image.convert("RGB")


def _mrz_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = _clean_mrz_line(raw_line)
        # Some OCR layouts collapse the two machine-readable rows into one long line.
        if line.startswith("P") and line.count("<") >= 2 and len(line) >= 68:
            split_at = 44 if len(line) >= 78 else 36
            lines.extend(part for part in (line[:split_at], line[split_at:]) if len(part) >= 10)
        elif len(line) >= 10:
            lines.append(line)
    return lines


def _mrz_value(char: str) -> int:
    if char == "<":
        return 0
    if "0" <= char <= "9":
        return ord(char) - ord("0")
    if "A" <= char <= "Z":
        return ord(char) - ord("A") + 10
    raise ValueError("Invalid MRZ character")


def _mrz_check_digit(value: str) -> str:
    weights = (7, 3, 1)
    checksum = sum(_mrz_value(char) * weights[index % 3] for index, char in enumerate(value)) % 10
    return str(checksum)


def _expected_digit(char: str) -> str | None:
    char = DIGIT_CONFUSIONS.get(char, char)
    return char if char.isdigit() else None


def _passport_number_candidate(raw_number: str, raw_check: str) -> tuple[str, bool, bool]:
    check_digit = _expected_digit(raw_check)
    number = raw_number[:9].ljust(9, "<")
    if check_digit is None:
        return number.replace("<", ""), False, False
    if _mrz_check_digit(number) == check_digit:
        return number.replace("<", ""), True, False

    # Fix only common OCR confusions and only when a unique check-digit-valid result exists.
    valid_candidates: set[str] = set()
    for index, char in enumerate(number):
        for replacement in MRZ_CONFUSIONS.get(char, ()):
            candidate = number[:index] + replacement + number[index + 1:]
            if _mrz_check_digit(candidate) == check_digit:
                valid_candidates.add(candidate)
    if len(valid_candidates) == 1:
        corrected = next(iter(valid_candidates))
        return corrected.replace("<", ""), True, corrected != number
    return number.replace("<", ""), False, False


def parse_mrz_text(text: str) -> dict[str, Any] | None:
    """Find a TD3/TD2 passport MRZ and return fields plus checksum diagnostics."""
    lines = _mrz_lines(text)
    best: dict[str, Any] | None = None
    best_score = -1
    for first_index, line1 in enumerate(lines):
        if not line1.startswith("P") or line1.count("<") < 1 or len(line1) < 15:
            continue
        # Ordinary passports use TD3 (44 chars); TD2 (36 chars) is also supported.
        expected_length = 44 if len(line1) >= 39 else 36
        if len(line1) > expected_length:
            line1 = line1[:expected_length]
        name_field = line1[5:]
        name_parts = re.split(r"<{2,}", name_field, maxsplit=1)
        surname_field = re.split(r"<{3,}", name_parts[0], maxsplit=1)[0]
        given_field = name_parts[1] if len(name_parts) > 1 else ""
        # Three or more consecutive fillers mark the unused tail of the MRZ name field.
        # Ignore spurious letters that OCR sometimes reads out of that padded region.
        given_field = re.split(r"<{3,}", given_field, maxsplit=1)[0]
        surname = re.sub(r"<+", " ", surname_field).strip()
        given_names = re.sub(r"<+", " ", given_field).strip()
        full_name = " ".join(part for part in (surname, given_names) if part)
        if not full_name:
            continue

        for second_index in range(first_index + 1, min(first_index + 4, len(lines))):
            line2 = lines[second_index]
            if len(line2) < 10 or abs(len(line2) - expected_length) > 5:
                continue
            passport_number, passport_valid, corrected = _passport_number_candidate(line2[:9], line2[9])
            if not passport_number:
                continue
            composite_valid = False
            if len(line2) == 44:
                corrected_line2 = passport_number.ljust(9, "<")[:9] + line2[9:]
                composite_data = corrected_line2[0:10] + corrected_line2[13:20] + corrected_line2[21:43]
                composite_check = _expected_digit(corrected_line2[43])
                composite_valid = composite_check is not None and _mrz_check_digit(composite_data) == composite_check
            score = int(passport_valid) * 100 + int(composite_valid) * 50 + min(len(line2), expected_length)
            result = {
                "employee_name": full_name,
                "passport_number": passport_number,
                "mrz_detected": True,
                "passport_checksum_valid": passport_valid,
                "composite_checksum_valid": composite_valid,
                "ocr_corrected_passport_character": corrected,
            }
            if score > best_score:
                best, best_score = result, score
    return best


def _label_value(text: str, label_pattern: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    for index, line in enumerate(lines):
        match = re.search(label_pattern, line, flags=re.IGNORECASE)
        if not match:
            continue
        remainder = line[match.end():].strip(" \t:;#-–—")
        if remainder:
            return remainder
        for next_line in lines[index + 1:index + 3]:
            candidate = next_line.strip(" \t:;#-–—")
            if candidate:
                return candidate
    return ""


def _extract_visible_labels(text: str) -> tuple[str, str]:
    passport = ""
    passport_match = re.search(
        r"\b(?:passport\s*(?:number|no\.?|#)|document\s*(?:number|no\.?))\s*[:#\-]?\s*([A-Z0-9][A-Z0-9\-]{4,19})\b",
        text,
        flags=re.IGNORECASE,
    )
    if passport_match:
        passport = passport_match.group(1)
    if not passport:
        label = _label_value(text, r"\bpassport\s*(?:number|no\.?|#)\b")
        token = re.search(r"\b[A-Z0-9][A-Z0-9\-]{4,19}\b", label, flags=re.IGNORECASE)
        passport = token.group(0) if token else ""

    surname = _label_value(text, r"\bsurnames?\b")
    given_names = _label_value(text, r"\bgiven\s+names?\b")
    if not surname and not given_names:
        name_label = _label_value(text, r"\b(?:full\s+)?name\b")
        full_name = name_label
    else:
        full_name = " ".join(part for part in (surname, given_names) if part)
    full_name = " ".join(full_name.split())
    passport = " ".join(passport.split()).strip(" :;#-")
    return full_name, passport


def _oriented_image(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    if pytesseract is not None:
        try:
            orientation = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
            rotation = int(orientation.get("rotate", 0)) % 360
            confidence = float(orientation.get("orientation_conf", 0.0))
            if confidence >= 5.0 and rotation in (90, 180, 270):
                image = image.rotate(-rotation, expand=True)
        except Exception:
            # OSD commonly fails on sparse passport photos; the original orientation is still OCR'd.
            pass
    max_dimension = max(image.width, image.height)
    scale = min(3.0, 1800.0 / max(image.width, 1), 3200.0 / max(max_dimension, 1))
    if scale > 1.05:
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
    return image


def _load_scan_pages(raw: bytes, filename: str) -> tuple[list[tuple[Image.Image, str]], str]:
    suffix = Path(filename).suffix.casefold()
    if suffix not in {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".heic", ".heif"}:
        raise ValueError("Use a PDF, JPG, PNG, TIFF, BMP, WEBP, or HEIC passport scan.")
    if not raw:
        raise ValueError("The uploaded scan is empty.")
    if len(raw) > MAX_SCAN_FILE_BYTES:
        raise ValueError(f"Each scan must be no larger than {MAX_SCAN_FILE_BYTES // (1024 * 1024)} MB.")

    pages: list[tuple[Image.Image, str]] = []
    note = ""
    if suffix == ".pdf":
        try:
            pdf = fitz.open(stream=raw, filetype="pdf")
        except Exception as exc:
            raise ValueError("That PDF could not be opened. Try a readable, unencrypted scan.") from exc
        if pdf.is_encrypted:
            pdf.close()
            raise ValueError("Encrypted PDFs are not supported. Upload an unlocked scan.")
        if len(pdf) == 0:
            pdf.close()
            raise ValueError("The uploaded PDF has no pages.")
        page_limit = min(len(pdf), MAX_OCR_PDF_PAGES)
        if len(pdf) > page_limit:
            note = f" Only the first {page_limit} PDF pages were checked."
        for page_index in range(page_limit):
            page = pdf[page_index]
            page_text = page.get_text("text")
            max_dimension = max(float(page.rect.width), float(page.rect.height), 1.0)
            scale = min(2.5, 3200.0 / max_dimension)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
            pages.append((image, page_text))
        pdf.close()
    else:
        try:
            image_file = Image.open(io.BytesIO(raw))
            if image_file.width * image_file.height > MAX_IMAGE_PIXELS:
                image_file.close()
                raise ValueError("The image is too large to OCR. Resize it to under 40 megapixels and try again.")
            image_file.seek(0)
            image_file.load()
            image = _as_rgb(ImageOps.exif_transpose(image_file))
            image_file.close()
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("That image could not be opened. Try a clear JPG, PNG, TIFF, BMP, WEBP, or HEIC scan.") from exc
        pages.append((image, ""))
    return pages, note


def _ocr_image(image: Image.Image) -> tuple[str, str]:
    if pytesseract is None:
        raise RuntimeError("OCR support is not installed. Install pytesseract and the Tesseract OCR engine.")
    image = _oriented_image(image)
    full_text = pytesseract.image_to_string(image, lang="eng", config="--oem 3 --psm 6")
    if not parse_mrz_text(full_text):
        visible_name, visible_passport = _extract_visible_labels(full_text)
        if not (visible_name and visible_passport):
            full_text += "\n" + pytesseract.image_to_string(image, lang="eng", config="--oem 3 --psm 11")

    rotated_full_text: list[str] = []
    rotated_mrz_text: list[str] = []
    orientation_text_by_rotation: dict[int, str] = {}
    if image.width < image.height * 0.85:
        # A portrait phone photo often contains a landscape passport turned sideways.
        for rotation in (90, 270):
            rotated = image.rotate(-rotation, expand=True)
            text = pytesseract.image_to_string(rotated, lang="eng", config="--oem 3 --psm 11")
            candidate = parse_mrz_text(text)
            if not candidate or not candidate.get("passport_checksum_valid"):
                text_block = pytesseract.image_to_string(rotated, lang="eng", config="--oem 3 --psm 6")
                text = text_block + "\n" + text
                candidate = parse_mrz_text(text)
            orientation_text_by_rotation[rotation] = text
            rotated_full_text.append(text)
            if candidate and candidate.get("passport_checksum_valid"):
                return full_text + "\n" + text, text

    gray = ImageOps.autocontrast(ImageOps.grayscale(image))
    mrz_text_parts: list[str] = []
    # The MRZ is near the bottom of a passport page, but scans often have margins or
    # the page occupies only part of the image. Try overlapping lower-page crops.
    for top_fraction in (0.42, 0.54, 0.66, 0.76):
        top = int(gray.height * top_fraction)
        mrz_crop = gray.crop((int(gray.width * 0.01), top, int(gray.width * 0.99), gray.height))
        if mrz_crop.width < 100 or mrz_crop.height < 20:
            continue
        scale = min(4.0, max(1.0, 2800.0 / max(mrz_crop.width, 1)))
        if scale > 1.0:
            mrz_crop = mrz_crop.resize(
                (int(mrz_crop.width * scale), int(mrz_crop.height * scale)),
                Image.Resampling.LANCZOS,
            )
        mrz_crop = ImageEnhance.Contrast(ImageOps.autocontrast(mrz_crop)).enhance(1.7)
        psm6 = pytesseract.image_to_string(
            mrz_crop,
            lang="eng",
            config=f"--oem 3 --psm 6 -c tessedit_char_whitelist={MRZ_WHITELIST}",
        )
        mrz_text_parts.append(psm6)
        candidate = parse_mrz_text("\n".join(mrz_text_parts))
        if candidate and candidate.get("passport_checksum_valid"):
            # PSM 7 reads each MRZ row separately and usually improves name padding.
            psm7 = pytesseract.image_to_string(
                mrz_crop,
                lang="eng",
                config=f"--oem 3 --psm 7 -c tessedit_char_whitelist={MRZ_WHITELIST}",
            )
            mrz_text_parts.append(psm7)
            candidate = parse_mrz_text("\n".join(mrz_text_parts))
            if candidate and candidate.get("passport_checksum_valid") and candidate.get("employee_name"):
                break
        else:
            psm7 = pytesseract.image_to_string(
                mrz_crop,
                lang="eng",
                config=f"--oem 3 --psm 7 -c tessedit_char_whitelist={MRZ_WHITELIST}",
            )
            mrz_text_parts.append(psm7)
            candidate = parse_mrz_text("\n".join(mrz_text_parts))
            if candidate and candidate.get("passport_checksum_valid"):
                break
            # A thresholded pass helps low-contrast or faded MRZ rows.
            threshold = mrz_crop.point(lambda pixel: 255 if pixel > 155 else 0)
            mrz_text_parts.append(pytesseract.image_to_string(
                threshold,
                lang="eng",
                config=f"--oem 3 --psm 6 -c tessedit_char_whitelist={MRZ_WHITELIST}",
            ))
    mrz_text = "\n".join(mrz_text_parts)
    parsed = parse_mrz_text(mrz_text) or parse_mrz_text(full_text)
    if parsed and parsed.get("passport_checksum_valid"):
        return full_text, mrz_text

    # Phone scans may have no EXIF orientation and sparse pages can make Tesseract's
    # orientation detector uncertain. Try the other page orientations before giving up.
    for rotation in (90, 270, 180):
        rotated = image.rotate(-rotation, expand=True)
        if rotation in orientation_text_by_rotation:
            text = orientation_text_by_rotation[rotation]
            candidate = parse_mrz_text(text)
        else:
            text = pytesseract.image_to_string(rotated, lang="eng", config="--oem 3 --psm 11")
            candidate = parse_mrz_text(text)
            if not candidate or not candidate.get("passport_checksum_valid"):
                text_block = pytesseract.image_to_string(rotated, lang="eng", config="--oem 3 --psm 6")
                text = text_block + "\n" + text
                candidate = parse_mrz_text(text)
            orientation_text_by_rotation[rotation] = text
            rotated_full_text.append(text)
        if candidate and candidate.get("passport_checksum_valid"):
            return full_text + "\n" + text, mrz_text + "\n" + text

        rotated_gray = ImageOps.autocontrast(ImageOps.grayscale(rotated))
        top = int(rotated_gray.height * 0.42)
        crop = rotated_gray.crop((int(rotated_gray.width * 0.01), top, int(rotated_gray.width * 0.99), rotated_gray.height))
        if crop.width < 100 or crop.height < 20:
            continue
        scale = min(4.0, max(1.0, 2800.0 / max(crop.width, 1)))
        if scale > 1.0:
            crop = crop.resize((int(crop.width * scale), int(crop.height * scale)), Image.Resampling.LANCZOS)
        crop = ImageEnhance.Contrast(ImageOps.autocontrast(crop)).enhance(1.7)
        crop_text = pytesseract.image_to_string(
            crop,
            lang="eng",
            config=f"--oem 3 --psm 6 -c tessedit_char_whitelist={MRZ_WHITELIST}",
        )
        crop_text += "\n" + pytesseract.image_to_string(
            crop,
            lang="eng",
            config=f"--oem 3 --psm 7 -c tessedit_char_whitelist={MRZ_WHITELIST}",
        )
        rotated_mrz_text.append(crop_text)
        candidate = parse_mrz_text(crop_text)
        if candidate and candidate.get("passport_checksum_valid"):
            return full_text + "\n" + text, mrz_text + "\n" + crop_text

    return (
        full_text + "\n" + "\n".join(rotated_full_text),
        mrz_text + "\n" + "\n".join(rotated_mrz_text),
    )


def extract_passport_data(raw: bytes, filename: str) -> dict[str, Any]:
    """OCR one passport image/PDF, preferring MRZ data with valid check digits."""
    ready, message = ocr_status()
    if not ready:
        raise RuntimeError(message)
    pages, page_note = _load_scan_pages(raw, filename)
    full_text_parts: list[str] = []
    mrz_text_parts: list[str] = []
    candidates: list[dict[str, Any]] = []
    try:
        for image, native_text in pages:
            if native_text:
                full_text_parts.append(native_text)
                native_mrz = parse_mrz_text(native_text)
                if native_mrz:
                    candidates.append(native_mrz)
                    if native_mrz.get("passport_checksum_valid") and native_mrz.get("employee_name"):
                        continue
            full_text, mrz_text = _ocr_image(image)
            if full_text:
                full_text_parts.append(full_text)
            if mrz_text:
                mrz_text_parts.append(mrz_text)
                candidate = parse_mrz_text(mrz_text)
                if candidate:
                    candidates.append(candidate)
            # Also test the full-page OCR output for an MRZ missed by the crop.
            candidate = parse_mrz_text(full_text)
            if candidate:
                candidates.append(candidate)
    except Exception as exc:
        if isinstance(exc, (RuntimeError, ValueError)):
            raise
        raise RuntimeError(f"OCR failed while reading this scan: {exc}") from exc

    best_mrz = max(
        candidates,
        key=lambda item: (
            int(bool(item.get("passport_checksum_valid"))),
            int(bool(item.get("composite_checksum_valid"))),
            len(item.get("employee_name", "")),
        ),
        default=None,
    )
    if best_mrz:
        passed = bool(best_mrz.get("passport_checksum_valid"))
        composite = bool(best_mrz.get("composite_checksum_valid"))
        if passed and composite:
            status = "MRZ detected; passport-number and composite check digits passed. Verify the name against the scan."
            kind = "verified"
        elif passed:
            status = "MRZ detected; passport-number check digit passed. Verify the name and remaining details against the scan."
            kind = "verified"
        else:
            status = "MRZ-like text detected, but the passport-number check digit did not validate. Correct it manually."
            kind = "warning"
        if best_mrz.get("ocr_corrected_passport_character"):
            status += " One ambiguous passport-number character was corrected using its check digit."
        return {
            "employee_name": best_mrz.get("employee_name", ""),
            "passport_number": best_mrz.get("passport_number", ""),
            "mrz_detected": True,
            "passport_checksum_valid": passed,
            "composite_checksum_valid": composite,
            "status": status + page_note,
            "status_kind": kind,
        }

    full_text = "\n".join(full_text_parts)
    name, passport = _extract_visible_labels(full_text)
    if name or passport:
        status = "Visible passport text candidates extracted without a validated MRZ. Check both fields carefully."
        kind = "warning"
    else:
        status = "No reliable fields were recognized. Type the full name and passport number manually below."
        kind = "warning"
    return {
        "employee_name": name,
        "passport_number": passport,
        "mrz_detected": False,
        "passport_checksum_valid": False,
        "composite_checksum_valid": False,
        "status": status + page_note,
        "status_kind": kind,
    }
