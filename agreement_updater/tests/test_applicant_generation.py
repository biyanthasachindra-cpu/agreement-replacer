import io
import re
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw, ImageFont

import app as updater
from passport_ocr import ocr_status


class ApplicantGenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="agreement-batch-tests-")
        self.batch_dir_patch = patch.object(updater, "BATCH_DIR", Path(self.temp.name) / "batches")
        self.batch_dir_patch.start()
        self.client = updater.app.test_client()

    def tearDown(self):
        self.batch_dir_patch.stop()
        self.temp.cleanup()

    def _zip_files(self, response):
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "application/zip")
        data = response.get_data()
        response.close()
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            return {name: archive.read(name) for name in archive.namelist()}

    def test_single_manual_applicant_preview_and_downloads(self):
        response = self.client.post("/generate-applicant-batch", data={
            "template_id": "afgo_bym",
            "manual_full_name": "TEST APPLICANT ASANKA PERSON",
            "manual_passport_number": "PTEST12345",
        })
        self.assertEqual(response.status_code, 302)
        batch_url = response.headers["Location"]
        batch_id = batch_url.rsplit("/", 1)[-1]
        review = self.client.get(batch_url)
        self.assertEqual(review.status_code, 200)
        self.assertIn(b"TEST APPLICANT ASANKA PERSON", review.data)
        self.assertIn(b"Inline PDF preview", review.data)
        self.assertIn(b"/visual/", review.data)

        manifest = updater._load_applicant_batch(batch_id)
        self.assertEqual(manifest["applicant_count"], 1)
        filename = manifest["files"][0]["filename"]
        visual_preview = self.client.get(f"/batches/{batch_id}/visual/{filename}")
        self.assertEqual(visual_preview.status_code, 200)
        self.assertEqual(visual_preview.mimetype, "text/html")
        self.assertIn(b"data:image/png;base64,", visual_preview.data)
        preview = self.client.get(f"/batches/{batch_id}/preview/{filename}")
        self.assertEqual(preview.status_code, 200)
        self.assertEqual(preview.mimetype, "application/pdf")
        self.assertIn("inline", preview.headers["Content-Disposition"])
        preview_data = preview.get_data()
        preview.close()
        pdf = updater.fitz.open(stream=preview_data, filetype="pdf")
        self.assertEqual(len(pdf), 4)
        pdf_text = "\n".join(page.get_text() for page in pdf)
        self.assertIn("TEST APPLICANT ASANKA PERSON", pdf_text)
        self.assertIn("PTEST12345", pdf_text)
        pdf.close()

        download = self.client.get(f"/batches/{batch_id}/download/{filename}")
        self.assertEqual(download.status_code, 200)
        self.assertIn("attachment", download.headers["Content-Disposition"])
        download.get_data()
        download.close()
        files = self._zip_files(self.client.get(f"/batches/{batch_id}/download-zip"))
        self.assertEqual(list(files), [filename])
        self.assertTrue(files[filename].startswith(b"%PDF"))

    def test_csv_import_combined_with_manual_rows(self):
        csv_file = io.BytesIO(
            b"full_name,passport_number\n"
            b"CSV PERSON ONE,C12345678\n"
            b"CSV PERSON TWO,C87654321\n"
        )
        response = self.client.post("/generate-applicant-batch", data={
            "template_id": "bym",
            "manual_full_name": ["MANUAL APPLICANT TWO", ""],
            "manual_passport_number": ["M7777777", ""],
            "applicants_csv": (csv_file, "applicants.csv"),
        }, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 302)
        batch_id = response.headers["Location"].rsplit("/", 1)[-1]
        review = self.client.get(response.headers["Location"])
        self.assertEqual(review.status_code, 200)
        for name in ("MANUAL APPLICANT TWO", "CSV PERSON ONE", "CSV PERSON TWO"):
            self.assertIn(name.encode(), review.data)

        manifest = updater._load_applicant_batch(batch_id)
        self.assertEqual(manifest["applicant_count"], 3)
        files = self._zip_files(self.client.get(f"/batches/{batch_id}/download-zip"))
        self.assertEqual(len(files), 3)
        for item in manifest["files"]:
            pdf = updater.fitz.open(stream=files[item["filename"]], filetype="pdf")
            self.assertEqual(len(pdf), 4)
            text = "\n".join(page.get_text() for page in pdf)
            self.assertIn(item["employee_name"], text)
            self.assertIn(item["passport_number"], text)
            pdf.close()

    @unittest.skipUnless(ocr_status()[0], "Tesseract OCR engine is not installed")
    def test_passport_scan_ocr_review_then_generates_pdf(self):
        line1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
        line2 = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"
        image = Image.new("RGB", (1800, 560), "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(str(updater.BASE_DIR / "assets" / "DejaVuSans.ttf"), 48)
        draw.text((35, 350), line1, font=font, fill="black")
        draw.text((35, 430), line2, font=font, fill="black")
        upload = io.BytesIO()
        image.save(upload, format="PNG")

        review = self.client.post("/extract-passports", data={
            "template_id": "bym",
            "passport_scans": (io.BytesIO(upload.getvalue()), "synthetic_passport.png"),
        }, content_type="multipart/form-data")
        self.assertEqual(review.status_code, 200)
        self.assertIn(b"ERIKSSON ANNA MARIA", review.data)
        self.assertIn(b"L898902C3", review.data)
        self.assertIn(b"check digits passed", review.data)
        name_match = re.search(rb'name="manual_full_name"[^>]*value="([^"]*)"', review.data)
        passport_match = re.search(rb'name="manual_passport_number"[^>]*value="([^"]*)"', review.data)
        self.assertIsNotNone(name_match, "OCR name candidate must reach the editable review form")
        self.assertIsNotNone(passport_match, "OCR passport candidate must reach the editable review form")
        extracted_name = name_match.group(1).decode("utf-8")
        extracted_passport = passport_match.group(1).decode("utf-8")

        generated = self.client.post("/generate-applicant-batch", data={
            "template_id": "bym",
            "manual_full_name": [extracted_name],
            "manual_passport_number": [extracted_passport],
        })
        self.assertEqual(generated.status_code, 302)
        batch_url = generated.headers["Location"]
        batch_id = batch_url.rsplit("/", 1)[-1]
        manifest = updater._load_applicant_batch(batch_id)
        self.assertEqual(manifest["files"][0]["employee_name"], extracted_name)
        self.assertEqual(manifest["files"][0]["passport_number"], extracted_passport)
        filename = manifest["files"][0]["filename"]
        generated_pdf = self.client.get(f"/batches/{batch_id}/preview/{filename}")
        self.assertEqual(generated_pdf.status_code, 200)
        generated_pdf_bytes = generated_pdf.get_data()
        generated_pdf.close()
        pdf = updater.fitz.open(stream=generated_pdf_bytes, filetype="pdf")
        text = "\n".join(page.get_text() for page in pdf)
        self.assertIn(extracted_name, text)
        self.assertIn(extracted_passport, text)
        pdf.close()
        visual = self.client.get(f"/batches/{batch_id}/visual/{filename}")
        self.assertEqual(visual.status_code, 200)
        self.assertIn(b"data:image/png;base64,", visual.data)

    def test_standard_template_has_rasterized_preview_viewer(self):
        preview = self.client.get("/preview-pages/afgo_bym")
        self.assertEqual(preview.status_code, 200)
        self.assertEqual(preview.mimetype, "text/html")
        self.assertIn(b"data:image/png;base64,", preview.data)
        self.assertIn(b"Page 1 of 4", preview.data)
        pdf_preview = self.client.get("/preview/afgo_bym")
        self.assertEqual(pdf_preview.status_code, 200)
        self.assertEqual(pdf_preview.mimetype, "application/pdf")
        pdf_preview.get_data()
        pdf_preview.close()

    def test_ocr_form_has_visible_selection_and_progress_feedback(self):
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn(b'id="passport-ocr-form"', page.data)
        self.assertIn(b'id="ocr-submit-status"', page.data)
        self.assertIn(b"No scan selected. Choose a PDF, image, or HEIC file first.", page.data)
        self.assertIn(b"Reading scans locally", page.data)
        self.assertIn(b'aria-live="polite"', page.data)
        self.assertIn(b"MANUAL APPLICANT ENTRY", page.data)
        self.assertIn(b"MANUAL / CSV", page.data)
        self.assertIn(b"does not upload passport scans or run OCR", page.data)
        self.assertIn(b"OCR INTAKE", page.data)
        self.assertIn(b"PASSPORT SCAN OCR", page.data)
        self.assertIn(b"Extract applicant name and passport number (OCR)", page.data)

    def test_oversized_upload_gets_a_friendly_limit_message(self):
        old_limit = updater.app.config["MAX_CONTENT_LENGTH"]
        updater.app.config["MAX_CONTENT_LENGTH"] = 256
        try:
            response = self.client.post(
                "/extract-passports",
                data=b"x" * 512,
                content_type="application/octet-stream",
                follow_redirects=True,
            )
        finally:
            updater.app.config["MAX_CONTENT_LENGTH"] = old_limit
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"25 MB combined limit", response.data)

    def test_invalid_applicant_is_rejected_and_csv_template_is_available(self):
        response = self.client.post("/generate-applicant-batch", data={
            "template_id": "bym",
            "manual_full_name": "Name Without Passport",
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"enter both the applicant full name and passport number", response.data)

        template = self.client.get("/applicant-csv-template")
        self.assertEqual(template.status_code, 200)
        self.assertIn(b"full_name,passport_number", template.data)
        self.assertEqual(self.client.get("/batches/not-a-batch").status_code, 404)


if __name__ == "__main__":
    unittest.main()
