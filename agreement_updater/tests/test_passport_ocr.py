import io
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw, ImageFont

from passport_ocr import extract_passport_data, ocr_status, parse_mrz_text


LINE_1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
LINE_2 = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"


class PassportMRZTests(unittest.TestCase):
    def test_td3_mrz_extracts_fields_and_validates_checksums(self):
        result = parse_mrz_text(f"{LINE_1}\n{LINE_2}")
        self.assertEqual(result["employee_name"], "ERIKSSON ANNA MARIA")
        self.assertEqual(result["passport_number"], "L898902C3")
        self.assertTrue(result["passport_checksum_valid"])
        self.assertTrue(result["composite_checksum_valid"])

    def test_unique_ocr_confusion_is_corrected_only_by_checksum(self):
        noisy_line_2 = LINE_2.replace("L898902C3", "L8989O2C3", 1)
        result = parse_mrz_text(f"{LINE_1}\n{noisy_line_2}")
        self.assertEqual(result["passport_number"], "L898902C3")
        self.assertTrue(result["passport_checksum_valid"])
        self.assertTrue(result["ocr_corrected_passport_character"])

    def test_mrz_rows_collapsed_into_one_ocr_line_are_split_and_parsed(self):
        result = parse_mrz_text(LINE_1 + LINE_2)
        self.assertEqual(result["employee_name"], "ERIKSSON ANNA MARIA")
        self.assertEqual(result["passport_number"], "L898902C3")
        self.assertTrue(result["passport_checksum_valid"])

    @unittest.skipUnless(ocr_status()[0], "Tesseract OCR engine is not installed")
    def test_passport_image_upload_extracts_candidates(self):
        image = Image.new("RGB", (1800, 560), "white")
        draw = ImageDraw.Draw(image)
        font_path = Path(__file__).resolve().parents[1] / "assets" / "DejaVuSans.ttf"
        font = ImageFont.truetype(str(font_path), 48)
        draw.text((35, 350), LINE_1, font=font, fill="black")
        draw.text((35, 430), LINE_2, font=font, fill="black")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        result = extract_passport_data(buffer.getvalue(), "synthetic_passport.png")
        self.assertEqual(result["employee_name"], "ERIKSSON ANNA MARIA")
        self.assertEqual(result["passport_number"], "L898902C3")
        self.assertTrue(result["passport_checksum_valid"])

    @unittest.skipUnless(ocr_status()[0], "Tesseract OCR engine is not installed")
    def test_mrz_above_old_fixed_lower_page_crop_is_found(self):
        image = Image.new("RGB", (1800, 1000), "white")
        draw = ImageDraw.Draw(image)
        font_path = Path(__file__).resolve().parents[1] / "assets" / "DejaVuSans.ttf"
        font = ImageFont.truetype(str(font_path), 48)
        draw.text((35, 420), LINE_1, font=font, fill="black")
        draw.text((35, 500), LINE_2, font=font, fill="black")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        result = extract_passport_data(buffer.getvalue(), "passport_with_margins.png")
        self.assertEqual(result["employee_name"], "ERIKSSON ANNA MARIA")
        self.assertEqual(result["passport_number"], "L898902C3")
        self.assertTrue(result["passport_checksum_valid"])

    @unittest.skipUnless(ocr_status()[0], "Tesseract OCR engine is not installed")
    def test_sideways_phone_scan_is_read_when_orientation_detection_is_uncertain(self):
        image = Image.new("RGB", (1800, 560), "white")
        draw = ImageDraw.Draw(image)
        font_path = Path(__file__).resolve().parents[1] / "assets" / "DejaVuSans.ttf"
        font = ImageFont.truetype(str(font_path), 48)
        draw.text((35, 350), LINE_1, font=font, fill="black")
        draw.text((35, 430), LINE_2, font=font, fill="black")
        image = image.rotate(-90, expand=True)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        with patch("passport_ocr.pytesseract.image_to_osd", side_effect=RuntimeError("orientation unclear")):
            result = extract_passport_data(buffer.getvalue(), "sideways_passport.png")
        self.assertEqual(result["employee_name"], "ERIKSSON ANNA MARIA")
        self.assertEqual(result["passport_number"], "L898902C3")
        self.assertTrue(result["passport_checksum_valid"])

    @unittest.skipUnless(ocr_status()[0], "Tesseract OCR engine is not installed")
    def test_heic_photo_upload_is_supported(self):
        try:
            from pillow_heif import from_pillow
        except ImportError:
            self.skipTest("pillow-heif is not installed")

        image = Image.new("RGB", (1800, 560), "white")
        draw = ImageDraw.Draw(image)
        font_path = Path(__file__).resolve().parents[1] / "assets" / "DejaVuSans.ttf"
        font = ImageFont.truetype(str(font_path), 48)
        draw.text((35, 350), LINE_1, font=font, fill="black")
        draw.text((35, 430), LINE_2, font=font, fill="black")
        buffer = io.BytesIO()
        from_pillow(image).save(buffer, quality=100)

        result = extract_passport_data(buffer.getvalue(), "synthetic_passport.heic")
        self.assertEqual(result["employee_name"], "ERIKSSON ANNA MARIA")
        self.assertEqual(result["passport_number"], "L898902C3")
        self.assertTrue(result["passport_checksum_valid"])


if __name__ == "__main__":
    unittest.main()
