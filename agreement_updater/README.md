# Employment Agreement Updater

A local Flask editor for the five uploaded Romanian individual employment agreements. It keeps the original PDFs unchanged, stores editable values in `data/state.json`, and regenerates updated PDF copies under `generated/` from the originals on every save.

## Start it

Python 3.10+ is recommended. Passport-scan OCR also requires the Tesseract system engine and English trained data. On Debian/Ubuntu install it with `sudo apt-get install tesseract-ocr`; on Windows/macOS install Tesseract and ensure the `tesseract` executable is on `PATH`.

```bash
python3 -m pip install -r requirements.txt
python3 app.py
```

Open `http://127.0.0.1:5000` on the computer running the app. In the Arena workspace the app is started on `0.0.0.0:5000` for the live preview.

To check all five templates without starting the web server:

```bash
python3 app.py --check
python3 -m unittest discover -s tests -v
```

## Use it

- Select an agreement tab and edit the employee, employer, job, salary, meal allowance, work-permit, dates, hours, leave, trial, and notice values.
- Click **Save changes & regenerate PDFs**. Every mapped occurrence in each agreement is updated together, and all five output PDFs are rebuilt from the untouched source files.
- To copy one chosen value to every applicable agreement, use **Apply one value to all agreements**. This deliberately overwrites document-specific values for that field.
- Use **View pages** for a rasterized, in-browser preview of every page; **Download PDF** or **Download all updated PDFs (.zip)** retrieves the five updated template outputs.
- **Reset to source values** restores the initial field values and regenerates clean outputs.

## Generate agreements for applicants

- In **Create agreements for applicants**, add one applicant manually or add multiple rows. You can also upload a CSV, or combine manual rows and CSV rows in one batch.
- The CSV header must contain `full_name,passport_number`. The editor also accepts common alternatives such as `employee_name,passport_no`. Use **Download CSV template** for an example; uploads are UTF-8 or Windows-1252 CSV files, up to 50 applicants per batch.
- Choose one agreement template for the batch. The template's saved employer, job, pay, and other agreement terms are retained; the applicant full name and passport number are replaced for each generated PDF.
- Click **Generate PDFs & review**. The system checks each generated PDF's page count, readable text, and applicant name/passport number, then opens a review page with a visual page-image preview and a separate PDF download for every applicant.
- Or use **Extract applicant name and passport number** to upload one passport scan per applicant (PDF, JPG, PNG, TIFF, BMP, WEBP, or HEIC/HEIF; up to 20 files, 10 MB each, 25 MB combined per upload). Tesseract runs locally; OCR checks multiple lower-page crops and image orientations, prioritizes the MRZ and validates passport-number check digits, with visible-text OCR as a fallback. The review form is editable and must be checked against the actual passport before generation. OCR is not infallible.
- Uploaded scans are processed temporarily and are not saved to applicant-batch storage. The resulting agreements and batch manifest are saved under `generated/applicant_batches/`; the manifest contains applicant names and passport numbers.
- After reviewing, download a PDF from its card or use **Download all as ZIP**. Recent batches remain available from the editor for later preview/download.

## Folder layout

- `source_pdfs/` — unchanged copies of the five uploaded PDFs.
- `generated/` — updated PDFs, the all-files ZIP, and saved applicant-batch PDFs, manifests, and ZIPs under `applicant_batches/`.
- `data/documents.json` — field schema and source-location map.
- `data/state.json` — current editable values; reset from the UI if needed.
- `assets/` — bundled DejaVu Sans fonts used for replacement text.

## Review notes

- The AFGO document contains actual yellow highlights; those are redrawn behind the corresponding changed fields. The other four source PDFs use bold or regular black text for their changeable values, not yellow highlighting.
- The ASIA and DNB source PDFs show `ATRACTIV ASIGURARI SRL` in their employer signature blocks, which does not match the employer name in Clause A. The app flags this and leaves the signature name unlinked by default. Check **Link employer signature to Clause A** to use the Clause A employer name there.
- The DNB source has no net salary or COR code; no net-salary field is inserted. A COR code may be added in the editor if needed, and the job line will include it.
- This is a document-editing aid, not legal advice or a legal-validation service. Generated PDFs are drafts. Changing a signed or stamped original may affect its legal validity; review and obtain appropriate approval before use.
