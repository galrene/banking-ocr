# Banking OCR

Small script to extract category spending from **Raiffeisen graphical overview** screenshots and write results to a CSV.

## Files
- [ocr.py](ocr.py) — main script
- [src_img/](src_img/) — directory for input screenshots.

## Requirements
- Python 3.8+
- tesseract OCR binary installed and on PATH
- Python packages: pytesseract, pillow, opencv-python, pandas

Install Python deps:
```sh
pip install pytesseract pillow opencv-python pandas
```

## Usage
0. Take screenshots and transfer to PC 
    - transfer has to preserve original file format otherwise ocr preprocessing will fail (ex. e-mail)
1. Clone the repo
2. Run the script from the project root:
```sh
python3 ocr.py
```
This:
- reads PNG files from [src_img/](src_img/)
- preprocesses them
- parses detected lines with [`parse_ocr_text`](ocr.py)
- writes results to [ocr.csv](ocr.csv) and copies to clipboard (Excel format)

## Customization
- Adjust crop coordinates in [`preprocess`](ocr.py) for different screenshot layouts. Currently configured for iPhone 13
- Change input extension in [`ocr_images`](ocr.py) if images are not `.PNG`.
- Update `rename_map` in [`main`](ocr.py) to remap categories before saving.
- Update `custom_orcer` in [`main`](ocr.py) to specify custom output ordering.

## Notes
- Ensure the Tesseract executable is installed (e.g., `sudo apt install tesseract-ocr`).

## Known issues
1. Doesn't handle multi-line categories at this time.