# Project 4 (Optional) — Image or Text Recognition 👁️

**Goal:** Implement a basic image or text recognition task using available libraries
(pre-trained models — no training from scratch).

This solution takes **Path 1: OCR** with `pytesseract` (Python wrapper for Google's
Tesseract engine). The machine's "optic nerve": an image is not a picture, it's a
3D array of pixel intensities — pre-processing turns that noisy matrix into
something the recognizer can read.

## The pipeline

| Phase | What happens |
|---|---|
| **Input** | Any image path passed as an argument; if none is given, a noisy sample image is generated so the demo is fully self-contained. |
| **Pre-processing** | 1. **Grayscale** — collapse RGB into a 1D intensity matrix. 2. **Gaussian blur** — remove micro-noise. 3. **Otsu thresholding** — force every pixel to black or white for perfect character contrast. |
| **Recognition** | Tesseract with `--psm 6` (single uniform text block — layout configuration is critical for accuracy). |
| **Output** | Extracted text with per-word confidence. The **80% confidence gate** drops low-certainty guesses: `if confidence >= 0.80: keep else: drop`. |

## Milestone validation checklist

- [x] **Library integration** — pytesseract + OpenCV, error-free
- [x] **Pre-processing integrity** — grayscale + thresholding demonstrated (saved to `preprocessed.png`)
- [x] **Accuracy benchmarking** — validated mean confidence reported, 80% minimum gate
- [x] **Visual confirmation** — legible OCR string printed with pass/drop breakdown

## Setup

Requires the Tesseract engine (not just the Python wrapper):

```bash
sudo apt install tesseract-ocr
pip install pytesseract opencv-python
```

## Run it

```bash
python3 ocr.py                  # generates and reads the sample image
python3 ocr.py my_photo.png     # or read your own image
```

Actual output (Tesseract 5.5.0):

```
Input: sample_input.png (640x220 px)
Pre-processed image saved to preprocessed.png (grayscale -> blur -> Otsu)

=== Recognized text (confidence >= 80%) ===
DECODELABS PROJECT 4 OCR PIPELINE ONLINE

Per-word confidence:
  [PASS] DECODELABS      92%
  [PASS] PROJECT         95%
  [PASS] 4               95%
  [PASS] OCR             96%
  [PASS] PIPELINE        96%
  [PASS] ONLINE          96%

Validated mean confidence: 95.0%
```

## Key skills demonstrated

Using pre-trained AI libraries (transfer learning mindset: download the degree,
don't retrain it), image pre-processing, confidence thresholds / softmax outputs,
and interpreting model outputs honestly.
