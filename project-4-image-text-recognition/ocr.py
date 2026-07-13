"""
Project 4 (Optional) — Image or Text Recognition, Path 1: OCR (DecodeLabs)

Recognition pipeline using a pre-trained engine (Tesseract via pytesseract):
  INPUT   -> image (a sample is generated if none is provided)
  PROCESS -> pre-processing: grayscale -> Gaussian blur -> Otsu thresholding
  OUTPUT  -> extracted text + per-word confidence, filtered at the 80% gate

Usage:
  python3 ocr.py [path/to/image.png]
"""

import sys

import cv2
import numpy as np
import pytesseract

CONFIDENCE_GATE = 80  # Project 4 minimum standard


def make_sample_image(path: str = "sample_input.png") -> str:
    """Generate a noisy sample image so the pipeline can be demoed end to end."""
    img = np.full((220, 640, 3), 235, dtype=np.uint8)
    cv2.putText(img, "DECODELABS PROJECT 4", (40, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (40, 40, 40), 3)
    cv2.putText(img, "OCR PIPELINE ONLINE", (40, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (60, 60, 60), 3)
    noise = np.random.default_rng(42).integers(-25, 25, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(path, img)
    return path


def preprocess(image: np.ndarray) -> np.ndarray:
    """The logic skeleton: systematic image pre-processing."""
    # Step 1: grayscale — collapse the 3D RGB matrix into a 1D intensity matrix.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Step 2: Gaussian blur — smooth out micro-imperfections and artifact noise.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Step 3: thresholding (Otsu) — force every pixel to choose a side:
    # pure black-and-white gives perfect contrast for character contours.
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def main() -> None:
    image_path = sys.argv[1] if len(sys.argv) > 1 else make_sample_image()
    image = cv2.imread(image_path)
    if image is None:
        sys.exit(f"Could not read image: {image_path}")
    print(f"Input: {image_path} ({image.shape[1]}x{image.shape[0]} px)")

    binary = preprocess(image)
    cv2.imwrite("preprocessed.png", binary)
    print("Pre-processed image saved to preprocessed.png (grayscale -> blur -> Otsu)")

    # PSM 6: single uniform block of text. Layout configuration is critical for accuracy.
    config = "--psm 6"
    data = pytesseract.image_to_data(binary, config=config,
                                     output_type=pytesseract.Output.DICT)

    # The 80% gate: without a confidence filter, an AI treats every guess with
    # equal certainty — confident hallucinations and false positives.
    accepted, rejected = [], []
    for word, conf in zip(data["text"], data["conf"]):
        word = word.strip()
        conf = int(conf)
        if not word or conf < 0:
            continue
        (accepted if conf >= CONFIDENCE_GATE else rejected).append((word, conf))

    print(f"\n=== Recognized text (confidence >= {CONFIDENCE_GATE}%) ===")
    print(" ".join(w for w, _ in accepted) or "(nothing passed the gate)")

    print("\nPer-word confidence:")
    for word, conf in accepted:
        print(f"  [PASS] {word:<15} {conf}%")
    for word, conf in rejected:
        print(f"  [DROP] {word:<15} {conf}%  (below the {CONFIDENCE_GATE}% gate)")

    if accepted:
        avg = sum(c for _, c in accepted) / len(accepted)
        print(f"\nValidated mean confidence: {avg:.1f}%")


if __name__ == "__main__":
    main()
