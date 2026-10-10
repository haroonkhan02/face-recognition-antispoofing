"""Gradio demo: compare two faces, and score a photo for presentation attacks.

Example:
    faceauth-demo            # http://127.0.0.1:7860
"""

from __future__ import annotations

import argparse

import cv2

from .align import align_face
from .detect import FaceDetector
from .embed import get_embedder
from .gallery import VERIFY_THRESHOLD


def build_app():
    import gradio as gr

    from .app import SpoofScorer

    detector, embedder = FaceDetector(), get_embedder("arcface")
    try:
        spoof = SpoofScorer()
    except FileNotFoundError:
        spoof = None

    def aligned(rgb):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        face = detector.largest(bgr)
        if face is None:
            raise gr.Error("No face found.")
        return bgr, face, align_face(bgr, face.landmarks)

    def compare(a, b):
        if a is None or b is None:
            return None, None, ""
        _, _, ca = aligned(a)
        _, _, cb = aligned(b)
        sim = float((embedder([ca]) * embedder([cb])).sum())
        verdict = "same person" if sim >= VERIFY_THRESHOLD else "different people"
        return (cv2.cvtColor(ca, cv2.COLOR_BGR2RGB), cv2.cvtColor(cb, cv2.COLOR_BGR2RGB),
                f"cosine similarity {sim:.3f} -> {verdict} (threshold {VERIFY_THRESHOLD})")

    def liveness(img):
        if img is None or spoof is None:
            return "anti-spoofing model not trained" if spoof is None else ""
        bgr, face, _ = aligned(img)
        p = spoof(bgr, face)
        verdict = "ATTACK" if p >= spoof.threshold else "live"
        return f"P(spoof) = {p:.3f} -> {verdict} (threshold {spoof.threshold:.3f}, set on val)"

    with gr.Blocks(title="Face verification & anti-spoofing") as app:
        gr.Markdown("# Face verification · anti-spoofing\nUse photos of people who consent. "
                    "Nothing is stored.")
        with gr.Tab("Same person?"):
            with gr.Row():
                a, b = gr.Image(label="photo A", type="numpy"), gr.Image(label="photo B",
                                                                         type="numpy")
            with gr.Row():
                ca, cb = gr.Image(label="aligned A"), gr.Image(label="aligned B")
            out = gr.Textbox(label="result")
            gr.Button("Compare").click(compare, [a, b], [ca, cb, out])
        with gr.Tab("Real or spoof?"):
            img = gr.Image(label="photo (whole scene, not just the face)", type="numpy")
            res = gr.Textbox(label="result")
            gr.Button("Check").click(liveness, img, res)
    return app


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args(argv)
    build_app().launch(server_port=args.port)


if __name__ == "__main__":
    main()

