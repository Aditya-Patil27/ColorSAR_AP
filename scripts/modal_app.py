"""
ColorSAR dashboard (app.py) hosted on Modal - public URL, CPU only.

    python -m modal deploy scripts/modal_app.py

Needs model1_run/model_1.pth, model3_run/model_3.pth and demo_samples/v_2 (8 test-scene
SAR/optical pairs) on the machine that deploys. One container stays warm so the link
answers at once; set min_containers=0 after the evaluation to stop the (small) cost.
"""
import os

import modal

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("libgl1", "libglib2.0-0")
         .pip_install("torch==2.9.0", "torchvision==0.24.0", index_url="https://download.pytorch.org/whl/cpu")
         .pip_install("gradio==6.31.0", "fastapi", "numpy<2.3", "scikit-image==0.26.0",
                      "opencv-python-headless", "matplotlib==3.10.1", "pillow")
         .env({"COLORSAR_NO_BROWSER": "1"}))
for f in ("app.py", "colorize.py", "color_utils.py", "model_architecture.py", "pipeline.py", "sample_input.jpeg"):
    image = image.add_local_file(os.path.join(REPO, f), "/app/" + f)
for d in ("model1_run", "model3_run"):
    image = image.add_local_dir(os.path.join(REPO, d), "/app/" + d, ignore=["logs", "logs/**", "*.log"])
image = image.add_local_dir(os.path.join(REPO, "demo_samples", "v_2"), "/app/v_2")

app = modal.App("colorsar-demo", image=image)


@app.function(cpu=2, memory=4096, min_containers=1, max_containers=1, timeout=600)
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def ui():
    import sys

    os.chdir("/app")
    sys.path.insert(0, "/app")
    import gradio as gr
    from fastapi import FastAPI

    import app as colorsar   # /app/app.py: loads both models, builds the Gradio Blocks

    return gr.mount_gradio_app(FastAPI(), colorsar.demo, path="/")
