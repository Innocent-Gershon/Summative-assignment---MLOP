"""
locustfile.py - Flood request simulation for Weather Classifier API
Usage:
    locust -f locustfile.py --host http://localhost:8000
    # Then open http://localhost:8089 to configure users & spawn rate

    # Headless (CLI) example — 100 users, 10/s ramp, 60s duration:
    locust -f locustfile.py --host http://localhost:8000 \
           --headless -u 100 -r 10 --run-time 60s \
           --html locust_report.html
"""

import os
import io
import random
from pathlib import Path
from locust import HttpUser, task, between
from PIL import Image
import numpy as np


def _make_dummy_image_bytes() -> bytes:
    """Generate a random 224x224 RGB image as JPEG bytes."""
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# Pre-generate a small pool of dummy images to avoid per-request overhead
_IMAGE_POOL = [_make_dummy_image_bytes() for _ in range(10)]

# If real test images exist, use them instead
_TEST_DIR = Path(__file__).parent / "data" / "test"
_REAL_IMAGES = []
for cls_dir in _TEST_DIR.iterdir():
    if cls_dir.is_dir():
        _REAL_IMAGES.extend(list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png")))


class WeatherAPIUser(HttpUser):
    """Simulates a user hitting the Weather Classifier API."""
    wait_time = between(0.5, 2)  # seconds between tasks

    @task(5)
    def predict(self):
        """Most frequent task: POST /predict with an image."""
        if _REAL_IMAGES:
            img_path = random.choice(_REAL_IMAGES)
            with open(img_path, "rb") as f:
                img_bytes = f.read()
            filename = img_path.name
        else:
            img_bytes = random.choice(_IMAGE_POOL)
            filename = "test.jpg"

        self.client.post(
            "/predict",
            files={"file": (filename, img_bytes, "image/jpeg")},
            name="/predict",
        )

    @task(2)
    def health_check(self):
        """Lightweight health probe."""
        self.client.get("/health", name="/health")

    @task(1)
    def get_status(self):
        """Status endpoint."""
        self.client.get("/status", name="/status")

    @task(1)
    def class_distribution(self):
        """Visualization endpoint."""
        self.client.get("/visualizations/class-distribution", name="/viz/class-dist")
