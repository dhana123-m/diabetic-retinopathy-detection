"""
Full frontend and backend integration tests for Diabetic Retinopathy Detection System.
"""

import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import create_app


def run_tests():
    app = create_app()
    client = app.test_client()

    passed = 0
    failed = 0
    results = []

    def test(name, condition, detail=""):
        nonlocal passed, failed
        if condition:
            passed += 1
            results.append(f"  PASS  {name}")
            print(f"  PASS  {name}")
        else:
            failed += 1
            results.append(f"  FAIL  {name} - {detail}")
            print(f"  FAIL  {name} - {detail}")

    print("=" * 60)
    print("  BACKEND API TESTS")
    print("=" * 60)

    # --- Test 1: Home page ---
    resp = client.get("/")
    test("GET / returns 200", resp.status_code == 200, f"got {resp.status_code}")
    test("GET / contains title", b"Diabetic Retinopathy Detection" in resp.data)
    test("GET / contains nav", b"nav-brand" in resp.data)
    test("GET / contains stats grid", b"stats-grid" in resp.data)
    test("GET / contains footer", b"footer" in resp.data)
    test("GET / has disclaimer", b"disclaimer" in resp.data.lower())

    # --- Test 2: Dashboard ---
    resp = client.get("/dashboard")
    test("GET /dashboard returns 200", resp.status_code == 200)
    test("GET /dashboard has charts", b"pieChart" in resp.data)
    test("GET /dashboard has risk chart", b"riskChart" in resp.data)
    test("GET /dashboard has timeline", b"timelineChart" in resp.data)

    # --- Test 3: Analysis page ---
    resp = client.get("/analysis")
    test("GET /analysis returns 200", resp.status_code == 200)
    test("GET /analysis has upload area", b"uploadArea" in resp.data)
    test("GET /analysis has file input", b"fileInput" in resp.data)
    test("GET /analysis has analyze button", b"analyzeBtn" in resp.data)
    test("GET /analysis has drag-drop", b"dragover" in resp.data)
    test("GET /analysis has preview section", b"previewSection" in resp.data)
    test("GET /analysis has result section", b"resultSection" in resp.data)
    test("GET /analysis has loading section", b"loadingSection" in resp.data)

    # --- Test 4: History page ---
    resp = client.get("/history")
    test("GET /history returns 200", resp.status_code == 200)
    test("GET /history has title", b"Prediction History" in resp.data)

    # --- Test 5: About page ---
    resp = client.get("/about")
    test("GET /about returns 200", resp.status_code == 200)
    test("GET /about has disclaimer", b"Healthcare Disclaimer" in resp.data)
    test("GET /about explains DR", b"Diabetic Retinopathy" in resp.data)
    test("GET /about explains EfficientNet", b"EfficientNet" in resp.data)
    test("GET /about explains Grad-CAM", b"Grad-CAM" in resp.data)

    # --- Test 6: API Stats ---
    resp = client.get("/api/stats")
    test("GET /api/stats returns 200", resp.status_code == 200)
    data = json.loads(resp.data)
    test("GET /api/stats has total", "total" in data)
    test("GET /api/stats has class_counts", "class_counts" in data)
    test("GET /api/stats has risk_counts", "risk_counts" in data)
    test("GET /api/stats has recent", "recent" in data)

    # --- Test 7: Predict without image ---
    resp = client.post("/predict")
    test("POST /predict no image returns 400", resp.status_code == 400)

    # --- Test 8: Predict with invalid format ---
    data_invalid = {"image": (io.BytesIO(b"fake"), "test.txt")}
    resp = client.post("/predict", data=data_invalid, content_type="multipart/form-data")
    test("POST /predict invalid format returns 400", resp.status_code == 400)

    # --- Test 9: Predict with valid JPEG (demo mode) ---
    from PIL import Image

    img = Image.new("RGB", (224, 224), color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    resp = client.post(
        "/predict",
        data={"image": (buf, "test_image.jpg")},
        content_type="multipart/form-data",
    )
    test("POST /predict valid JPEG returns 200", resp.status_code == 200, f"got {resp.status_code}")
    pred_data = json.loads(resp.data)
    test("POST /predict returns success", pred_data.get("success") is True)
    test("POST /predict has prediction", "prediction" in pred_data)
    test("POST /predict has gradcam", "gradcam" in pred_data)
    test("POST /predict has record_id", "record_id" in pred_data)
    pred = pred_data.get("prediction", {})
    test("POST /predict has class_name", "class_name" in pred)
    test("POST /predict has confidence", "confidence" in pred)
    test("POST /predict has risk_level", "risk_level" in pred)
    test("POST /predict has probabilities", "probabilities" in pred)
    test("POST /predict has all_class_names", "all_class_names" in pred)
    gc = pred_data.get("gradcam", {})
    test("POST /predict gradcam has original", "original" in gc)
    test("POST /predict gradcam has overlay", "overlay" in gc)
    test("POST /predict gradcam has heatmap", "heatmap" in gc)
    test("POST /predict gradcam has comparison", "comparison" in gc)

    # --- Test 10: Predict with valid PNG ---
    img2 = Image.new("RGB", (500, 500), color="blue")
    buf2 = io.BytesIO()
    img2.save(buf2, format="PNG")
    buf2.seek(0)
    resp2 = client.post(
        "/predict",
        data={"image": (buf2, "test_image.png")},
        content_type="multipart/form-data",
    )
    test("POST /predict valid PNG returns 200", resp2.status_code == 200)

    # --- Test 11: Predict with JPEG (different extension) ---
    img3 = Image.new("RGB", (100, 100), color="green")
    buf3 = io.BytesIO()
    img3.save(buf3, format="JPEG")
    buf3.seek(0)
    resp3 = client.post(
        "/predict",
        data={"image": (buf3, "test.jpeg")},
        content_type="multipart/form-data",
    )
    test("POST /predict valid .jpeg returns 200", resp3.status_code == 200)

    # --- Test 12: Result detail page ---
    record_id = pred_data.get("record_id")
    if record_id:
        resp = client.get(f"/result/{record_id}")
        test(f"GET /result/{record_id} returns 200", resp.status_code == 200)
        test("Result page has class name", pred["class_name"].encode() in resp.data)
        test("Result page has disclaimer", b"disclaimer" in resp.data.lower())
    else:
        test("GET /result/<id> skipped (no record_id)", False, "no record_id")

    # --- Test 13: History after prediction ---
    resp = client.get("/history")
    test("GET /history after prediction returns 200", resp.status_code == 200)
    test("History shows prediction record", b"data-table" in resp.data)

    # --- Test 14: Stats after prediction ---
    resp = client.get("/api/stats")
    data = json.loads(resp.data)
    test("Stats total > 0 after prediction", data["total"] > 0, f"total={data['total']}")

    # --- Test 15: Upload served images ---
    gradcam_overlay = gc.get("overlay", "")
    if gradcam_overlay:
        resp = client.get(f"/uploads/{gradcam_overlay}")
        test(f"GET /uploads/{gradcam_overlay} returns 200", resp.status_code == 200)

    gradcam_heatmap = gc.get("heatmap", "")
    if gradcam_heatmap:
        resp = client.get(f"/uploads/{gradcam_heatmap}")
        test(f"GET /uploads/{gradcam_heatmap} returns 200", resp.status_code == 200)

    comparison = gc.get("comparison", "")
    if comparison:
        resp = client.get(f"/uploads/{comparison}")
        test(f"GET /uploads/{comparison} returns 200", resp.status_code == 200)

    # --- Test 16: Clear history ---
    resp = client.post("/clear-history")
    test("POST /clear-history returns 200", resp.status_code == 200)
    ch_data = json.loads(resp.data)
    test("POST /clear-history success", ch_data.get("success") is True)

    # --- Test 17: Stats after clear ---
    resp = client.get("/api/stats")
    data = json.loads(resp.data)
    test("Stats total = 0 after clear", data["total"] == 0, f"total={data['total']}")

    # --- Test 18: History empty after clear ---
    resp = client.get("/history")
    test("GET /history empty state after clear", b"No Predictions Yet" in resp.data)

    # --- Test 19: 404 handler ---
    resp = client.get("/nonexistent-page")
    test("GET /nonexistent-page returns 404", resp.status_code == 404)

    # --- Test 20: Multiple predictions for history ---
    for i in range(3):
        img_t = Image.new("RGB", (224, 224), color=(i * 80, 50, 100))
        buf_t = io.BytesIO()
        img_t.save(buf_t, format="JPEG")
        buf_t.seek(0)
        client.post(
            "/predict",
            data={"image": (buf_t, f"multi_{i}.jpg")},
            content_type="multipart/form-data",
        )
    resp = client.get("/api/stats")
    data = json.loads(resp.data)
    test("Multiple predictions counted correctly", data["total"] == 3, f"total={data['total']}")

    # --- Test 21: Dashboard populated ---
    resp = client.get("/dashboard")
    test("Dashboard renders with data", resp.status_code == 200)
    test("Dashboard has class counts", b"classCounts" in resp.data)
    test("Dashboard has Chart.js", b"chart.js" in resp.data.lower() or b"Chart" in resp.data)

    # --- Test 22: CSS loads ---
    resp = client.get("/static/css/style.css")
    test("GET /static/css/style.css returns 200", resp.status_code == 200)
    test("CSS contains root variables", b":root" in resp.data)

    # --- Test 23: JS loads ---
    resp = client.get("/static/js/app.js")
    test("GET /static/js/app.js returns 200", resp.status_code == 200)
    test("JS has showNotification function", b"showNotification" in resp.data)

    # --- Test 24: Demo banner hidden when DEMO_MODE=False ---
    resp = client.get("/")
    from config import DEMO_MODE as _dm
    if _dm:
        test("Demo banner visible when DEMO_MODE=True", b"DEMO MODE" in resp.data)
    else:
        test("Demo banner absent when DEMO_MODE=False", b"DEMO MODE" not in resp.data)

    # --- Test 25: Result page for each prediction ---
    resp = client.get("/api/stats")
    data = json.loads(resp.data)
    for rec in data.get("recent", [])[:3]:
        resp = client.get(f"/result/{rec['id']}")
        test(f"GET /result/{rec['id']} renders", resp.status_code == 200)

    # --- Summary ---
    print()
    print("=" * 60)
    total = passed + failed
    print(f"  RESULTS: {passed}/{total} passed, {failed} failed")
    print("=" * 60)
    if failed > 0:
        print("\n  Failed tests:")
        for r in results:
            if "FAIL" in r:
                print(f"    {r}")

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
