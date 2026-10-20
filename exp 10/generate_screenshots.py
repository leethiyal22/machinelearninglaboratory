"""
generate_screenshots.py - Automated script to capture required project screenshots.
Captures:
- Fig. 1 — Dataset Overview
- Fig. 2 — Data Preprocessing
- Fig. 3 — Exploratory Data Analysis
- Fig. 4 — Model Training & Comparison
- Fig. 5 — Model Evaluation
- Fig. 6 — Diabetes Prediction (with real inference results & disclaimer)
"""

import os
import time
import shutil
import threading
import subprocess
import tempfile
from PIL import Image
from app import app

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
SCREENSHOTS_DIR = os.path.join("static", "screenshots")

def run_flask_server():
    app.run(host='127.0.0.1', port=5005, debug=False, use_reloader=False)

def capture_url(url, output_png, window_size="1300,3200"):
    temp_dir = tempfile.gettempdir()
    temp_img = os.path.join(temp_dir, f"capture_{int(time.time() * 1000)}.png")

    cmd = [
        CHROME_PATH,
        "--headless=new",
        "--no-sandbox",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=2000",
        f"--window-size={window_size}",
        f"--screenshot={temp_img}",
        url
    ]

    subprocess.run(cmd, check=True, capture_output=True)
    if os.path.exists(temp_img):
        shutil.copy2(temp_img, output_png)
        os.remove(temp_img)
        print(f"Captured: {output_png}")
        return True
    return False

def generate_all_figures():
    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

    # 1. Start Flask in background thread
    print("Starting Flask server for screenshot capture on port 5005...")
    server_thread = threading.Thread(target=run_flask_server, daemon=True)
    server_thread.start()
    time.sleep(2.5)  # Wait for Flask to initialize

    # Test client to generate a rendered HTML file for the prediction state
    print("Generating prediction state HTML...")
    client = app.test_client()
    resp_predict = client.post('/predict', data={
        'Pregnancies': '6',
        'Glucose': '148',
        'BloodPressure': '72',
        'SkinThickness': '35',
        'Insulin': '0',
        'BMI': '33.6',
        'DiabetesPedigreeFunction': '0.627',
        'Age': '50'
    })

    temp_predict_html = os.path.join(tempfile.gettempdir(), 'predicted_page.html')
    # Replace relative static urls with local server urls
    html_content = resp_predict.text.replace('href="/static/style.css"', 'href="http://127.0.0.1:5005/static/style.css"')
    html_content = html_content.replace('src="/static/plots/', 'src="http://127.0.0.1:5005/static/plots/')
    with open(temp_predict_html, 'w', encoding='utf-8') as f:
        f.write(html_content)

    # 2. Capture full dashboard page
    base_full_png = os.path.join(tempfile.gettempdir(), 'base_dashboard.png')
    capture_url("http://127.0.0.1:5005/", base_full_png, window_size="1300,4800")

    # 3. Capture predicted state page
    predicted_full_png = os.path.join(tempfile.gettempdir(), 'predicted_dashboard.png')
    capture_url(f"file:///{temp_predict_html.replace(os.sep, '/')}", predicted_full_png, window_size="1300,5200")

    # 4. Now let's crop each figure cleanly using PIL
    print("Cropping specific figures...")
    base_img = Image.open(base_full_png)
    pred_img = Image.open(predicted_full_png)

    # We can also capture focused views by rendering individual components or cropping from coordinates:
    # Let's inspect the dimensions
    w, h = base_img.size
    print(f"Base image size: {w}x{h}")
    pw, ph = pred_img.size
    print(f"Pred image size: {pw}x{ph}")

    # Let's save the high-resolution figures:
    # Fig 1: Dataset Overview (approx y: 150 to 800)
    # Fig 2: Preprocessing (approx y: 800 to 1600)
    # Fig 3: EDA (approx y: 1600 to 2900)
    # Fig 4: Model Training (approx y: 2900 to 3900)
    # Fig 5: Model Evaluation (approx y: 3900 to 4800)
    # Fig 6: Prediction Form & Result (from predicted page)
    # Better yet, let's create custom HTML pages for each figure or crop precise bounding boxes:
    # Actually, rendering standalone figure HTML cards produces 100% pixel-perfect, crisp, beautifully framed screenshots!

    fig_names = [
        ("fig1_dataset_overview.png", "dataset-overview"),
        ("fig2_data_preprocessing.png", "preprocessing"),
        ("fig3_eda_visualizations.png", "eda"),
        ("fig4_model_training.png", "model-training"),
        ("fig5_model_evaluation.png", "model-evaluation"),
        ("fig6_diabetes_prediction.png", "prediction-form")
    ]

    for fname, sec_id in fig_names:
        # Generate targeted HTML containing just this section + header/disclaimer styling
        target_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <link rel="stylesheet" href="http://127.0.0.1:5005/static/style.css">
    <style>
        body {{ background-color: #f8fafc; padding: 25px; }}
        .card {{ margin: 0 auto; max-width: 1200px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }}
    </style>
</head>
<body>
    <div class="container" style="margin: 0 auto; padding: 0;">
"""
        source_html = html_content if sec_id == "prediction-form" else resp_predict.text
        # Extract section
        start_tag = f'<section id="{sec_id}"'
        end_tag = '</section>'
        s_idx = source_html.find(start_tag)
        e_idx = source_html.find(end_tag, s_idx) + len(end_tag)

        if s_idx != -1 and e_idx != -1:
            extracted_section = source_html[s_idx:e_idx]
            # Replace paths with http
            extracted_section = extracted_section.replace('src="/static/plots/', 'src="http://127.0.0.1:5005/static/plots/')
            target_html += extracted_section + "\n</div>\n</body>\n</html>"

            temp_fhtml = os.path.join(tempfile.gettempdir(), f"target_{sec_id}.html")
            with open(temp_fhtml, 'w', encoding='utf-8') as f:
                f.write(target_html)

            out_dest = os.path.join(SCREENSHOTS_DIR, fname)
            # Determine appropriate height
            height = 1400 if sec_id in ["eda", "model-training", "prediction-form"] else 1000
            capture_url(f"file:///{temp_fhtml.replace(os.sep, '/')}", out_dest, window_size=f"1240,{height}")
            # Auto-trim transparent or extra white padding
            try:
                im = Image.open(out_dest)
                # Crop to content height if needed
                bbox = im.getbbox()
                if bbox:
                    # Keep a slight margin
                    cropped = im.crop((0, 0, im.width, min(bbox[3] + 40, im.height)))
                    cropped.save(out_dest)
            except Exception as e:
                print(f"Trim notice: {e}")

    print("All 6 required figure screenshots generated successfully!")

if __name__ == '__main__':
    generate_all_figures()
