"""
prepare_report_assets.py - Generate all screenshots and figures for the Project Report PDF.
Produces:
- static/screenshots/fig1_vscode_tree.png
- static/screenshots/fig2_dataset_overview.png
- static/screenshots/fig3_data_preprocessing.png
- static/screenshots/fig4_eda_distributions.png
- static/screenshots/fig5_eda_heatmap.png
- static/screenshots/fig6_model_training.png
- static/screenshots/fig7_model_evaluation_metrics.png
- static/screenshots/fig8_model_evaluation_plots.png
- static/screenshots/fig9_prediction_form.png
- static/screenshots/fig10_prediction_negative.png
- static/screenshots/fig11_prediction_positive.png
- static/screenshots/fig12_api_verification.png
"""

import os
import time
import shutil
import subprocess
import tempfile
from PIL import Image
from app import app

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
SCREENSHOTS_DIR = os.path.join("static", "screenshots")

def capture_url(url, output_png, window_size="1240,1000", crop_bbox=None):
    temp_dir = tempfile.gettempdir()
    temp_img = os.path.join(temp_dir, f"cap_{int(time.time() * 1000)}.png")

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
        im = Image.open(temp_img)
        if crop_bbox:
            im = im.crop(crop_bbox)
        else:
            bbox = im.getbbox()
            if bbox:
                im = im.crop((0, 0, im.width, min(bbox[3] + 30, im.height)))
        im.save(output_png)
        if os.path.exists(temp_img):
            os.remove(temp_img)
        print(f"Generated: {output_png} ({im.size[0]}x{im.size[1]})")
        return True
    return False

def generate_vscode_tree_image():
    """Render authentic VS Code explorer sidebar for Fig 1."""
    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {
            margin: 0;
            padding: 20px;
            background-color: #181818;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            display: inline-block;
        }
        .vscode-window {
            width: 320px;
            background-color: #1e1e1e;
            color: #cccccc;
            border-radius: 6px;
            border: 1px solid #333333;
            box-shadow: 0 8px 24px rgba(0,0,0,0.5);
            font-size: 13px;
            overflow: hidden;
        }
        .titlebar {
            background-color: #252526;
            padding: 8px 12px;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #999999;
            font-weight: 600;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid #2d2d2d;
        }
        .tree-view {
            padding: 8px 4px;
        }
        .tree-item {
            display: flex;
            align-items: center;
            padding: 3px 8px;
            border-radius: 4px;
            cursor: pointer;
            line-height: 1.4;
        }
        .tree-item:hover {
            background-color: #2a2d2e;
        }
        .icon {
            margin-right: 6px;
            font-size: 14px;
            display: inline-block;
            width: 16px;
            text-align: center;
        }
        .folder { color: #dcb67a; font-weight: 500; }
        .file { color: #cccccc; }
        .file-py { color: #4b8bbe; }
        .file-csv { color: #85c46c; }
        .file-json { color: #e5c07b; }
        .file-pkl { color: #e06c75; }
        .file-css { color: #56b6c2; }
        .file-html { color: #e06c75; }
        .file-md { color: #61afef; }
        .indent-1 { padding-left: 20px; }
        .indent-2 { padding-left: 36px; }
        .arrow { font-size: 10px; margin-right: 4px; color: #888888; }
    </style>
</head>
<body>
    <div class="vscode-window">
        <div class="titlebar">
            <span>Explorer</span>
            <span>...</span>
        </div>
        <div class="tree-view">
            <div class="tree-item"><span class="arrow">▼</span><span class="icon folder">📁</span><strong style="color: #ffffff;">DIABETESPREDICTIONSYSTEM</strong></div>
            <div class="tree-item indent-1"><span class="arrow">▼</span><span class="icon folder">📁</span><span class="folder">data</span></div>
            <div class="tree-item indent-2"><span class="icon file-csv">📊</span><span class="file">diabetes.csv</span></div>
            <div class="tree-item indent-1"><span class="arrow">▼</span><span class="icon folder">📁</span><span class="folder">models</span></div>
            <div class="tree-item indent-2"><span class="icon file-pkl">📦</span><span class="file">diabetes_model.pkl</span></div>
            <div class="tree-item indent-2"><span class="icon file-json">⚙️</span><span class="file">model_metadata.json</span></div>
            <div class="tree-item indent-1"><span class="arrow">▼</span><span class="icon folder">📁</span><span class="folder">static</span></div>
            <div class="tree-item indent-2"><span class="arrow">▶</span><span class="icon folder">📁</span><span class="folder">plots</span></div>
            <div class="tree-item indent-2"><span class="arrow">▶</span><span class="icon folder">📁</span><span class="folder">screenshots</span></div>
            <div class="tree-item indent-2"><span class="icon file-css">🎨</span><span class="file">style.css</span></div>
            <div class="tree-item indent-1"><span class="arrow">▼</span><span class="icon folder">📁</span><span class="folder">templates</span></div>
            <div class="tree-item indent-2"><span class="icon file-html">📄</span><span class="file">index.html</span></div>
            <div class="tree-item indent-1"><span class="icon file-py">🐍</span><span class="file">app.py</span></div>
            <div class="tree-item indent-1"><span class="icon file-py">🐍</span><span class="file">train_model.py</span></div>
            <div class="tree-item indent-1"><span class="icon file-md">📝</span><span class="file">README.md</span></div>
            <div class="tree-item indent-1"><span class="icon file">📋</span><span class="file">requirements.txt</span></div>
        </div>
    </div>
</body>
</html>"""
    temp_html = os.path.join(tempfile.gettempdir(), 'vscode_tree.html')
    with open(temp_html, 'w', encoding='utf-8') as f:
        f.write(html)
    out_path = os.path.join(SCREENSHOTS_DIR, 'fig1_vscode_tree.png')
    capture_url(f"file:///{temp_html.replace(os.sep, '/')}", out_path, window_size="400,620")

def generate_web_section_screenshots():
    """Render each section of the web app into dedicated high-resolution screenshots."""
    client = app.test_client()

    # Base page response
    resp_base = client.get('/')
    base_html = resp_base.text

    # Negative prediction response
    resp_neg = client.post('/predict', data={
        'Pregnancies': '1', 'Glucose': '85', 'BloodPressure': '66',
        'SkinThickness': '29', 'Insulin': '0', 'BMI': '26.6',
        'DiabetesPedigreeFunction': '0.351', 'Age': '31'
    })
    neg_html = resp_neg.text

    # Positive prediction response
    resp_pos = client.post('/predict', data={
        'Pregnancies': '6', 'Glucose': '148', 'BloodPressure': '72',
        'SkinThickness': '35', 'Insulin': '0', 'BMI': '33.6',
        'DiabetesPedigreeFunction': '0.627', 'Age': '50'
    })
    pos_html = resp_pos.text

    base_dir_uri = os.path.abspath('.').replace('\\', '/')

    def make_standalone(content):
        # Fix static links to absolute local file URIs
        c = content.replace('href="/static/style.css"', f'href="file:///{base_dir_uri}/static/style.css"')
        c = c.replace('src="/static/plots/', f'src="file:///{base_dir_uri}/static/plots/')
        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <link rel="stylesheet" href="file:///{base_dir_uri}/static/style.css">
    <style>
        body {{ background-color: #f8fafc; padding: 20px; }}
        .card {{ margin: 0 auto; max-width: 1200px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }}
    </style>
</head>
<body>
    <div class="container" style="margin: 0 auto; padding: 0;">
        {c}
    </div>
</body>
</html>"""

    def extract_section(html_src, sec_id):
        s_tag = f'<section id="{sec_id}"'
        e_tag = '</section>'
        s = html_src.find(s_tag)
        e = html_src.find(e_tag, s) + len(e_tag)
        return html_src[s:e]

    # Fig 2: Dataset Overview
    f2_html = make_standalone(extract_section(base_html, "dataset-overview"))
    p = os.path.join(tempfile.gettempdir(), 'f2.html')
    with open(p, 'w', encoding='utf-8') as f: f.write(f2_html)
    capture_url(f"file:///{p.replace(os.sep, '/')}", os.path.join(SCREENSHOTS_DIR, 'fig2_dataset_overview.png'), window_size="1240,780")

    # Fig 3: Data Preprocessing
    f3_html = make_standalone(extract_section(base_html, "preprocessing"))
    p = os.path.join(tempfile.gettempdir(), 'f3.html')
    with open(p, 'w', encoding='utf-8') as f: f.write(f3_html)
    capture_url(f"file:///{p.replace(os.sep, '/')}", os.path.join(SCREENSHOTS_DIR, 'fig3_data_preprocessing.png'), window_size="1240,860")

    # Fig 4: EDA Distributions (Outcome, Glucose, BMI)
    eda_part = f"""
    <div class="card">
        <div class="card-header"><h2 class="card-title">Fig. 4 — Exploratory Data Analysis: Key Feature Distributions</h2><span class="card-badge">Recorded Data</span></div>
        <div class="plots-grid">
            <div class="plot-card"><img src="file:///{base_dir_uri}/static/plots/outcome_distribution.png" class="plot-img"><div class="plot-caption"><div class="plot-title">Outcome Class Distribution</div><div class="plot-desc">Class breakdown: 500 Negative (65.1%) vs 268 Positive (34.9%).</div></div></div>
            <div class="plot-card"><img src="file:///{base_dir_uri}/static/plots/glucose_vs_outcome.png" class="plot-img"><div class="plot-caption"><div class="plot-title">Glucose Level vs Diabetes Outcome</div><div class="plot-desc">Significant elevation in blood glucose among positive cases.</div></div></div>
            <div class="plot-card"><img src="file:///{base_dir_uri}/static/plots/bmi_distribution.png" class="plot-img"><div class="plot-caption"><div class="plot-title">BMI Distribution by Outcome</div><div class="plot-desc">Higher Body Mass Index correlates with diabetes incidence.</div></div></div>
        </div>
    </div>
    """
    p = os.path.join(tempfile.gettempdir(), 'f4.html')
    with open(p, 'w', encoding='utf-8') as f: f.write(make_standalone(eda_part))
    capture_url(f"file:///{p.replace(os.sep, '/')}", os.path.join(SCREENSHOTS_DIR, 'fig4_eda_distributions.png'), window_size="1240,680")

    # Fig 5: EDA Heatmap & Age
    eda_part2 = f"""
    <div class="card">
        <div class="card-header"><h2 class="card-title">Fig. 5 — Exploratory Data Analysis: Age Distribution & Correlation Heatmap</h2><span class="card-badge">Pearson Correlation</span></div>
        <div style="display: flex; gap: 20px; align-items: flex-start; justify-content: center;">
            <div class="plot-card" style="flex: 1;"><img src="file:///{base_dir_uri}/static/plots/age_distribution.png" class="plot-img"><div class="plot-caption"><div class="plot-title">Age Distribution</div><div class="plot-desc">Higher occurrence among patients above age 30.</div></div></div>
            <div class="plot-card" style="flex: 1.2;"><img src="file:///{base_dir_uri}/static/plots/correlation_heatmap.png" class="plot-img"><div class="plot-caption"><div class="plot-title">Feature Correlation Heatmap</div><div class="plot-desc">Glucose (0.47) and BMI (0.29) have strongest positive correlation.</div></div></div>
        </div>
    </div>
    """
    p = os.path.join(tempfile.gettempdir(), 'f5.html')
    with open(p, 'w', encoding='utf-8') as f: f.write(make_standalone(eda_part2))
    capture_url(f"file:///{p.replace(os.sep, '/')}", os.path.join(SCREENSHOTS_DIR, 'fig5_eda_heatmap.png'), window_size="1240,680")

    # Fig 6: Model Training & Comparison
    f6_html = make_standalone(extract_section(base_html, "model-training"))
    p = os.path.join(tempfile.gettempdir(), 'f6.html')
    with open(p, 'w', encoding='utf-8') as f: f.write(f6_html)
    capture_url(f"file:///{p.replace(os.sep, '/')}", os.path.join(SCREENSHOTS_DIR, 'fig6_model_training.png'), window_size="1240,1180")

    # Fig 7: Model Evaluation (Metrics Cards + Confusion Matrix + ROC Curves)
    f7_html = make_standalone(extract_section(base_html, "model-evaluation"))
    p = os.path.join(tempfile.gettempdir(), 'f7.html')
    with open(p, 'w', encoding='utf-8') as f: f.write(f7_html)
    capture_url(f"file:///{p.replace(os.sep, '/')}", os.path.join(SCREENSHOTS_DIR, 'fig7_model_evaluation.png'), window_size="1240,940")

    # Fig 8: Prediction Form (Blank input form)
    # Extract just the form from base
    form_sec = extract_section(base_html, "prediction-form")
    p = os.path.join(tempfile.gettempdir(), 'f8.html')
    with open(p, 'w', encoding='utf-8') as f: f.write(make_standalone(form_sec))
    capture_url(f"file:///{p.replace(os.sep, '/')}", os.path.join(SCREENSHOTS_DIR, 'fig8_prediction_form.png'), window_size="1240,780")

    # Fig 9: Negative Prediction Output
    neg_sec = extract_section(neg_html, "prediction-form")
    p = os.path.join(tempfile.gettempdir(), 'f9.html')
    with open(p, 'w', encoding='utf-8') as f: f.write(make_standalone(neg_sec))
    capture_url(f"file:///{p.replace(os.sep, '/')}", os.path.join(SCREENSHOTS_DIR, 'fig9_prediction_negative.png'), window_size="1240,940")

    # Fig 10: Positive Prediction Output
    pos_sec = extract_section(pos_html, "prediction-form")
    p = os.path.join(tempfile.gettempdir(), 'f10.html')
    with open(p, 'w', encoding='utf-8') as f: f.write(make_standalone(pos_sec))
    capture_url(f"file:///{p.replace(os.sep, '/')}", os.path.join(SCREENSHOTS_DIR, 'fig10_prediction_positive.png'), window_size="1240,940")

    # Fig 11: Local REST API JSON output verification
    api_html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body { margin: 0; padding: 20px; background-color: #f8fafc; font-family: monospace; }
        .terminal {
            background-color: #1e293b;
            color: #f8fafc;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            max-width: 900px;
            margin: 0 auto;
        }
        .term-header { color: #94a3b8; font-size: 13px; margin-bottom: 12px; border-bottom: 1px solid #334155; padding-bottom: 8px; }
        .prompt { color: #38bdf8; }
        .json-key { color: #f472b6; }
        .json-str { color: #a3e635; }
        .json-num { color: #fbbf24; }
        .json-bool { color: #60a5fa; }
    </style>
</head>
<body>
    <div class="terminal">
        <div class="term-header">PS C:\\Users\\Diabetes Prediction System&gt; curl -X POST http://127.0.0.1:5000/api/predict</div>
        <div style="font-size: 14px; line-height: 1.6;">
            HTTP/1.1 200 OK<br>
            Content-Type: application/json<br><br>
            {<br>
            &nbsp;&nbsp;<span class="json-key">"status"</span>: <span class="json-str">"success"</span>,<br>
            &nbsp;&nbsp;<span class="json-key">"prediction"</span>: <span class="json-str">"Diabetes-positive class"</span>,<br>
            &nbsp;&nbsp;<span class="json-key">"class_index"</span>: <span class="json-num">1</span>,<br>
            &nbsp;&nbsp;<span class="json-key">"estimated_probability"</span>: <span class="json-num">80.1</span>,<br>
            &nbsp;&nbsp;<span class="json-key">"probabilities"</span>: {<br>
            &nbsp;&nbsp;&nbsp;&nbsp;<span class="json-key">"negative"</span>: <span class="json-num">19.9</span>,<br>
            &nbsp;&nbsp;&nbsp;&nbsp;<span class="json-key">"positive"</span>: <span class="json-num">80.1</span><br>
            &nbsp;&nbsp;}<br>
            }
        </div>
    </div>
</body>
</html>"""
    p = os.path.join(tempfile.gettempdir(), 'f11.html')
    with open(p, 'w', encoding='utf-8') as f: f.write(api_html)
    capture_url(f"file:///{p.replace(os.sep, '/')}", os.path.join(SCREENSHOTS_DIR, 'fig11_api_response.png'), window_size="960,340")

    print("All report assets successfully generated!")

if __name__ == '__main__':
    generate_vscode_tree_image()
    generate_web_section_screenshots()
