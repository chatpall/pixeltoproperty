# PixelToProperty

**Turn a tensile-test chart image into digitized data and mechanical properties.**

PixelToProperty is a free web tool that takes an image of a stress–strain
curve — a screenshot, a scanned figure, or a photo of a printed graph — and
automatically extracts the underlying data points and standard mechanical
properties: Young's modulus (E), 0.2% offset yield strength (Rp0.2),
ultimate tensile strength (Rm), elongation (A), and the Hollomon power-law
hardening fit (K, n).

🔗 **Live app:** *[add your Railway URL here]*

---

## What it does

1. **Detects the chart frame** and calibrates the axes (including OCR of
   the axis tick labels) using OpenCV + Tesseract.
2. **Digitizes the curve** pixel-by-pixel into an ordered (strain, stress)
   point series.
3. **Computes engineering properties** — E, Rp0.2, Rm, A — using a
   physically-anchored elastic-window detector, not a fixed pixel offset.
4. **Fits the Hollomon hardening law** on the true stress–strain curve
   (converted up to the point of maximum load, before necking).
5. **Reports a confidence rating** (HIGH / MEDIUM / LOW) on the elastic
   modulus estimate, so a low-resolution source image is flagged instead of
   silently producing a misleading number.

## Scope: elastic-plastic metallic materials

This tool is intentionally scoped to **monotonic tensile curves of
elastic-plastic metals**, not materials in general. Every computed
quantity — the offset yield method, the Hollomon power law, the
engineering-to-true curve conversion — corresponds to a specific mechanical
model that is only physically valid for that material class (see the
in-app Roadmap/About section for details). It is not intended for
viscoelastic polymers, elastomers, or materials without a clear
elastic-to-plastic transition.

## Tech stack

- **[Streamlit](https://streamlit.io)** — web UI
- **OpenCV** (`opencv-python-headless`) — chart frame detection, curve
  isolation, Hough-transform line detection
- **Tesseract OCR** (via `pytesseract`) — axis tick label reading
- **NumPy / SciPy / scikit-image** — calibration fitting, curve
  skeletonization, property computation
- **Matplotlib** — result plots

## Project structure

```
app.py                     Streamlit UI (pages, layout, plotting)
digitization.py            Frame detection, axis calibration, curve digitization
engineering_properties.py  E / Rp0.2 / Rm / A computation
true_curve.py               True stress-strain conversion + Hollomon fit
requirements.txt            Python dependencies
packages.txt                 System dependency (tesseract-ocr) for Streamlit Cloud
Dockerfile                   Container build (pinned versions - see below)
docker-compose.yml            Local Docker testing
```

## Running locally

### Option A — plain Python
```bash
pip install -r requirements.txt
# Tesseract OCR must also be installed system-wide - see:
# https://tesseract-ocr.github.io/tessdoc/Installation.html
streamlit run app.py
```
Opens at `http://localhost:8501`.

### Option B — Docker
```bash
docker compose up -d --build
```
Also opens at `http://localhost:8501`. This is the recommended way to
develop against, since it uses the exact same pinned Python/OpenCV/
Tesseract versions as production (see *Why Docker* below).

## Deployment (Railway)

This repo is set up to deploy directly on [Railway](https://railway.com):

1. Push this repo to GitHub.
2. On Railway: **New Project → Deploy from GitHub repo**.
3. Railway detects the `Dockerfile` automatically and builds from it.
4. **Settings → Networking → Generate Domain** to get a public URL.

### Why Docker instead of deploying straight from source

Different OpenCV builds can return slightly different data shapes from
`cv2.HoughLinesP`, which broke frame detection when the dev and deployment
environments used different versions. The `Dockerfile` pins exact
versions for Python, OpenCV, and Tesseract, so the container behaves
identically in development and in production.

### Note on the port

Railway assigns a **dynamic port** via the `$PORT` environment variable at
deploy time — it is not always 8501. The `Dockerfile`'s `CMD` uses the
shell form (`streamlit run app.py --server.port=$PORT ...`) specifically so
`$PORT` gets expanded at container start; the exec-array form (`CMD
["streamlit", "run", ...]`) does **not** expand environment variables and
will cause an immediate 502 on Railway.

## Roadmap

Already working:
- OCR-based curve digitization (frame + axis detection, curve extraction)
- Axis calibration from detected tick marks
- Engineering property computation (E, Rp0.2, Rm, elongation) + Hollomon fit

Planned:
- Manual correction of digitized points before computing properties
- Batch processing (multiple images, summary table export)
- PDF report export (charts + values, not just CSV)
- Save/reuse calibration settings across a series of charts
- Compare multiple curves on one chart
- Detection of multiple curves within a single image
- Session history / undo-redo between calibration attempts

## AI-assisted development

This project was built through close, iterative collaboration with
**Claude** (Anthropic) as a pair-programming and documentation assistant —
architecture decisions, debugging (e.g. the frame-detection and Railway
port fixes below), translations, and this documentation were all worked
out together, turn by turn, over an extended back-and-forth.

To be specific about how that worked: every change was proposed by
Claude, then reviewed, tested, and directed by the project author, who
made the actual engineering decisions (what to build, what tradeoffs to
accept, which physical/material-science constraints the tool should
respect) and verified every fix against real chart images before it was
accepted. AI assistance sped up the process considerably; it does not
change who is responsible for the result, or who gets credit for
directing it — this was authored *with* Claude, not *by* it.

If you're curious what that process actually looked like in practice, the
project's issue log (tracking every real bug found during development,
including root causes and fixes) reflects it directly.

## Feedback

This is an active project and feedback from anyone working with
tensile-test data digitization is very welcome.

📧 chatpall+pixeltoproperty@protonmail.com
