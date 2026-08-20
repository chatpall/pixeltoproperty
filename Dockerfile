# syntax=docker/dockerfile:1

# ============================================================================
# PixelToProperty — Dockerfile
#
# Zabaľuje appku (app.py + digitization.py + engineering_properties.py +
# true_curve.py) presne so systémovými aj Python závislosťami, ktoré
# vyžaduje requirements.txt a packages.txt. Fixáciou base image a verzií
# balíkov toto natrvalo rieši problém 7.1 zo ZAZNAM_PROBLEMOV.md
# (rozdielna verzia OpenCV medzi vývojovým a produkčným prostredím).
# ============================================================================

FROM python:3.11-slim

# ----------------------------------------------------------------------------
# Systémové závislosti
#   - tesseract-ocr: z packages.txt, potrebné pre pytesseract
#   - libgl1, libglib2.0-0, libsm6, libxext6, libxrender1: bežné runtime
#     závislosti pre opencv-python-headless (aj headless verzia OpenCV má
#     interné odkazy na tieto zdieľané knižnice pri importe cv2)
#   - libgomp1: potrebné pre scikit-image / scipy (OpenMP runtime)
# ----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ----------------------------------------------------------------------------
# Python závislosti — kopírované a inštalované ako samostatná vrstva,
# aby sa pri zmene len app.py/*.py nemuseli preinštalovať všetky balíky
# (Docker layer caching).
# ----------------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ----------------------------------------------------------------------------
# Zdrojový kód appky
# ----------------------------------------------------------------------------
COPY app.py digitization.py engineering_properties.py true_curve.py ./

# ----------------------------------------------------------------------------
# Streamlit nastavenia pre produkčnú prevádzku v kontajneri
#   - headless: bez pokusu otvoriť prehliadač vo vnútri kontajnera
#   - address 0.0.0.0: počúvať na všetkých rozhraniach (nutné, inak appka
#     nebude dostupná zvonka kontajnera)
#   - port 8501: štandardný Streamlit port, mapovaný von cez EXPOSE/-p
# ----------------------------------------------------------------------------
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

# Jednoduchý healthcheck — Streamlit má vstavaný _stcore/health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app.py"]
