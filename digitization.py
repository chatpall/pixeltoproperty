"""
MODUL 1: DIGITALIZACIA KRIVKY
=============================
Zodpovednost tohto modulu: obrazok grafu -> (strain, stress) numpy polia.
Nic viac - ziadne mechanicke vlastnosti, ziadny prevod na true krivku.
To je zamerne, aby sa dal tento modul pouzit/testovat/menit nezavisle od
zvysku (napr. vymenit OCR engine, zlepsit izolaciu farby a pod. bez rizika
rozbitia vypoctov mechanickych vlastnosti v inych moduloch).

Verejne (pre ostatne moduly a GUI) urcene funkcie a triedy:
    Frame, AxisCalibration   - datove triedy
    detect_frame(gray)       -> Frame
    calibrate_axes(gray, frame) -> AxisCalibration
    digitize_curve(crop_bgr, calib, crop_offset, upscale_factor=4)
                              -> (strain: np.ndarray, stress: np.ndarray, color: str)

POUZITE KNIZNICE (licencie vhodne pre komercne pouzitie):
    opencv-python (Apache-2.0), numpy (BSD-3), scikit-image (BSD-3),
    networkx (BSD-3), pytesseract + Tesseract OCR (Apache-2.0)
    ALEBO (ak Tesseract nie je dostupny - napr. bez admin prav na instalaciu
    systemoveho binarky) easyocr (Apache-2.0) - CISTO PYTHONOVA alternativa,
    instaluje sa len cez `pip install easyocr` (ziadny systemovy binarny
    subor, teda funguje aj bez admin prav - pip pise len do uzivatelovho
    Python prostredia). Nevyhoda: tazsia zavislost (PyTorch), pomalsie na
    prvom spusteni (stahuje modely).
"""

import re
from dataclasses import dataclass, field

import cv2
import numpy as np
import networkx as nx
from skimage.morphology import skeletonize

# OCR BACKEND: skusime najprv Tesseract (rychlejsi, lahsi), ak nie je
# dostupny (napr. ziadne admin prava na instalaciu systemoveho binarky),
# padneme na EasyOCR (cisto pip-installable, ziadny system binary).
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

EASYOCR_AVAILABLE = False
_easyocr_reader = None
if not TESSERACT_AVAILABLE:
    try:
        import easyocr
        EASYOCR_AVAILABLE = True
    except ImportError:
        EASYOCR_AVAILABLE = False


def _get_easyocr_reader():
    """Lazy-loaded EasyOCR reader (nacitanie modelu je pomale, ~niekolko
    sekund az desiatky sekund pri prvom spusteni - preto sa vytvori len raz
    a znovu sa pouzije pri dalsich volaniach v ramci behu appky)."""
    global _easyocr_reader
    if _easyocr_reader is None:
        _easyocr_reader = easyocr.Reader(["en"], gpu=False)
    return _easyocr_reader



# ----------------------------------------------------------------------------
# 1a) Detekcia ramu grafu
# ----------------------------------------------------------------------------

@dataclass
class Frame:
    x_left: int
    x_right: int
    y_top: int
    y_bottom: int
    x_right_source: str = "unknown"   # "detected_line" | "width_fallback" - diagnostika/metadata


def detect_frame(gray: np.ndarray) -> Frame:
    """Najde hranice grafovej plochy pomocou Houghovej transformacie na dlhe rovne
    ciary (predpoklad: osi/ram su najdlhsie rovne ciary v obrazku).

    Pravy okraj (x_right) sa uraci ADAPTIVNE podla toho, kolko dokazu mame:
    - Ak Hough najde ASPON 2 zreteľne odlisene vertikalne ciary (t.j. graf ma
      kompletny ram vratane praveho okraja, typicke pre grafy s mriezkou), pravy
      okraj = najpravejsia z nich (max(v_lines)). Toto ma OPORU V DATACH.
    - Ak najde len 1 (typicke pre Excel-style grafy, kde je nakreslena len os
      Y+X bez praveho/horneho ramika), padneme na fallback 'sirka obrazka - 5px'
      (predpoklad, ze graf zabera takmer cely obrazok - platilo pre nase prve
      testovacie obrazky, ktore boli vopred orezane tesne okolo grafu).

    Zdroj rozhodnutia sa zaznamena do Frame.x_right_source - uzitocne ako
    diagnosticke/databazove metadata (pozri diskusiu o "triedach dokumentov
    zdola nahor" namiesto vopred pevne danej klasifikacie).

    minLineLength sa POSTUPNE ZNIZUJE (0.5 -> 0.35 -> 0.2 -> 0.1 z min(h,w)),
    ak vyssi prah nenajde ciary v OBOCH smeroch. DOLEZITE PRE PRIPADY, KED graf
    zabera len CAST celkovej vysky/sirky obrazka (napr. nad grafom je tabulka
    so specifikaciami vzorky, ktora zvacsi celkovu vysku obrazka bez toho, aby
    patrila ku grafu) - realne overene: graf vysoky ~320px v obrazku vysokom
    766px (kvoli tabulke nad nim) nemal ZIADNU vertikalnu ciaru dost dlhu na
    povodny pevny prah 0.5*766=383px, hoci horizontalne (cez celu sirku) ciary
    nasiel v hojnom pocte - jednostranne zlyhanie prahu, nie chyba obrazka."""
    h, w = gray.shape
    edges = cv2.Canny(gray, 50, 150)

    h_lines, v_lines = [], []
    for length_frac in (0.5, 0.35, 0.2, 0.1):
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=80,
            minLineLength=int(length_frac * min(h, w)), maxLineGap=10,
        )
        cur_h, cur_v = [], []
        if lines is not None:
            for l in lines:
                x1, y1, x2, y2 = l[0]
                if abs(y1 - y2) < 5:
                    cur_h.append((y1 + y2) // 2)
                elif abs(x1 - x2) < 5:
                    cur_v.append((x1 + x2) // 2)
        # DOLEZITA OPRAVA (realny nalez): povodne sa h_lines aj v_lines PREPISOVALI
        # pri kazdej iteracii, aj ked uz h_lines boli v poriadku pri prisnejsom
        # (strict = spolahlivejsom) prahu - loop pokracoval len kvoli chybajucim
        # v_lines a nechtiac tym POKAZIL uz funkcne h_lines (na jednom obrazku
        # bez ram-ovej ohranicky Houghovo v_lines nikdy nenaslo 2 ciary pri
        # ziadnom prahu, takze sa doslo az k najvolnejsiemu prahu 0.1, kde uz
        # h_lines zachytilo aj maly text POD grafom ako 'ciaru', posunulo to
        # y_bottom o desiatky pixelov). Opravene: h_lines/v_lines sa 'zamknu'
        # hned ako sa raz najdu (na najprísnejsom moznom prahu), dalsie
        # uvolnovanie prahu uz do nich nezasahuje.
        if not h_lines and cur_h:
            h_lines = cur_h
        if not v_lines and cur_v:
            v_lines = cur_v
        if h_lines and v_lines:
            break

    if not h_lines:
        raise RuntimeError(
            "Nepodarilo sa detegovat ram grafu (osi). Skus iny obrazok "
            "s vyraznejsimi/kontrastnejsimi osami."
        )

    # h_lines moze obsahovat aj OSAMELE ciary MIMO samotneho grafu (napr. deliaca
    # ciara pod hlavickou tabulky so specifikaciami vzorky nad grafom - realne
    # overene v praxi, sposobilo to nespravny odhad y_top_est). Skutocna mriezka
    # grafu je vzdy HUSTY ZHLUK mnohych blizkych ciar, zatial co osamela ciara
    # inde v dokumente je izolovana (velka medzera k najblizsej dalsej ciare).
    # Zoberieme preto NAJVACSI (podla poctu ciar) zhluk, nie min/max zo VSETKYCH.
    sorted_h = sorted(set(h_lines))
    clusters = [[sorted_h[0]]]
    for v in sorted_h[1:]:
        if v - clusters[-1][-1] <= 60:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    largest_cluster = max(clusters, key=len)
    h_lines = largest_cluster

    if len(v_lines) < 2:
        # FALLBACK: HoughLinesP niekedy nevie poskladat vertikalne ciary do
        # dlhych segmentov, ked su prerusovane castymi krizeniami s hustou
        # mriezkou (kazde krizenie vytvori 'roh', ktory rozbije liniu na
        # kratke fragmenty). Priamy stlpcovy profil hustoty hran je na toto
        # odolnejsi - realne overene na obrazku, kde stlpec so ~97% hranovych
        # pixelov v ramci vysky grafu Houghovi unikol uplne (0 v_lines), hoci
        # bol vizualne aj numericky jednoznacne pritomny.
        #
        # DOLEZITE (dalsi realny nalez): povodne sa fallback spustal LEN pri
        # NULA v_lines ("if not v_lines"). Na inom obrazku Hough nasiel PRESNE
        # JEDNU v_lines (napr. mimo skutocneho grafu, v oblasti loga) - to
        # nesplnalo "prazdny zoznam", takze fallback sa nespustil, a chybny
        # jediny bod sa pouzil ako oba okraje naraz (x_left aj x_right cez
        # width-5 fallback), co dalo uplne nespravny ram. Opravene na "< 2",
        # kedze na urcenie OBOCH okrajov (lavy+pravy) potrebujeme aspon 2
        # nezavisle najdene ciary.
        y_top_est, y_bottom_est = min(h_lines), max(h_lines)
        col_density = edges[y_top_est:y_bottom_est, :].sum(axis=0) / 255.0
        row_count = max(y_bottom_est - y_top_est, 1)
        candidate_cols = np.where(col_density > 0.85 * row_count)[0]
        if len(candidate_cols) == 0:
            raise RuntimeError(
                "Nepodarilo sa detegovat vertikalne hranice ramu grafu (osi) - "
                "ani Houghovou transformaciou, ani stlpcovym profilom hustoty hran. "
                "Skus iny obrazok s vyraznejsimi/kontrastnejsimi osami."
            )
        # zoskupime bilzke stlpce (do 3px) do jednej hranice
        v_lines = []
        for c in candidate_cols:
            if not v_lines or c - v_lines[-1] > 3:
                v_lines.append(int(c))
            else:
                v_lines[-1] = int(c)  # posun na posledny v skupine

    x_left = min(v_lines)
    # Zoskupime bilzke v_lines (do 5px) - anti-aliasing casto vytvori 2 tesne
    # susedne detekcie tej istej fyzickej ciary, co by inak falosne vyzeralo
    # ako "2 rozne ciary" aj ked je v obrazku realne len jedna (os Y).
    distinct_v = []
    for v in sorted(set(v_lines)):
        if not distinct_v or v - distinct_v[-1] > 5:
            distinct_v.append(v)

    if len(distinct_v) >= 2 and max(distinct_v) > x_left + 0.3 * w:
        x_right = max(distinct_v)
        x_right_source = "detected_line"
    else:
        x_right = w - 5
        x_right_source = "width_fallback"

    return Frame(
        x_left=x_left, x_right=x_right,
        y_top=min(h_lines), y_bottom=max(h_lines),
        x_right_source=x_right_source,
    )


# ----------------------------------------------------------------------------
# 1b) OCR kalibracia osi (pixel <-> fyzikalna hodnota)
# ----------------------------------------------------------------------------

@dataclass
class AxisCalibration:
    x_slope: float
    x_intercept: float
    y_slope: float
    y_intercept: float
    x_ticks_raw: list = field(default_factory=list)   # vsetci OCR kandidati [(px, val), ...]
    y_ticks_raw: list = field(default_factory=list)
    x_inliers: np.ndarray = None                        # ktore z x_ticks_raw pouzite vo fite
    y_inliers: np.ndarray = None
    x_is_percent_hint: bool = None   # True = OCR nazvu osi nasiel '%' (spolahlivy priamy dokaz).
                                       # None = nenasiel/nejednoznacne - volajuci nech pouzije
                                       # fallback heuristiku podla velkosti cisel.

    def px_to_x(self, px_original):
        """Prevod pixel (X, povodny obrazok) -> fyzikalna hodnota (strain)."""
        return self.x_slope * px_original + self.x_intercept

    def px_to_y(self, py_original):
        """Prevod pixel (Y, povodny obrazok) -> fyzikalna hodnota (stress)."""
        return self.y_slope * py_original + self.y_intercept


NUMBER_RE = re.compile(r"^-?\d+\.?\d*$")


def _ocr_axis_ticks(strip: np.ndarray, horizontal: bool, scale: int = 4):
    """OCR na pruhu obrazka (popisky osi). Kluceve pre spolahlivost:
    - Otsu threshold (nie pevna hodnota) - prisposobi sa kontrastu obrazka
    - whitelist len na cislice/bodku/minus - znizi OCR halucinacie
    - volajuci musi vopred preskocit ~3px od samotnej osovej ciary (tie miesaju
      OCR-u znaky, napr. "0" sa cita ako "r@)")

    Pouzije Tesseract, ak je dostupny (rychlejsi). Ak nie (napr. bez admin
    prav na instalaciu systemoveho binarky), automaticky padne na EasyOCR
    (cisto pip-installable alternativa - viz poznamka na zaciatku modulu)."""
    strip_big = cv2.resize(strip, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    if TESSERACT_AVAILABLE:
        _, strip_bin = cv2.threshold(strip_big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        data = pytesseract.image_to_data(
            strip_bin, config="--psm 6 -c tessedit_char_whitelist=0123456789.-",
            output_type=pytesseract.Output.DICT,
        )
        results = []
        for i in range(len(data["text"])):
            txt = data["text"][i].strip()
            if not NUMBER_RE.match(txt):
                continue
            value = float(txt)
            if horizontal:
                center = (data["left"][i] + data["width"][i] / 2) / scale
            else:
                center = (data["top"][i] + data["height"][i] / 2) / scale
            results.append((center, value))
        return results

    if EASYOCR_AVAILABLE:
        reader = _get_easyocr_reader()
        strip_rgb = cv2.cvtColor(strip_big, cv2.COLOR_GRAY2RGB) if strip_big.ndim == 2 else strip_big
        ocr_results = reader.readtext(strip_rgb, allowlist="0123456789.-", detail=1)
        results = []
        for bbox, txt, conf in ocr_results:
            txt = txt.strip()
            if not NUMBER_RE.match(txt):
                continue
            value = float(txt)
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            if horizontal:
                center = (min(xs) + max(xs)) / 2.0 / scale
            else:
                center = (min(ys) + max(ys)) / 2.0 / scale
            results.append((center, value))
        return results

    return []  # ziadny OCR backend nie je dostupny


def _robust_linear_fit(px, val, n_iters=500, threshold_frac=0.02, seed=0):
    """RANSAC-styl fit: OCR obcas precita aj nezmyselne hodnoty. Namiesto naivneho
    np.polyfit cez VSETKY body najprv najdeme najvacsiu skupinu bodov, ktora spolu
    lezi na jednej priamke ('inliers'), a fitujeme len cez nu."""
    px = np.asarray(px, dtype=float)
    val = np.asarray(val, dtype=float)
    n = len(px)
    if n < 2:
        raise RuntimeError("Potrebne aspon 2 body pre linearny fit.")
    if n == 2:
        slope = (val[1] - val[0]) / (px[1] - px[0])
        intercept = val[0] - slope * px[0]
        return slope, intercept, np.array([True, True])

    rng = np.random.default_rng(seed)
    scale = max(np.ptp(val), 1e-6)
    best_inliers, best_score = None, -1
    for _ in range(n_iters):
        i, j = rng.choice(n, 2, replace=False)
        if px[i] == px[j]:
            continue
        slope = (val[j] - val[i]) / (px[j] - px[i])
        intercept = val[i] - slope * px[i]
        resid = np.abs(slope * px + intercept - val)
        inliers = resid < threshold_frac * scale
        score = int(inliers.sum())
        if score > best_score:
            best_score, best_inliers = score, inliers

    if best_inliers is None or best_inliers.sum() < 2:
        slope, intercept = np.polyfit(px, val, 1)
        return slope, intercept, np.ones(n, dtype=bool)

    slope, intercept = np.polyfit(px[best_inliers], val[best_inliers], 1)
    return slope, intercept, best_inliers


def _find_first_text_band(strip_gray: np.ndarray, axis="rows", max_search: int = None,
                           ink_threshold: int = 200, min_ink_fraction: float = None,
                           relative_frac: float = 0.15, gap_to_stop: int = 8):
    """Najde prvy suvisly 'pas' textu (riadok cisel pod osou X, alebo stlpec cisel
    vlavo od osi Y) priamo z profilu tmavosti pixelov - NEZAVISLE od rozlisenia
    obrazka (na rozdiel od pevnych/skalovanych pixelovych konstant, ktore zlyhali
    na velkych/vysoko-DPI renderoch - viz sprievodna dokumentacia).

    axis="rows": hlada pas POSTUPNE SMEROM DOLE (pre X-os popisky pod ramom).
    axis="cols": hlada pas POSTUPNE SMEROM VLAVO OD KONCA (pre Y-os popisky
                 vlavo od ramu - strip_gray uz ma byt v poradi 'najblizsie k ramu
                 na konci pola', t.j. prehodeny/flip tak aby sme prehladavali
                 od ramu smerom von).

    Prah na 'co uz je text' je ADAPTIVNY relativne k vlastnemu profilu obrazka
    (relative_frac * max(profile)), nie pevne cislo - riadkovy (X-os cisla) a
    stlpcovy (Y-os cisla) profil maju prirodzene INU hustotu atramentu (siršie
    vodorovne ťahy cislic vs. uzsie zvislé), takze jedno pevne cislo pre oboje
    bud podstrelilo Y-os, alebo prestrelilo na X-osi sum tesne pod ciarou
    (overene chybou v praxi - viz sprievodna dokumentacia).

    Vrati (start_idx, end_idx) ohranicujuce najdeny pas (s malou rezervou), alebo
    None ak sa ziadny pas nenasiel."""
    if axis == "rows":
        profile = np.mean(strip_gray < ink_threshold, axis=1)  # podiel tmavych pixelov na riadok
    else:
        profile = np.mean(strip_gray < ink_threshold, axis=0)

    if min_ink_fraction is None:
        max_profile = float(np.max(profile)) if len(profile) else 0.0
        min_ink_fraction = max(0.01, relative_frac * max_profile) if max_profile > 0 else 0.01

    n = len(profile)
    if max_search is not None:
        n = min(n, max_search)

    start = None
    gap_count = 0
    end = None
    for i in range(n):
        has_ink = profile[i] > min_ink_fraction
        if start is None:
            if has_ink:
                start = i
        else:
            if has_ink:
                gap_count = 0
            else:
                gap_count += 1
                if gap_count >= gap_to_stop:
                    end = i - gap_count + 1
                    break
    if start is None:
        return None
    if end is None:
        end = n
    pad = 2
    return max(0, start - pad), min(len(profile), end + pad)


def calibrate_axes(gray: np.ndarray, frame: Frame) -> AxisCalibration:
    """OCR-based kalibracia oboch osi. Vyzaduje aspon 2 citatelne tick hodnoty
    na kazdej osi (odporucane min. 3-4 pre spolahlivy fit).

    Oblast s popiskami sa najde ADAPTIVNE cez profil tmavosti pixelov
    (_find_first_text_band), NIE cez pevne/skalovane pixelove konstanty - tie
    zlyhavaju na velkych/vysoko-DPI obrazkoch (napr. cela strana PDF pri 300 DPI),
    kde su popisky vyrazne vacsie v absolutnych pixeloch nez v malych testovacich
    obrazkoch, na ktorych boli povodne pevne hodnoty odladene."""
    h, w = gray.shape

    # X-os: hladame prvy pas textu POD ramom (mimo samotnej osovej ciary, preto
    # zacina az kus pod frame.y_bottom, aby sme nechytili tick znacky/ciaru)
    search_zone_x = gray[frame.y_bottom + 2:min(frame.y_bottom + 2 + max(200, (frame.y_bottom - frame.y_top) // 3), h), :]
    band = _find_first_text_band(search_zone_x, axis="rows")
    if band is not None:
        x_strip = search_zone_x[band[0]:band[1], :]
    else:
        x_strip = search_zone_x[:max(19, h // 100), :]  # fallback - povodna hruba heuristika

    # Priamy dokaz jednotky osi X ('%' v nazve osi, napr. 'Strain (%)') - hladame
    # DALSI pas textu HNED POD tick cislami (nazov osi je vzdy tesne pod nimi).
    # Spolahlivejsie nez odhad z velkosti cisel (ten zlyhal na materiali s velmi
    # malou taznostou, kde bol cely rozsah osi < 1, aj ked bola v %).
    x_is_percent_hint = None
    if band is not None:
        below_ticks = search_zone_x[band[1]:, :]
        title_band = _find_first_text_band(below_ticks, axis="rows", max_search=below_ticks.shape[0])
        if title_band is not None and (TESSERACT_AVAILABLE or EASYOCR_AVAILABLE):
            title_strip = below_ticks[title_band[0]:title_band[1], :]
            title_big = cv2.resize(title_strip, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            title_text = ""
            try:
                if TESSERACT_AVAILABLE:
                    _, title_bin = cv2.threshold(title_big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    title_text = pytesseract.image_to_string(title_bin, config="--psm 7")
                else:
                    reader = _get_easyocr_reader()
                    title_rgb = cv2.cvtColor(title_big, cv2.COLOR_GRAY2RGB)
                    title_text = " ".join(reader.readtext(title_rgb, detail=0))
            except Exception:
                title_text = ""
            if "%" in title_text:
                x_is_percent_hint = True

    # Y-os: hladame prvy pas textu VLAVO OD ramu. Prehladavame SPRAVA DOLAVA
    # (od osi smerom von), preto strip najprv vyrezeme a OTOCIME.
    search_width = min(frame.x_left, max(200, (frame.y_bottom - frame.y_top) // 3))
    search_zone_y = gray[:, max(0, frame.x_left - search_width):max(0, frame.x_left - 2)]
    search_zone_y_flipped = search_zone_y[:, ::-1]  # teraz index 0 = najblizsie k ramu
    band_y = _find_first_text_band(search_zone_y_flipped, axis="cols")
    if band_y is not None:
        # band_y je vo flipnutych suradniciach, prepocitame spat
        w_zone = search_zone_y.shape[1]
        col_start = w_zone - band_y[1]
        col_end = w_zone - band_y[0]
        y_strip = search_zone_y[:, max(0, col_start):max(0, col_end)]
    else:
        y_strip = search_zone_y  # fallback - cela hladana zona

    x_ticks = _ocr_axis_ticks(x_strip, horizontal=True)
    y_ticks = _ocr_axis_ticks(y_strip, horizontal=False)

    if len(x_ticks) < 2 or len(y_ticks) < 2:
        raise RuntimeError(
            f"OCR naslo len {len(x_ticks)} citatelnych cisel na osi X a "
            f"{len(y_ticks)} na osi Y (potrebne aspon 2 na kazdej). "
            "Skus obrazok s ostrejsimi/vacsimi popiskami."
        )

    x_px = np.array([t[0] for t in x_ticks])
    x_val = np.array([t[1] for t in x_ticks])
    y_px = np.array([t[0] for t in y_ticks])
    y_val = np.array([t[1] for t in y_ticks])

    x_slope, x_intercept, x_inliers = _robust_linear_fit(x_px, x_val)
    y_slope, y_intercept, y_inliers = _robust_linear_fit(y_px, y_val)

    return AxisCalibration(
        x_slope=x_slope, x_intercept=x_intercept,
        y_slope=y_slope, y_intercept=y_intercept,
        x_ticks_raw=x_ticks, y_ticks_raw=y_ticks,
        x_inliers=x_inliers, y_inliers=y_inliers,
        x_is_percent_hint=x_is_percent_hint,
    )


# ----------------------------------------------------------------------------
# 1c) Detekcia farby a izolacia krivky
# ----------------------------------------------------------------------------

def detect_curve_color(hsv_crop: np.ndarray) -> str:
    """Odhadne farbu krivky (cervena/modra/cierna) z dominantnej sytosti a odtienu."""
    s = hsv_crop[:, :, 1]
    sat_mask = s > 60
    if sat_mask.sum() < 50:
        return "black"
    h_channel = hsv_crop[:, :, 0]
    median_hue = np.median(h_channel[sat_mask])
    if median_hue < 15 or median_hue > 165:
        return "red"
    elif 90 <= median_hue <= 140:
        return "blue"
    return "black"


def isolate_curve_mask(crop_bgr: np.ndarray, color: str) -> np.ndarray:
    """Vrati binarnu masku krivky (uint8 0/255), uz spojenu do jedneho suvisleho
    objektu. POZOR: vracia DILATOVANU masku (nie tenku povodnu) - digitalizacia
    cez skeleton potrebuje suvislu masku, inak sa skeleton rozpadne na fragmenty
    pri mikro-medzerach vznikutych na ostrych zakrutach krivky.

    POZOR: tato funkcia predpoklada SPOJITU ciaru. Pre bodove (marker-based)
    krivky pouzi namiesto nej isolate_curve_points_mask + extract_point_centroids
    (viz nizsie) - dilatacia navrhnuta pre medzery na zakrutach ciary NESTACI na
    premostenie medzier medzi riedkymi bodovymi znackami."""
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)

    if color in ("red", "blue"):
        mask = (hsv[:, :, 1] > 60).astype(np.uint8) * 255
    else:
        # Cierna krivka na sivom pozadi s mriezkou - narocnejsi pripad (menej
        # otestovane na realnych obrazkoch nez farebna vetva).
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        dark_mask = (gray < 90).astype(np.uint8) * 255
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                                 minLineLength=int(0.4 * crop_bgr.shape[1]), maxLineGap=5)
        grid_mask = np.zeros_like(dark_mask)
        if lines is not None:
            for l in lines:
                x1, y1, x2, y2 = l[0]
                cv2.line(grid_mask, (x1, y1), (x2, y2), 255, thickness=2)
        mask = cv2.bitwise_and(dark_mask, cv2.bitwise_not(grid_mask))

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    # DOLEZITE: RECT kernel, nie ELLIPSE - elipsovity kernel nema plny dosah v rohoch
    # a nespoji fragmenty krivky na ostrych zakrutach (overene chybou v praxi).
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask_dilated = cv2.dilate(mask_clean, kernel_dilate)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_dilated, connectivity=8)
    if n_labels <= 1:
        raise RuntimeError("Po izolacii nezostali ziadne pixely krivky - skontroluj obrazok.")
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))

    return np.where(labels == largest_label, 255, 0).astype(np.uint8)


# ----------------------------------------------------------------------------
# 1c-bis) Bodova (marker-based) krivka - alternativna vetva k spojitej ciare
# ----------------------------------------------------------------------------
#
# Niektore reporty (napr. ISO 6892-1 video-extenzometer vystupy) nevykresluju
# spojitu ciaru, ale husty oblak samostatnych bodovych markerov. Skeleton+graf
# pristup je pre toto navrhnuty zle (je urceny na 1-pixel-hrubu SPOJITU ciaru).
# Kedze markery su farebne (na rozdiel od ciernej/sivej mriezky a textu), je
# tu VYRAZNE jednoduchsie a spolahlivejsie: najst tazisko kazdeho markera
# priamo (ziadna dilatacia/skeleton potrebna) a zoradit podla X.

def detect_curve_style(crop_bgr: np.ndarray, color: str,
                        max_component_fraction_threshold: float = 0.3,
                        min_components_for_dots: int = 15) -> str:
    """Rozhodne 'line' vs 'dots' PODLA DOKAZU v obrazku (nie vopred danej triedy
    dokumentu) - spocita rozlozenie velkosti spojitych komponent PRED akoukolvek
    dilataciou:
    - Spojita ciara: po prahovani je typicky JEDNA (alebo par) velka komponenta,
      ktora tvori vacsinu celkovej plochy.
    - Bodova krivka: desiatky-stovky MALYCH komponent podobnej velkosti, ziadna
      z nich netvori vyznamny podiel celku.

    Realne namerane na testovacom obrazku: 150 komponent, najvacsia = 3% celku
    -> jednoznacne 'dots'. Kontinualna ciara z predoslych testov: 1 dominantna
    komponenta po dilatacii (~100% podielu pred dilatacio tiez typicky > 30%,
    kedze aj tenka ciara tvori jeden dlhy suvisly blob)."""
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    if color in ("red", "blue"):
        mask = (hsv[:, :, 1] > 60).astype(np.uint8) * 255
    else:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        mask = (gray < 90).astype(np.uint8) * 255

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return "line"  # prazdna maska - necha padnut na povodnu vetvu, tam sa vypise zmysluplna chyba

    areas = stats[1:, cv2.CC_STAT_AREA]
    total_area = areas.sum()
    largest_fraction = areas.max() / total_area if total_area > 0 else 1.0
    n_components = len(areas)

    if n_components >= min_components_for_dots and largest_fraction < max_component_fraction_threshold:
        return "dots"
    return "line"


def isolate_curve_points_mask(crop_bgr: np.ndarray, color: str) -> np.ndarray:
    """Analogia isolate_curve_mask, ale BEZ dilatacie a BEZ vyberu 'najvacsej
    komponenty' - pre bodove krivky chceme VSETKY markery, nie jeden zhluk."""
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    if color in ("red", "blue"):
        mask = (hsv[:, :, 1] > 60).astype(np.uint8) * 255
    else:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        mask = (gray < 90).astype(np.uint8) * 255

    # Lahke ciscenie (odstrani 1-2px osamely sum), ZIADNA velka dilatacia -
    # to by pri hustom oblaku bodov mohlo susedne markery falosne spojit
    # do jednej komponenty a skreslit tazisko.
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)


def extract_point_centroids(mask_points: np.ndarray, min_marker_area: int = 3):
    """Najde tazisko kazdeho samostatneho markera (connected component) a
    zoradi ich podla X - pre tahove skusky je strain monotonne rastuci s casom
    merania, takze zoradenie podla X zodpoveda fyzikalnemu poradiu merania
    (na rozdiel od spojitej ciary tu nehrozi 'vertikalny useky s viacerymi Y
    na jedno X', kedze kazdy marker je JEDEN diskretny nameran bod, nie
    spojita krivka ktoru treba sledovat pixel-po-pixeli).
    Vrati (py, px) analogicky k digitize_mask_to_path."""
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_points, connectivity=8)
    if n_labels <= 1:
        raise RuntimeError("Po izolacii nezostali ziadne bodove markery - skontroluj obrazok.")

    valid = [i for i in range(1, n_labels) if stats[i, cv2.CC_STAT_AREA] >= min_marker_area]
    if not valid:
        raise RuntimeError(
            f"Ziadny marker nedosiahol minimalnu plochu {min_marker_area}px - "
            "skontroluj kvalitu/rozlisenie obrazka."
        )

    pts = centroids[valid]  # (x, y) poradie z OpenCV
    order = np.argsort(pts[:, 0])
    pts = pts[order]
    return pts[:, 1], pts[:, 0]  # (py, px) - konzistentne s digitize_mask_to_path





# ----------------------------------------------------------------------------
# 1d) Digitalizacia masky do usporiadanej cesty (skeleton + graf)
# ----------------------------------------------------------------------------

def _neighbors(coords_set, y, x):
    result = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            if (y + dy, x + dx) in coords_set:
                result.append((y + dy, x + dx))
    return result


def digitize_mask_to_path(mask_curve: np.ndarray):
    """Skeletonizuje masku a najde najdlhsiu cestu medzi dvoma koncovymi bodmi.
    Toto (namiesto 'priemer y na kazdy x-stlpec') je nutne pre krivky s vertikalnymi
    / prevratenymi usekmi (strmy pokles po Rm, strmy elasticky nabeh) - column-mean
    by take useky proste stratil/skreslil. Vrati (py, px) - pole pixel-y a pixel-x
    suradnic V PORADI POZDLZ KRIVKY (nie zoradene podla x!)."""
    skeleton = skeletonize(mask_curve > 0)
    ys, xs = np.where(skeleton)
    coords = set(zip(ys.tolist(), xs.tolist()))
    if len(coords) < 2:
        raise RuntimeError("Skeleton krivky je prazdny/prilis maly.")

    endpoints = [(y, x) for (y, x) in coords if len(_neighbors(coords, y, x)) == 1]
    if not endpoints:
        endpoints = list(coords)[:1]

    graph = nx.Graph()
    for (y, x) in coords:
        for (ny, nx_) in _neighbors(coords, y, x):
            graph.add_edge((y, x), (ny, nx_), weight=np.hypot(ny - y, nx_ - x))

    best_path, best_length = None, -1.0
    for start in endpoints:
        lengths = nx.single_source_dijkstra_path_length(graph, start, weight="weight")
        farthest = max(lengths, key=lengths.get)
        if lengths[farthest] > best_length:
            best_length = lengths[farthest]
            best_path = (start, farthest)

    path_nodes = nx.shortest_path(graph, best_path[0], best_path[1], weight="weight")
    path_arr = np.array(path_nodes)
    return path_arr[:, 0], path_arr[:, 1]


# ----------------------------------------------------------------------------
# 1e) Verejne vysokourovnove API - toto pouzivaju ostatne moduly a GUI
# ----------------------------------------------------------------------------

UPSCALE_FACTOR = 4  # bikubicke zvacsenie pred izolaciou - viz poznamka nizsie


def digitize_curve(crop_bgr: np.ndarray, calib: AxisCalibration, crop_offset: tuple,
                    upscale_factor: int = UPSCALE_FACTOR):
    """Hlavna vstupna funkcia modulu 1: z orezanej oblasti (BGR) grafu vyrobi
    usporiadane (strain, stress) numpy polia.

    Bikubicke zvacsenie pred izolaciou/digitalizaciou: anti-aliasing na hranach
    ciary pri zvacseni nesie sub-pixelovu informaciu, ktoru natívne (nizke)
    rozlisenie zdrojoveho obrazka nema - vyrazne to spresnuje najme neskorsi
    vypocet modulu pruznosti E (v module 2), kde je potrebna jemnejsia citlivost
    nez 1 povodny pixel.

    AUTOMATICKY si vyberie strategiu izolacie (spojita ciara vs. bodove markery)
    podla detect_curve_style - viz jej docstring pre kriteria rozhodnutia.

    Vracia: (strain, stress, color, curve_style) - color je detegovana farba
    krivky, curve_style je "line"/"dots" (pre info/log/databazove metadata).
    """
    crop_big = cv2.resize(crop_bgr, None, fx=upscale_factor, fy=upscale_factor,
                           interpolation=cv2.INTER_CUBIC)
    hsv_crop = cv2.cvtColor(crop_big, cv2.COLOR_BGR2HSV)
    color = detect_curve_color(hsv_crop)

    curve_style = detect_curve_style(crop_big, color)

    if curve_style == "dots":
        mask_points = isolate_curve_points_mask(crop_big, color)
        py, px = extract_point_centroids(mask_points)
    else:
        mask_curve = isolate_curve_mask(crop_big, color)
        py, px = digitize_mask_to_path(mask_curve)

    px = px / upscale_factor
    py = py / upscale_factor

    off_x, off_y = crop_offset
    px_orig = px + off_x
    py_orig = py + off_y

    strain = calib.px_to_x(px_orig)
    stress = calib.px_to_y(py_orig)
    if strain[0] > strain[-1]:
        strain, stress = strain[::-1], stress[::-1]

    return strain, stress, color, curve_style
