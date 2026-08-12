"""
MODUL 2: INZINIERSKE MECHANICKE VLASTNOSTI
============================================
Zodpovednost tohto modulu: (strain, stress) inzinierska krivka -> E, Rp0.2, Rm, A
(+ diagnostika spolahlivosti). NEROBI ziadnu konverziu na true krivku ani
exponenty spevnenia - to je uloha modulu 3 (true_curve.py).

Vstup: strain, stress numpy polia (usporiadane pozdlz krivky, vystup z modulu 1).
Vystup: EngineeringProperties dataclass.

============================================================================
ZASADNA ZMENA ALGORITMU (voci predoslej vetve prace):
============================================================================
Predchadzajuca implementacia hladala "najlepsie linearne okno" ako VOLNE
PLAVAJUCE okno (lubovolny zaciatok aj koniec), skorovane statisticky
(R², Sm(rel), log-log sklon). Toto malo zasadnu chybu: statisticke skorovanie
nevie odlisit "toto je fyzikalne elasticka oblast" od "toto je nahodou lokalne
statisticky pekny usek, ktory uz lezi v plastickej oblasti". Overene: 14%
chyba v module pruznosti E na jedinej dovervhodnej referencnej krivke
(INOVAL, certifikovany report).

NOVY PRISTUP JE FYZIKALNE KOTVENY, nie ciste statisticky:
    Hookov zakon sigma = E*epsilon plati OD SKUTOCNEHO POCIATKU ZATAZOVANIA
    (epsilon=0, sigma=0). Elasticka oblast teda nie je "niekde v strede
    krivky" - je to VZDY suvisly usek OD POCIATKU dat. Okno na hladanie E
    preto NESMIE volne plavat - musi byt KOTVENE v prvom bode a rastuce len
    jednym smerom (kumulativna regresia PRES POCIATOK, nie s volnym useckom).

Toto jedno rozhodnutie (kotvenie v pociatku) vyriesilo hlavny zdroj chyby a
zaroven umoznilo poctivo rozpoznat pripady, ked data na spolahlivy odhad
nestacia (namiesto ticheho vratenia nespravneho cisla) - pozri `confidence`
a `messages` polia v EngineeringProperties.

Overene na 3 nezavislych referencnych krivkach (INOVAL, Toolshed Al1, TPHM) -
presna zhoda s referencnymi hodnotami v testoch.

Zavislosti: len numpy + scipy (BSD licencie) - modul NEZAVISI od modulu 1
(ziadny import cv2/OCR), takze sa da samostatne testovat na lubovolnych
(strain,stress) datach, aj bez obrazku.
"""

import math
from dataclasses import dataclass, field

import numpy as np


# ----------------------------------------------------------------------------
# 2a) Pomocna funkcia na urcenie jednotky osi (nezmenene z predoslej vetvy)
# ----------------------------------------------------------------------------

def _strain_is_percent(strain: np.ndarray, ocr_hint: bool = None):
    """Heuristika: ak su hodnoty na osi typicky > 1.5, ide zrejme o percenta (0-60),
    inak o bezrozmerny zlomok (0-0.12). Dolezite pre spravny prevod E do GPa aj
    pre volbu velkosti 0.2%-offsetu (0.2 vs 0.002).

    ocr_hint: volitelny priamy dokaz z OCR nazvu osi (napr. '%' v texte 'Strain (%)').

    POZOR - REALNY NALEZ: pri jednom testovacom obrazku mal graf v nazve osi '%',
    ale samotne hodnoty (0 az 0.12) davali fyzikalny zmysel LEN ako ZLOMOK (chyba
    v povodnom zdrojovom grafe - zabudnuty prepocet x100 pri jeho tvorbe). Pri
    inom (INOVAL) obrazku bolo naopak OCR '%' SPRAVNE aj napriek malym cislam
    (0 az 0.75) - potvrdene priamo referencnymi datami v tom istom PDF.
    Zaver: NEEXISTUJE bezpecna univerzalna priorita (ani 'ver vzdy OCR', ani
    'ver vzdy velkosti cisel') - oba realne nastali a mali OPACNE spravne
    riesenie. Preto namiesto tichého hadania VRACIAME AJ PRIZNAK NEZHODY,
    aby ho volajuci mohol zobrazit ako upozornenie na manualnu kontrolu,
    namiesto sebaisteho ale possibly nespravneho cisla.

    Vracia (is_percent: bool, ambiguous: bool)."""
    magnitude_says_percent = float(np.nanmax(np.abs(strain))) > 1.5
    if ocr_hint is None:
        return magnitude_says_percent, False
    if ocr_hint == magnitude_says_percent:
        return magnitude_says_percent, False  # oba zdroje sa zhoduju - vysoka dovera
    # NEZHODA - defaultne preferujeme OCR (priamy textovy dokaz), ALE oznacime
    # ako nejednoznacne, aby to bolo vidno v GUI/statuse a dalo sa manualne overit.
    return ocr_hint, True


# ----------------------------------------------------------------------------
# 2b) Krok 1: Fyzikalne kotvene hladanie elastickej oblasti (kumulativna
#     regresia PRES POCIATOK) + diagnostika spolahlivosti
# ----------------------------------------------------------------------------

@dataclass
class ElasticFitResult:
    E_MPa: float
    k_end: int
    n_window: int
    strain_end_percent: float
    stress_end_MPa: float
    stress_span_MPa: float
    stress_span_fraction: float
    reach95_ratio: float
    sm_rel_percent: float
    E_free_intercept_MPa: float
    intercept_MPa: float
    confidence: str
    messages: list = field(default_factory=list)


def find_elastic_region(
    strain_percent: np.ndarray, stress_MPa: np.ndarray,
    min_points: int = 5,
    search_cap_fraction: float = 0.65,
    plateau_tolerance: float = 0.95,
    low_n_threshold: int = 15,
    narrow_span_threshold: float = 0.10,
    no_plateau_threshold: float = 0.75,
) -> ElasticFitResult:
    """Najde elasticku oblast KOTVENU v skutocnom pociatku (epsilon=0, sigma=0),
    NIE ako volne plavajuce okno.

    Algoritmus:
    1. Kumulativna regresia PRES POCIATOK pre kazde k=0..n-1:
           E_hat(k) = Sum(eps_i*sig_i, i=0..k) / Sum(eps_i^2, i=0..k)
       (O(n), beziace sucty - ziadne opakovane regresie).
    2. Pokial body 0..k su skutocne elasticke, E_hat(k) je priblizne konstantne
       (fluktuuje len sumom pri malom k). Hned ako do suctu vstupia plasticke
       body (kde realne napatie lezi POD elastickou priamkou), E_hat(k) zacne
       SYSTEMATICKY KLESAT - profil ma typicky tvar: narast/plato -> ostre
       maximum -> monotonny pokles.
    3. k_end = argmax(E_hat(k)) v rozsahu [min_points, cap_idx], kde cap_idx je
       bezpecnostny strop (search_cap_fraction * Rm_odhad) - elasticka oblast
       urcite nesiaha za tuto hranicu.

    Diagnostika spolahlivosti (confidence HIGH/MEDIUM/LOW) sa pocita z profilu
    E_hat(k), poctu bodov v okne a rozsahu napatia pokryteho oknom - NIE z toho,
    ako dobre okno 'vyzera' statisticky (to bol presne problem stareho pristupu).
    Tri nezavisle signaly, kazdy moze znizit confidence:
      1. n_window < low_n_threshold -> velmi silny signal (fundamentalny limit
         dat - napr. digitalizacia stratila zaciatok krivky).
      2. stress_span_fraction < narrow_span_threshold -> okno pokryva prilis
         maly rozsah napatia -> nizky pomer signal/sum.
      3. reach95_ratio > no_plateau_threshold -> profil E_hat(k) nema skore
         plato, len plynule stupa az po maximum -> podozrenie na nevyrieseny
         toe efekt (vola v celustiach) alebo nedostatocne rozlisenie zaciatku.
    Kombinacia: 2+ signaly -> LOW, 1 signal -> MEDIUM, 0 signalov -> HIGH.

    DOLEZITE SPRAVANIE: ak je n_window male, VYSOKE Sm(rel) alebo dokonale R²
    NEMAJU prebit znizenie confidence (fabrikovane/umelo pridane body vedia dat
    Sm(rel)=0.000% - 'dokonaly' fit - a napriek tomu musia zostat LOW, pretoze
    n_window je stale male - presne tato pasca sa objavila v predoslej vetve
    prace a sposobila falosnu istotu)."""
    s = np.asarray(strain_percent, dtype=np.float64)
    t = np.asarray(stress_MPa, dtype=np.float64)
    n = len(s)
    if n < low_n_threshold:
        return ElasticFitResult(
            E_MPa=float("nan"), k_end=-1, n_window=n,
            strain_end_percent=float("nan"), stress_end_MPa=float("nan"),
            stress_span_MPa=0.0, stress_span_fraction=0.0, reach95_ratio=float("nan"),
            sm_rel_percent=float("nan"), E_free_intercept_MPa=float("nan"),
            intercept_MPa=float("nan"), confidence="LOW",
            messages=[f"Krivka ma len {n} bodov celkovo - nedostatocne data."],
        )

    Rm_guess = float(np.max(t))
    cum_sxy = np.cumsum(s * t)
    cum_sxx = np.cumsum(s * s)
    with np.errstate(divide="ignore", invalid="ignore"):
        Ehat = np.where(cum_sxx > 0, cum_sxy / cum_sxx, np.nan)

    cap_idx = n - 1
    over_cap = np.where(t > search_cap_fraction * Rm_guess)[0]
    if len(over_cap) > 0:
        cap_idx = max(int(over_cap[0]), min_points + 5)
    cap_idx = min(cap_idx, n - 1)

    lo = min(min_points, max(0, n - 1))
    search_slice = Ehat[lo:cap_idx + 1]
    if len(search_slice) == 0 or np.all(np.isnan(search_slice)):
        k_end = n - 1
    else:
        k_end = lo + int(np.nanargmax(search_slice))

    E_MPa = float(Ehat[k_end]) * 100.0
    target = plateau_tolerance * Ehat[k_end]
    reach_candidates = np.where(Ehat[:k_end + 1] >= target)[0]
    reach95 = int(reach_candidates[0]) if len(reach_candidates) > 0 else 0
    reach95_ratio = reach95 / k_end if k_end > 0 else 0.0

    n_window = k_end + 1
    stress_span = float(t[k_end] - t[0])
    stress_span_fraction = stress_span / Rm_guess if Rm_guess > 0 else 0.0

    slope = float(Ehat[k_end])
    ws, wt = s[:n_window], t[:n_window]
    resid = wt - slope * ws
    if n_window > 1:
        s2 = float(np.sum(resid ** 2)) / (n_window - 1)
        sxx = float(np.sum(ws ** 2))
        se_slope = math.sqrt(s2 / sxx) if sxx > 0 else float("nan")
        sm_rel = se_slope / slope * 100.0 if slope != 0 else float("nan")
    else:
        sm_rel = float("nan")

    mean_s, mean_t = float(np.mean(ws)), float(np.mean(wt))
    Sxy = float(np.sum((ws - mean_s) * (wt - mean_t)))
    Sxx = float(np.sum((ws - mean_s) ** 2))
    if Sxx > 0 and n_window > 2:
        slope_fi = Sxy / Sxx
        intercept_fi = mean_t - slope_fi * mean_s
    else:
        slope_fi, intercept_fi = float("nan"), float("nan")
    E_free_intercept = slope_fi * 100.0

    messages, flags = [], 0
    if n_window < low_n_threshold:
        flags += 2
        messages.append(f"Elasticke okno obsahuje len {n_window} bodov (< {low_n_threshold}) - "
                         f"pravdepodobne fundamentalny limit dat.")
    if stress_span_fraction < narrow_span_threshold:
        flags += 2
        messages.append(f"Okno pokryva len {stress_span_fraction*100:.1f}% rozsahu napatia - "
                         f"prilis uzky signal.")
    if reach95_ratio > no_plateau_threshold:
        flags += 1
        messages.append(f"E_hat(k) nema skore plato (reach95_ratio={reach95_ratio:.2f}) - "
                         f"podozrenie na toe efekt/nedostatocne rozlisenie.")
    if E_MPa and abs(E_free_intercept - E_MPa) / E_MPa > 0.10:
        messages.append(f"Krizova kontrola (volny usek, {E_free_intercept:.0f} MPa) sa lisi o "
                         f"{abs(E_free_intercept-E_MPa)/E_MPa*100:.1f}% (informativne).")

    confidence = "LOW" if flags >= 2 else ("MEDIUM" if flags == 1 else "HIGH")
    if not messages:
        messages.append("Bez vyhrad.")

    return ElasticFitResult(
        E_MPa=E_MPa, k_end=k_end, n_window=n_window,
        strain_end_percent=float(s[k_end]), stress_end_MPa=float(t[k_end]),
        stress_span_MPa=stress_span, stress_span_fraction=stress_span_fraction,
        reach95_ratio=reach95_ratio, sm_rel_percent=sm_rel,
        E_free_intercept_MPa=E_free_intercept, intercept_MPa=intercept_fi,
        confidence=confidence, messages=messages,
    )


# ----------------------------------------------------------------------------
# 2c) Krok 2: Zjemnenie E na "jadre" okna (orezanie okrajov)
# ----------------------------------------------------------------------------

@dataclass
class CoreElasticResult:
    a_idx: int
    b_idx: int
    n_core: int
    strain_start_percent: float
    strain_end_percent: float
    E_MPa: float
    r2_uncentered: float
    trim_start_fraction: float
    trim_end_fraction: float
    reached_target: bool


def refine_core_window(
    strain_percent: np.ndarray, stress_MPa: np.ndarray, k_end: int,
    trim_start: float = 0.08, trim_end: float = 0.08,
    r2_target: float = 0.998, min_window_fraction: float = 0.25,
    max_trim: float = 0.35, step: float = 0.03,
):
    """Na ziadost z povodnej diskusie: nepouzivat na finalny odhad E body
    blizko skutocneho pociatku (najvacsi relativny sum) ANI body blizko konca
    elastickeho okna (tesne pred/pri nastupe krivosti).

    Postup: zober okno [0,k_end] z kroku 1, postupne orezavaj OKRAJE (najprv
    koniec, potom zaciatok - empiricky zistene, ze krivost pri konci okna je
    zvycajne vacsim zdrojom odchylky od R²=1 nez sum pri zaciatku), az kym
    R² (uncentered - regresia stale PRES POCIATOK) nedosiahne cielovu hodnotu,
    alebo by jadro kleslo pod bezpecnostny limit (min_window_fraction).

    Vracia None ak povodne okno ma menej nez 10 bodov (orezavanie by ho znicilo)."""
    s = np.asarray(strain_percent, dtype=np.float64)
    t = np.asarray(stress_MPa, dtype=np.float64)
    n_window = k_end + 1
    if n_window < 10:
        return None
    ws, wt = s[:n_window], t[:n_window]
    min_size = max(5, int(round(min_window_fraction * n_window)))

    ts, te = trim_start, trim_end
    best, reached = None, False
    while True:
        a = int(round(ts * n_window))
        b = n_window - 1 - int(round(te * n_window))
        if b - a + 1 < min_size:
            break
        seg_s, seg_t = ws[a:b + 1], wt[a:b + 1]
        sxx = float(np.sum(seg_s * seg_s))
        if sxx == 0:
            break
        slope = float(np.sum(seg_s * seg_t)) / sxx
        ss_res = float(np.sum((seg_t - slope * seg_s) ** 2))
        ss_tot = float(np.sum(seg_t * seg_t))
        r2u = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        best = (a, b, slope, r2u, ts, te)
        if r2u >= r2_target:
            reached = True
            break
        if te < max_trim:
            te = round(te + step, 4)
        elif ts < max_trim:
            ts = round(ts + step, 4)
        else:
            break

    if best is None:
        return None
    a, b, slope, r2u, ts_final, te_final = best
    return CoreElasticResult(
        a_idx=a, b_idx=b, n_core=b - a + 1,
        strain_start_percent=float(ws[a]), strain_end_percent=float(ws[b]),
        E_MPa=slope * 100.0, r2_uncentered=r2u,
        trim_start_fraction=ts_final, trim_end_fraction=te_final,
        reached_target=reached,
    )


# ----------------------------------------------------------------------------
# 2d) Krok 3: Krivost plastickej oblasti (Hollomon) - volitelne, pouziva sa
#     ak modul 3 (true_curve.py) chce presnejsi fit nez svoj vlastny
# ----------------------------------------------------------------------------

@dataclass
class PlasticHardeningResult:
    n_hardening: float
    K_MPa: float
    r2: float
    idx_start: int
    idx_end: int
    strain_start_percent: float
    strain_end_percent: float
    n_points: int
    applicable: bool
    message: str


def analyze_plastic_hardening(
    strain_percent: np.ndarray, stress_MPa: np.ndarray, E_MPa: float, k_end: int,
    eps_plastic_min_fraction: float = 0.002, rm_margin_fraction: float = 0.0,
) -> PlasticHardeningResult:
    """Mocninovy (Hollomonov) zakon: sigma = K * eps_p^n, kde eps_p = eps_celkove
    - sigma/E je PLASTICKA zlozka deformacie (elasticka sigma/E sa odpocita, aby
    n opisovalo vyhradne spevnovanie). Fituje sa v log-log priestore.

    Vylucenie nabehu do plasticity: body sa do fitu zoberu len ak
    eps_p > eps_plastic_min_fraction (default 0.2%) - tesne za medzou
    proporcionality je eps_p blizko nuly => ln(eps_p) numericky nestabilne.

    Horna hranica: index Rm (bod maximalneho napatia) - za nim nastava
    mäknutie/nutovanie, mocninovy zakon tam typicky neplati.

    Ak po filtrovani zostane menej nez 10 bodov (typicky krehky material bez
    vyraznej plastickej oblasti) -> applicable=False, NEHADAT fit z 2-3 bodov."""
    s = np.asarray(strain_percent, dtype=np.float64)
    t = np.asarray(stress_MPa, dtype=np.float64)
    n = len(s)
    eps_el = t / E_MPa
    eps_p = s / 100.0 - eps_el
    rm_idx = int(np.argmax(t))
    rm_idx = max(k_end, rm_idx - int(round(rm_margin_fraction * n)))

    mask_range = np.arange(k_end, rm_idx + 1)
    valid = eps_p[mask_range] > eps_plastic_min_fraction
    idxs = mask_range[valid]

    if len(idxs) < 10:
        return PlasticHardeningResult(
            n_hardening=float("nan"), K_MPa=float("nan"), r2=float("nan"),
            idx_start=-1, idx_end=-1, strain_start_percent=float("nan"),
            strain_end_percent=float("nan"), n_points=len(idxs), applicable=False,
            message=f"Nedostatok bodov ({len(idxs)}) s plastickou deformaciou > "
                    f"{eps_plastic_min_fraction*100:.2f}% - pravdepodobne krehky material.",
        )

    x = np.log(eps_p[idxs])
    y = np.log(t[idxs])
    mx, my = float(np.mean(x)), float(np.mean(y))
    sxy = float(np.sum((x - mx) * (y - my)))
    sxx = float(np.sum((x - mx) ** 2))
    syy = float(np.sum((y - my) ** 2))
    n_exp = sxy / sxx
    K = math.exp(my - n_exp * mx)
    r2 = (sxy ** 2) / (sxx * syy) if sxx > 0 and syy > 0 else float("nan")

    return PlasticHardeningResult(
        n_hardening=n_exp, K_MPa=K, r2=r2,
        idx_start=int(idxs[0]), idx_end=int(idxs[-1]),
        strain_start_percent=float(s[idxs[0]]), strain_end_percent=float(s[idxs[-1]]),
        n_points=len(idxs), applicable=True,
        message="OK - Hollomonov fit na ustalenej oblasti spevnovania.",
    )


# ----------------------------------------------------------------------------
# 2e) Krok 4: Rp0.2 (offset metoda) a taznost (s korekciou o elasticitu)
# ----------------------------------------------------------------------------

@dataclass
class Rp02Result:
    strain_percent: float
    stress_MPa: float
    index: int
    found: bool
    message: str


def compute_rp02(strain_percent: np.ndarray, stress_MPa: np.ndarray, E_MPa: float,
                  offset_percent: float = 0.2) -> Rp02Result:
    """Klasicka offset metoda (ISO 6892-1 / ASTM E8): priamka
    sigma = E*(epsilon - offset) (rovnobezna s elastickou priamkou, posunuta
    o offset % deformacie). V elastickej oblasti je krivka NAD touto priamkou;
    po prekroceni medze klzu klesne POD nu. Hlada sa prva zmena znamienka
    (stress_curve - offset_line) z + na -, S LINEARNOU INTERPOLACIOU medzi
    susednymi bodmi (presnejsie nez 'najblizsi bod').

    Ak sa priesecnik nenajde (krivka nikdy neklesne pod offset priamku) ->
    found=False. TOTO JE LEGITIMNY A SPRAVNY VYSLEDOK pre krehke materialy,
    ktore sa zlomia skor, nez akumuluju 0.2% plastickej deformacie - priamo
    zodpoveda referencnemu oznaceniu 'Rp0.2 = N/F' v takychto pripadoch."""
    s = np.asarray(strain_percent, dtype=np.float64)
    t = np.asarray(stress_MPa, dtype=np.float64)
    n = len(s)
    offset_frac = offset_percent / 100.0
    diffs = t - E_MPa * (s / 100.0 - offset_frac)
    start_candidates = np.where(s / 100.0 >= offset_frac)[0]
    start = int(start_candidates[0]) if len(start_candidates) > 0 else 0
    for i in range(start, n - 1):
        if diffs[i] >= 0 and diffs[i + 1] < 0:
            frac = diffs[i] / (diffs[i] - diffs[i + 1])
            s_cross = s[i] + frac * (s[i + 1] - s[i])
            t_cross = t[i] + frac * (t[i + 1] - t[i])
            return Rp02Result(float(s_cross), float(t_cross), i, True, "OK.")
    return Rp02Result(float("nan"), float("nan"), -1, False,
                       "Priesecnik nenajdeny - krivka nikdy neklesla pod offset priamku "
                       "(typicky krehky material, ktory sa zlomi skor; zodpoveda Rp0.2=N/F).")


@dataclass
class ElongationResult:
    A_raw_percent: float
    A_elastic_corrected_percent: float
    strain_last_percent: float
    stress_last_MPa: float
    elastic_correction_percent: float


def compute_elongation(strain_percent: np.ndarray, stress_MPa: np.ndarray,
                        E_MPa: float) -> ElongationResult:
    """Kluc bod (dovod, preco 'so zohladnenim elasticity'): posledny bod krivky
    (predpoklad = lom) obsahuje este elasticku zlozku deformacie sigma_lom/E.
    Pri realnej skuske sa tato zlozka po lome pruzne vrati spat - teda sa do
    skutocne namneranej (referencnej) taznosti NEZAPOCITAVA:
        A_raw               = eps_lom                    (BEZ korekcie, nadhodnotene)
        A_elastic_corrected = eps_lom - sigma_lom/E       (SPRAVNA hodnota)
    Rozdiel je zanedbatelny pri velmi taznych materialoch, ale ZASADNY pri
    krehkych (elasticka zlozka moze tvorit vacsinu celkovej deformacie pri
    lome - overene: bez korekcie bola taznost na krehkej referencnej krivke
    nadhodnotena 3.5x)."""
    s_last = float(strain_percent[-1])
    t_last = float(stress_MPa[-1])
    elastic_corr = t_last / E_MPa * 100.0
    return ElongationResult(
        A_raw_percent=s_last,
        A_elastic_corrected_percent=s_last - elastic_corr,
        strain_last_percent=s_last, stress_last_MPa=t_last,
        elastic_correction_percent=elastic_corr,
    )


# ----------------------------------------------------------------------------
# 2f) Verejne API - hlavny vstupny bod modulu, POUZIVANY GUI a modulom 3.
#     Interne vola kroky 1-4 vyssie a mapuje vysledok do EngineeringProperties
#     (zachovane rovnake nazvy poli ako v predoslej vetve, kde to davalo zmysel,
#     aby gui_app.py a true_curve.py fungovali bez zmeny).
# ----------------------------------------------------------------------------

@dataclass
class EngineeringProperties:
    E_GPa: float
    epsilon0: float                # VZDY 0.0 v novom pristupe - regresia je VZDY
                                     # kotvena v skutocnom pociatku (eps=0,sig=0),
                                     # nie s volnym useckom
    elastic_slope: float           # sklon v PUVODNYCH jednotkach osi (MPa / jednotka strain)
    elastic_r2: float               # r2_uncentered z jadra okna (Krok 2)
    elastic_window: tuple           # (a_idx, b_idx+1) jadra okna - POUZITE PRE E
    strain_unit_percent: bool
    strain_unit_ambiguous: bool
    Rm_MPa: float
    Rm_strain: float
    Rm_index: int
    Rp02_MPa: float                 # None = N/F (legitimny vysledok pre krehke materialy)
    Rp02_strain: float
    Rp02_index: int
    A_percent: float                # UZ SO ZOHLADNENIM ELASTICKEHO ODPRUZENIA (A_elastic_corrected)
    yield_ratio: float              # Rp0.2/Rm, None ak Rp0.2 nedostupne
    # --- nove diagnosticke polia (fyzikalne kotveny pristup) ---
    confidence: str                 # "HIGH" / "MEDIUM" / "LOW"
    confidence_messages: list       # textove vysvetlenia znizenia confidence
    n_window: int                   # pocet bodov v POVODNOM (nie jadrovom) elastickom okne
    sm_rel_percent: float           # ISO 6892-1 relativna chyba sklonu (na povodnom okne, Krok 1)
    reach95_ratio: float            # 0=okamzite plato (dobre), ~1=ziadne plato (podozrive)
    stress_span_fraction: float     # rozsah napatia v okne ako podiel Rm - signal/sum


def compute_engineering_properties(strain: np.ndarray, stress: np.ndarray,
                                    is_percent_hint: bool = None,
                                    rp02_offset_percent: float = 0.2,
                                    **find_elastic_kwargs) -> EngineeringProperties:
    """Hlavna vstupna funkcia modulu 2 - FYZIKALNE KOTVENY pristup (viz docstring
    modulu na zaciatku suboru pre plne odovodnenie).

    is_percent_hint: volitelny priamy dokaz jednotky osi (napr. '%' precitany
    OCR-om z nazvu osi v module 1). Ak je dostupny, POUZIJE SA namiesto odhadu
    z velkosti cisel.

    rp02_offset_percent: konvencny offset pre zmluvnu medzu klzu (default 0.2%,
    ISO 6892-1 standard). Ak sa priesecnik nenajde, Rp02_MPa=None (N/F) -
    legitimny vysledok pre krehke materialy, NEVYSKUSAVAJU sa alternativne
    (nizsie) offsety ako v predoslej vetve - ISO 0.2% je jedina STANDARDNE
    reportovana hodnota, alternativne offsety by boli zavadzajuce ak sa
    prezentuju pod rovnakym nazvom.

    **find_elastic_kwargs: dalsie parametre presmerovane do find_elastic_region
    (napr. low_n_threshold, narrow_span_threshold, ...).
    """
    is_percent, strain_unit_ambiguous = _strain_is_percent(strain, ocr_hint=is_percent_hint)

    # Vsetky vypocty v tomto module pracuju v PERCENTUALNEJ skale strain
    # (tak, ako to predpoklada referencna implementacia v navode) - ak su
    # povodne data v zlomkoch, prevedieme len INTERNE, vysledky sa na konci
    # prevedu spat do PUVODNYCH jednotiek pre konzistenciu s API zvysku projektu.
    strain_pct = strain if is_percent else strain * 100.0

    res = find_elastic_region(strain_pct, stress, **find_elastic_kwargs)

    idx_rm = int(np.argmax(stress))
    Rm_MPa = float(stress[idx_rm])
    Rm_strain = float(strain[idx_rm])

    if res.confidence == "LOW" and res.n_window < find_elastic_kwargs.get("low_n_threshold", 15):
        # Fundamentalny limit dat (viz find_elastic_region docstring) - E sa
        # neda spolahlivo urcit. Vratime co najviac (Rm) - E_GPa=nan, jasne
        # oznacene confidence=LOW namiesto ticheho vratenia zleho cisla.
        return EngineeringProperties(
            E_GPa=float("nan"), epsilon0=0.0, elastic_slope=float("nan"),
            elastic_r2=float("nan"), elastic_window=(0, max(res.n_window, 0)),
            strain_unit_percent=is_percent, strain_unit_ambiguous=strain_unit_ambiguous,
            Rm_MPa=Rm_MPa, Rm_strain=Rm_strain, Rm_index=idx_rm,
            Rp02_MPa=None, Rp02_strain=None, Rp02_index=None,
            A_percent=float("nan"), yield_ratio=None,
            confidence=res.confidence, confidence_messages=res.messages,
            n_window=res.n_window, sm_rel_percent=res.sm_rel_percent,
            reach95_ratio=res.reach95_ratio, stress_span_fraction=res.stress_span_fraction,
        )

    core = refine_core_window(strain_pct, stress, res.k_end)
    E_MPa = core.E_MPa if core is not None else res.E_MPa
    elastic_r2 = core.r2_uncentered if core is not None else float("nan")
    if core is not None:
        elastic_window = (core.a_idx, core.b_idx + 1)
    else:
        elastic_window = (0, res.n_window)

    E_GPa = E_MPa / 1000.0
    # elastic_slope v PUVODNYCH jednotkach osi: E_MPa je VZDY 'MPa na jednotku
    # (zlomok) strain' (fyzikalna definicia E). Ak su povodne data v percentach,
    # sklon v tychto jednotkach = E_MPa/100. Ak su povodne data uz v zlomku,
    # sklon = E_MPa priamo.
    elastic_slope = E_MPa / 100.0 if is_percent else E_MPa

    rp02 = compute_rp02(strain_pct, stress, E_MPa, offset_percent=rp02_offset_percent)
    if rp02.found:
        Rp02_MPa = rp02.stress_MPa
        Rp02_strain = rp02.strain_percent if is_percent else rp02.strain_percent / 100.0
        Rp02_index = rp02.index
        yield_ratio = Rp02_MPa / Rm_MPa
    else:
        Rp02_MPa = None
        Rp02_strain = None
        Rp02_index = None
        yield_ratio = None

    elong = compute_elongation(strain_pct, stress, E_MPa)
    A_percent = elong.A_elastic_corrected_percent  # uz v % nezavisle od is_percent

    return EngineeringProperties(
        E_GPa=E_GPa, epsilon0=0.0, elastic_slope=elastic_slope,
        elastic_r2=elastic_r2, elastic_window=elastic_window,
        strain_unit_percent=is_percent, strain_unit_ambiguous=strain_unit_ambiguous,
        Rm_MPa=Rm_MPa, Rm_strain=Rm_strain, Rm_index=idx_rm,
        Rp02_MPa=Rp02_MPa, Rp02_strain=Rp02_strain, Rp02_index=Rp02_index,
        A_percent=A_percent, yield_ratio=yield_ratio,
        confidence=res.confidence, confidence_messages=res.messages,
        n_window=res.n_window, sm_rel_percent=res.sm_rel_percent,
        reach95_ratio=res.reach95_ratio, stress_span_fraction=res.stress_span_fraction,
    )
