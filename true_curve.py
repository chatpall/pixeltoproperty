"""
MODUL 3: PREVOD NA TRUE KRIVKU + EXPONENT SPEVNENIA
=====================================================
Zodpovednost tohto modulu: z INZINIERSKEJ krivky (strain, stress) vyrobit TRUE
stress-strain krivku a z nej Hollomonov fit (K, n).

DOLEZITE FYZIKALNE OBMEDZENIE (navrhnute pouzivatelom projektu):
    Prevod na true krivku pomocou standardnych vzorcov
        true_strain = ln(1 + engineering_strain)
        true_stress = engineering_stress * (1 + engineering_strain)
    PLATI LEN DO MEDZE PEVNOSTI (Rm) - po nej zacina krckovanie (necking), kde
    prestava platit predpoklad ROVNOMERNEJ deformacie po celej merne dlzke, na
    ktorom su tieto vzorce (odvodene zo zachovania objemu) zalozene. Bez
    priecneho tahomera (merania zuzenia priemeru vzorky pocas skusky) sa true
    krivka po Rm korektne dopocitat neda - preto sa v tomto module KRIVKA
    ZAMERNE OREZAVA presne na Rm a dalej sa nepokracuje.

    Tento modul teda VZDY predpoklada, ze vstupna krivka je inzinierska (aj
    keby uz nahodou bola true, prevod by ju len nepatrne skreslil pri malych
    elastickych deformaciach - v praxi zanedbatelne, keďze modul E aj Rp0.2 sa
    pocitaju uz v module 2 na povodnej krivke, nie na tejto konvertovanej).

Vstup: strain, stress, is_percent (z modulu 1/2), Rm_index a Rp02_index (z modulu 2).
Vystup: TrueCurveResult dataclass (true_strain, true_stress - orezane po Rm,
        plus HollomonFit ak sa podarilo).

Zavislosti: len numpy + scipy (rovnako ako modul 2) - NEZAVISI od modulu 1.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import linregress


# ----------------------------------------------------------------------------
# 3a) Prevod inzinierska -> true (orezany po Rm)
# ----------------------------------------------------------------------------

@dataclass
class TrueCurve:
    true_strain: np.ndarray   # v ROVNAKYCH jednotkach ako vstupny strain (% alebo zlomok)
    true_stress: np.ndarray   # MPa
    valid_up_to_index: int    # index Rm v PUVODNOM (inzinierskom) poli - po tento bod


def convert_engineering_to_true(strain: np.ndarray, stress: np.ndarray,
                                 is_percent: bool, rm_index: int) -> TrueCurve:
    """Prevedie inziniersku krivku na true krivku, OREZANU presne po Rm (vratane).

    true_strain = ln(1 + e)          e = inziniersky strain ako ZLOMOK (nie %)
    true_stress = sigma_eng * (1+e)

    Body za rm_index sa ZAHODIA - za medzou pevnosti (krckovanie) uz tento
    jednoduchy prevod fyzikalne neplati (viz docstring modulu vyssie).
    """
    if rm_index < 1:
        raise RuntimeError(
            "Rm index je prilis nizko (menej nez 1 bod pred medzou pevnosti) - "
            "nie je z coho urobit zmysluplnu true krivku."
        )

    e = strain[:rm_index + 1]
    s = stress[:rm_index + 1]
    e_fraction = e / 100.0 if is_percent else e

    if np.any(e_fraction <= -1.0):
        raise RuntimeError(
            "Inzinierska deformacia <= -100% v datach - fyzikalne nezmyselne, "
            "skontroluj kalibraciu osi alebo digitalizaciu."
        )

    true_strain_fraction = np.log1p(e_fraction)     # ln(1+e), numericky stabilnejsie
    true_stress = s * (1.0 + e_fraction)

    true_strain = true_strain_fraction * 100.0 if is_percent else true_strain_fraction

    return TrueCurve(true_strain=true_strain, true_stress=true_stress,
                      valid_up_to_index=rm_index)


# ----------------------------------------------------------------------------
# 3b) Hollomonov fit: sigma_true = K * eps_true_plastic^n
# ----------------------------------------------------------------------------

@dataclass
class HollomonFit:
    K_MPa: float          # koeficient pevnosti
    n: float               # exponent deformacneho spevnenia
    r2: float               # kvalita fitu (log-log linearna regresia)
    n_points: int
    strain_range: tuple    # (min, max) true plastickeho strain pouziteho vo fite


def compute_hollomon_fit(true_curve: TrueCurve, elastic_slope: float, epsilon0: float,
                          rp02_index: int, is_percent: bool) -> HollomonFit:
    """Fit Hollomonovej rovnice sigma = K * eps_p^n na oblast ROVNOMERNEJ plastickej
    deformacie - od Rp0.2 po Rm (presne oblast, kde je tento zakon fyzikalne
    zmysluplny; pred Rp0.2 je este vyrazna elasticka zlozka, po Rm uz nie je
    v true_curve ziadne data, kedze boli orezane pri konverzii).

    Postup: log-log linearna regresia
        ln(sigma_true) = ln(K) + n * ln(eps_true_plastic)
    n = smernica, K = exp(usek).

    elastic_slope, epsilon0: z modulu 2 (EngineeringProperties.elastic_slope/epsilon0)
    - pouzite na odhad elastickej zlozky true strain (priblizne, pre male elasticke
    deformacie je rozdiel medzi eng. a true elastickym sklonom zanedbatelny).
    """
    true_strain = true_curve.true_strain
    true_stress = true_curve.true_stress
    return _hollomon_fit_core(true_strain, true_stress, elastic_slope, epsilon0, rp02_index)


def _hollomon_fit_core(strain: np.ndarray, stress: np.ndarray, elastic_slope: float,
                        epsilon0: float, rp02_index: int) -> HollomonFit:
    """Spolocne jadro Hollomon fitu, pouzitelne na LUBOVOLNE (strain,stress) pole -
    bud uz skutocne true (z convert_engineering_to_true), alebo SUROVE nekonvertovane
    data (pouzivane v _classify_curve_form na otestovanie hypotezy 'vstup uz je true')."""
    eps_elastic = stress / elastic_slope
    eps_plastic = strain - epsilon0 - eps_elastic

    lo = min(rp02_index, len(strain) - 1)
    seg_eps = eps_plastic[lo:]
    seg_sigma = stress[lo:]

    mask = seg_eps > 1e-9
    if mask.sum() < 5:
        raise RuntimeError(
            "Nedostatok bodov s kladnym plastickym strain medzi Rp0.2 a Rm "
            "pre Hollomonov fit (potrebnych aspon 5)."
        )

    log_eps = np.log(seg_eps[mask])
    log_sigma = np.log(seg_sigma[mask])
    res = linregress(log_eps, log_sigma)

    n = res.slope
    K = float(np.exp(res.intercept))
    r2 = res.rvalue ** 2

    return HollomonFit(
        K_MPa=K, n=n, r2=r2, n_points=int(mask.sum()),
        strain_range=(float(seg_eps[mask].min()), float(seg_eps[mask].max())),
    )


# ----------------------------------------------------------------------------
# 3b-bis) Klasifikacia: JE vstup uz true, alebo je to inzinierska krivka?
# ----------------------------------------------------------------------------

@dataclass
class FormClassification:
    form_guess: str          # "engineering" | "true" | "neurcite"
    r2_as_engineering: float  # R2 Hollomon fitu PO konverzii (hypoteza: vstup je eng.)
    r2_as_true: float         # R2 Hollomon fitu BEZ konverzie (hypoteza: vstup uz je true)
    margin: float             # r2_as_true - r2_as_engineering


def classify_curve_form(strain: np.ndarray, stress: np.ndarray, is_percent: bool,
                         rm_index: int, rp02_index: int, elastic_slope: float,
                         epsilon0: float, margin_threshold: float = 0.01) -> FormClassification:
    """Self-konzistentny test: Hollomonov zakon (sigma=K*eps^n) je vlastnostou TRUE
    krivky. Namiesto spoliehania sa na OCR popisok osi (nespolahlive - rotovany
    text nizkej kvality castokrat OCR zlyha) alebo na tvar poklesu po Rm (tiez
    nespolahlive - aj skutocne true krivky casto vykazuju kratky prudky pokles
    tesne pred lomom, ktory je len artefaktom okamihu pretrhnutia, nie geometrie
    krckovania), porovnavame KVALITU FITU:

    - Fitujeme Hollomona PRIAMO na surove data (hypoteza: 'vstup uz je true').
    - Fitujeme Hollomona PO konverzii eng->true (hypoteza: 'vstup je inzniersky').
    - Ktora hypoteza dava vyssie R², je pravdepodobnejsia.

    DEFAULT (bezpecny) predpoklad ostava 'engineering' (t.j. konvertovat) - na
    'true' sa prepne LEN ak je rozdiel R² jasny (> margin_threshold). Dovod:
    ak by input v skutocnosti bol inziniersky a my by sme si nesprávne mysleli
    ze je true (teda nekonvertovali), chyba pri velkych deformaciach by bola
    OMNOHO vacsia, nez opacna chyba (zbytocne konvertovat uz-true data pri
    malych/strednych deformaciach, kde je skreslenie male). Preto potrebujeme
    jasny dukaz predtym, nez zmenime defaultne (bezpecnejsie) spravanie.

    Overene na 2 realnych testovacich obrazkoch: na obrazku, ktory bol v nazve
    suboru oznaceny ako 'true stress-strain', test spravne vratil 'true' s
    jasnym rozdielom R² (0.987 vs 0.970). Na druhom obrazku (bez jasneho
    oznacenia) vysiel rozdiel R² zanedbatelny (~0.0001) - spravne vratene ako
    'neurcite' namiesto vymyslania si istoty, ktora tam nie je.
    """
    r2_as_engineering = compute_hollomon_fit(
        convert_engineering_to_true(strain, stress, is_percent, rm_index),
        elastic_slope, epsilon0, rp02_index, is_percent,
    ).r2
    r2_as_true = _hollomon_fit_core(
        strain[:rm_index + 1], stress[:rm_index + 1], elastic_slope, epsilon0, rp02_index,
    ).r2

    margin = r2_as_true - r2_as_engineering
    if margin > margin_threshold:
        form_guess = "true"
    elif margin < -margin_threshold:
        form_guess = "engineering"
    else:
        form_guess = "neurcite"

    return FormClassification(
        form_guess=form_guess, r2_as_engineering=r2_as_engineering,
        r2_as_true=r2_as_true, margin=margin,
    )


# ----------------------------------------------------------------------------
# 3c) Verejne API - jedno volanie pre vsetko naraz (konverzia + klasifikacia + fit)
# ----------------------------------------------------------------------------

@dataclass
class TrueCurveResult:
    true_curve: TrueCurve            # ZOBRAZOVANA krivka - konvertovana AK form_guess=="engineering"
                                       # alebo "neurcite" (bezpecny default), ALE SUROVA (nekonvertovana)
                                       # data ak form_guess=="true" (viz nizsie preco)
    hollomon: HollomonFit            # fit podla SKUTOCNE POUZITEJ formy (form_used)
    classification: FormClassification  # AUTOMATICKY NAVRH (len informativny, viz form_used)
    form_used: str                    # "engineering" | "true" - forma SKUTOCNE pouzita vo vypocte
    form_was_user_confirmed: bool     # True ak form_used pochadza z explicitneho potvrdenia
                                        # pouzivatelom (form_override), False ak z automatickeho
                                        # navrhu (classification.form_guess) bez potvrdenia


def compute_true_curve_and_hollomon(strain: np.ndarray, stress: np.ndarray,
                                     is_percent: bool, rm_index: int, rp02_index: int,
                                     elastic_slope: float, epsilon0: float,
                                     form_override: str = None) -> TrueCurveResult:
    """Hlavna vstupna funkcia modulu 3.

    1) Skonvertuje vstup na true krivku (predpoklad: vstup je inziniersky), orezanu po Rm.
    2) Klasifikuje, ci vstup nebol NAHODOU uz v true tvare (classify_curve_form) - toto
       je len NAVRH, viz nizsie.
    3) Ako 'hollomon' vrati fit podla SKUTOCNE POUZITEJ formy - tou je `form_override`,
       ak je zadany, inak automaticky klasifikovany `form_guess`.

    form_override: "engineering" | "true" | None. POUZIVATEL PROJEKTU SI VSIMOL, ze
    automaticka klasifikacia (margin_threshold=0.01 v classify_curve_form) sa vie
    pomylit na hranicnych pripadoch - napr. realne vedel, ze zdrojovy obrazok bol
    inziniersky, ale algoritmus ho vyhodnotil ako 'true' (mala numericka vyhoda R²
    bez konverzie, nie skutocny fyzikalny fakt). AUTOMATICKA KLASIFIKACIA SA PRETO
    POUZIVA UZ LEN AKO NAVRH (s dokazmi r2_as_true/r2_as_engineering) - konecne
    rozhodnutie ma potvrdit POUZIVATEL v GUI (checkbox/radio), nie tichy algoritmus.
    Ak form_override nie je zadany (None), pouzije sa automaticky navrh ako fallback
    (napr. pri programovom/skriptovom pouziti modulu bez GUI)."""
    converted_curve = convert_engineering_to_true(strain, stress, is_percent, rm_index)
    classification = classify_curve_form(
        strain, stress, is_percent, rm_index, rp02_index, elastic_slope, epsilon0,
    )

    effective_form = form_override if form_override is not None else classification.form_guess

    if effective_form == "true":
        hollomon = _hollomon_fit_core(
            strain[:rm_index + 1], stress[:rm_index + 1], elastic_slope, epsilon0, rp02_index,
        )
        # vstup uz je true - NEKONVERTUJEME znova, vratime priamo surove (orezane) data
        true_curve = TrueCurve(
            true_strain=strain[:rm_index + 1], true_stress=stress[:rm_index + 1],
            valid_up_to_index=rm_index,
        )
    else:
        hollomon = compute_hollomon_fit(converted_curve, elastic_slope, epsilon0, rp02_index, is_percent)
        true_curve = converted_curve

    return TrueCurveResult(
        true_curve=true_curve, hollomon=hollomon, classification=classification,
        form_used=effective_form, form_was_user_confirmed=(form_override is not None),
    )
