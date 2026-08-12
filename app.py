"""
PixelToProperty - STREAMLIT WEBOVA VERZIA digitalizacie krivky tahovej skusky.

Toto je webova nahrada za gui_app.py (tkinter desktop verzia) - POUZIVA
ROVNAKE moduly (digitization.py, engineering_properties.py, true_curve.py)
bez akejkolvek zmeny v ich logike. Jedina zmena je vo vrstve rozhrania:
tkinter okna -> Streamlit prvky (st.file_uploader, st.image, st.button...).

SPUSTENIE LOKALNE (na test pred nahratim na web):
    streamlit run app.py

NASADENIE NA WEB: viz STAV_HLADANIA_LINEARITY.md alebo predchadzajucu
diskusiu - v skratke: nahraj tento subor + digitization.py +
engineering_properties.py + true_curve.py + requirements.txt + packages.txt
do GitHub repozitara (verejneho alebo sukromneho), potom prepoj cez
share.streamlit.io.
"""

import matplotlib
matplotlib.use("Agg")  # bezhlavy backend - nutne na serveri bez displeja

import traceback

import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

import digitization as dig
import engineering_properties as eng
import true_curve as tc


def _resize_for_display(img_bgr: np.ndarray, max_dim: int = 700) -> np.ndarray:
    """Zmensi obrazok LEN PRE NAHLAD v prehliadaci (nie pre samotne spracovanie -
    to stale bezi na PLNOM rozliseni originalu, presnost sa tymto nemeni).
    DOLEZITY REALNY NALEZ pri nasadeni: 'use_container_width=True' obmedzuje
    LEN SIRKU zobrazenia - ak je zdrojovy obrazok VYSOKY (velke rozlisenie/
    vyskovy pomer strán), vysledny nahlad moze byt aj tak obrovsky (pouzivatel
    musi neprimerane vela scrollovat). Tu obmedzime VACSI z rozmerov (sirku
    alebo vysku) na `max_dim` pixelov, pomer strán ostava zachovany."""
    h, w = img_bgr.shape[:2]
    scale = max_dim / max(h, w)
    if scale >= 1.0:
        return img_bgr  # uz je dost maly, netreba zvacsovat
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


st.set_page_config(page_title="PixelToProperty", page_icon="📐", layout="wide")
st.title("📐 PixelToProperty")
st.caption(
    "Digitalizácia krivky ťahovej skúšky — nahraj obrázok grafu "
    "(screenshot/foto/PDF export) → automatická detekcia rámu a osí (OCR) → "
    "digitalizácia krivky → E, Rp0.2, Rm, A s diagnostikou spoľahlivosti."
)

# ----------------------------------------------------------------------------
# Session state - Streamlit pri kazdej interakcii spusti CELY skript odznova,
# takze medzivysledky (obrazok, ram, kalibracia, digitalizovana krivka...) sa
# musia ulozit do st.session_state, inak by sa strácali pri kazdom kliknuti.
# ----------------------------------------------------------------------------
for key in ["img_bgr", "frame_info", "calib", "strain", "stress", "color",
            "style", "props", "true_result"]:
    if key not in st.session_state:
        st.session_state[key] = None


def reset_downstream_state():
    """Pri novom obrazku/novom kroku zahodit vsetky nadvazujuce vysledky,
    aby GUI nezobrazovalo neaktualne cisla zo starsieho obrazka."""
    for key in ["frame_info", "calib", "strain", "stress", "color", "style",
                "props", "true_result"]:
        st.session_state[key] = None


# ----------------------------------------------------------------------------
# KROK 0: Nahranie obrazka
# ----------------------------------------------------------------------------
uploaded_file = st.file_uploader("Nahraj obrázok grafu", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert("RGB")
    img_rgb = np.array(pil_img)
    new_img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # ak je to iny obrazok nez predtym, zahodit stare vysledky
    if st.session_state.img_bgr is None or not np.array_equal(new_img_bgr, st.session_state.img_bgr):
        reset_downstream_state()
    st.session_state.img_bgr = new_img_bgr

# ----------------------------------------------------------------------------
# KROK 1: Detekcia ramu + OCR kalibracia osi
# ----------------------------------------------------------------------------
if st.session_state.img_bgr is not None:
    if st.button("1) Detegovať rám a kalibrovať osi", type="primary"):
        try:
            gray = cv2.cvtColor(st.session_state.img_bgr, cv2.COLOR_BGR2GRAY)
            frame = dig.detect_frame(gray)
            calib = dig.calibrate_axes(gray, frame)
            st.session_state.frame_info = frame
            st.session_state.calib = calib
            # zahod nadvazujuce vysledky - mozu byt z inej kalibracie
            st.session_state.strain = None
            st.session_state.props = None
            st.session_state.true_result = None
        except Exception as e:
            st.error(f"Chyba pri detekcii rámu/osí: {e}")
            with st.expander("Zobraziť technický detail (presný riadok chyby)"):
                st.code(traceback.format_exc())

    if st.session_state.frame_info is not None:
        frame = st.session_state.frame_info
        calib = st.session_state.calib

        vis = st.session_state.img_bgr.copy()
        cv2.rectangle(vis, (frame.x_left, frame.y_top), (frame.x_right, frame.y_bottom),
                      (0, 255, 0), 3)
        st.image(cv2.cvtColor(_resize_for_display(vis), cv2.COLOR_BGR2RGB),
                 caption="Detegovaný rám (zelený obdĺžnik)")

        n_x_out = int(np.sum(~calib.x_inliers))
        n_y_out = int(np.sum(~calib.y_inliers))
        st.info(
            f"OCR kalibrácia: X os {len(calib.x_ticks_raw)} kandidátov "
            f"({n_x_out} zahodených ako odľahlé), Y os {len(calib.y_ticks_raw)} kandidátov "
            f"({n_y_out} zahodených). Skontroluj, či rám sedí na graf skôr než budeš pokračovať."
        )
        if calib.x_is_percent_hint is not None:
            st.caption(f"Jednotka osi X rozpoznaná OCR-om: {'%' if calib.x_is_percent_hint else 'zlomok'}")

        # ------------------------------------------------------------------
        # KROK 2: Volba formy krivky VOPRED (na ziadost z predoslej diskusie -
        # pouzivatel zadava formu PRED digitalizaciou, nie az po nej)
        # ------------------------------------------------------------------
        form_choice = st.radio(
            "Forma vstupnej krivky (zadaj vopred, ak ju poznáš):",
            options=["engineering", "true"], index=0, horizontal=True,
        )

        if st.button("2) Digitalizovať krivku a vypočítať vlastnosti", type="primary"):
            try:
                pad = 2
                img_bgr = st.session_state.img_bgr
                crop = img_bgr[frame.y_top + pad:frame.y_bottom - pad,
                                frame.x_left + pad:frame.x_right - pad]
                crop_offset = (frame.x_left + pad, frame.y_top + pad)
                strain, stress, color, style = dig.digitize_curve(crop, calib, crop_offset)
                st.session_state.strain = strain
                st.session_state.stress = stress
                st.session_state.color = color
                st.session_state.style = style

                props = eng.compute_engineering_properties(
                    strain, stress, is_percent_hint=calib.x_is_percent_hint,
                )
                st.session_state.props = props

                rp02_idx = props.Rp02_index if props.Rp02_index is not None else 0
                true_result = tc.compute_true_curve_and_hollomon(
                    strain, stress, props.strain_unit_percent, props.Rm_index, rp02_idx,
                    props.elastic_slope, props.epsilon0, form_override=form_choice,
                )
                st.session_state.true_result = true_result
            except Exception as e:
                st.error(f"Chyba pri digitalizácii/výpočte: {e}")
                with st.expander("Zobraziť technický detail (presný riadok chyby)"):
                    st.code(traceback.format_exc())

# ----------------------------------------------------------------------------
# KROK 3: Zobrazenie vysledkov (Modul 2 - inzinierske vlastnosti)
# ----------------------------------------------------------------------------
if st.session_state.props is not None:
    props = st.session_state.props
    strain = st.session_state.strain
    stress = st.session_state.stress

    st.header("Výsledky — inžinierske vlastnosti")
    st.caption(f"Štýl krivky: {st.session_state.style} ({len(strain)} digitalizovaných bodov, "
               f"farba: {st.session_state.color})")

    # --- Diagnostika spolahlivosti (kriticka cast - NEVYNECHAVAT, viz navod) ---
    conf_display = {"HIGH": "🟢 HIGH", "MEDIUM": "🟠 MEDIUM", "LOW": "🔴 LOW"}
    st.subheader(f"Spoľahlivosť odhadu: {conf_display.get(props.confidence, props.confidence)}")
    for msg in props.confidence_messages:
        st.caption(f"• {msg}")

    if props.confidence == "LOW" and (props.E_GPa != props.E_GPa):  # NaN check
        st.warning(
            "Elastická oblasť sa nedá spoľahlivo určiť z tohto obrázka — pravdepodobne "
            "fundamentálny limit rozlíšenia zdroja (napr. digitalizácia stratila začiatok "
            "krivky, alebo je skutočná elastická deformácia príliš malá na danú škálu grafu). "
            "E, Rp0.2 a A sa v tomto prípade nezobrazujú — bolo by to zavádzajúce číslo."
        )
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("E", f"{props.E_GPa:.2f} GPa")
        col2.metric("Rp0.2", f"{props.Rp02_MPa:.1f} MPa" if props.Rp02_MPa is not None else "N/F")
        col3.metric("Rm", f"{props.Rm_MPa:.1f} MPa")
        col4.metric("A (korig.)", f"{props.A_percent:.2f} %")

        with st.expander("Podrobná diagnostika fitu"):
            st.write(f"- Počet bodov v pôvodnom elastickom okne: {props.n_window}")
            st.write(f"- R² (regresia cez počiatok, na jadre okna): {props.elastic_r2:.5f}")
            st.write(f"- Sm(rel) (ISO 6892-1, relatívna chyba sklonu): {props.sm_rel_percent:.3f} %")
            st.write(f"- reach95_ratio (0=okamžité plató, ~1=žiadne plató): {props.reach95_ratio:.3f}")
            st.write(f"- Rozsah napätia pokrytý oknom, ako podiel Rm: {props.stress_span_fraction:.3f}")
            if props.yield_ratio is not None:
                st.write(f"- Yield ratio (Rp0.2/Rm): {props.yield_ratio:.3f}")

        # --- Graf ---
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(strain, stress, "o", markersize=3, color="#d62728", alpha=0.5,
                label=f"digitalizované body ({len(strain)})")
        a, b = props.elastic_window
        ax.plot(strain[a:b], stress[a:b], "-", linewidth=3, color="blue",
                label=f"elastické jadro (n={b - a}, R²={props.elastic_r2:.4f})")
        if props.Rp02_MPa is not None:
            ax.plot(props.Rp02_strain, props.Rp02_MPa, "s", color="purple", markersize=9,
                    label=f"Rp0.2={props.Rp02_MPa:.1f} MPa")
        ax.plot(props.Rm_strain, props.Rm_MPa, "o", color="black", markersize=9,
                label=f"Rm={props.Rm_MPa:.1f} MPa")
        ax.set_xlabel("strain (%)" if props.strain_unit_percent else "strain")
        ax.set_ylabel("stress (MPa)")
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    # --- Stiahnutie digitalizovanych dat ---
    csv_data = "strain,stress_MPa\n" + "\n".join(f"{s:.6f},{t:.6f}" for s, t in zip(strain, stress))
    st.download_button("Stiahnuť digitalizované body (CSV)", csv_data,
                        file_name="digitalizovana_krivka.csv", mime="text/csv")

# ----------------------------------------------------------------------------
# KROK 4: Modul 3 - true krivka + Hollomon
# ----------------------------------------------------------------------------
if st.session_state.true_result is not None:
    tr = st.session_state.true_result
    props = st.session_state.props
    strain = st.session_state.strain
    stress = st.session_state.stress

    st.header("True krivka + Hollomonov fit")
    agree = "zhoduje sa s tvojou voľbou" if tr.classification.form_guess == tr.form_used else \
            "POZOR: algoritmus by tipoval inú formu — over si to"
    st.caption(f"Použitá forma: **{tr.form_used}** (automatický návrh algoritmu: "
               f"'{tr.classification.form_guess}', {agree})")

    holl = tr.hollomon
    col1, col2, col3 = st.columns(3)
    col1.metric("n (exponent spevnenia)", f"{holl.n:.4f}")
    col2.metric("K", f"{holl.K_MPa:.1f} MPa")
    col3.metric("R²", f"{holl.r2:.4f}")

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    true_c = tr.true_curve
    ax2.plot(true_c.true_strain, true_c.true_stress, "o", markersize=3, color="#d62728",
             alpha=0.6, label=f"true krivka (po Rm, {len(true_c.true_strain)} bodov)")
    eps_fit = np.linspace(max(holl.strain_range[0], 1e-6), holl.strain_range[1], 100)
    sigma_fit = holl.K_MPa * eps_fit ** holl.n
    eps_elastic_fit = sigma_fit / props.elastic_slope
    eps_fit_display = eps_fit + eps_elastic_fit + props.epsilon0
    ax2.plot(eps_fit_display, sigma_fit, "--", linewidth=2, color="black",
             label=f"Hollomon: n={holl.n:.3f}, K={holl.K_MPa:.0f}")
    ax2.set_xlabel("true strain")
    ax2.set_ylabel("true stress (MPa)")
    ax2.legend(fontsize=9, loc="lower right")
    ax2.grid(alpha=0.3)
    st.pyplot(fig2)

st.divider()
st.caption(
    "Poznámka: appka nič neukladá natrvalo — obrázok aj výsledky existujú len počas "
    "tejto relácie v prehliadači."
)
