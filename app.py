"""
PixelToProperty - STREAMLIT WEBOVA VERZIA digitalizacie krivky tahovej skusky.

Tento subor pouziva ROVNAKE moduly (digitization.py, engineering_properties.py,
true_curve.py) bez akejkolvek zmeny v ich logike - meni sa LEN prezentacna
vrstva (rozlozenie, styl, responzivita), nie vypoctova logika.

DIZAJNOVY SYSTEM (odvodeny z referencneho navrhu pouzivatela projektu):
    Farby:  #2563EB (modra akcentova), #F8FAFC (pozadie kariet), #E2E8F0
            (okraje), #1E293B (text), #64748B (tlmeny text),
            #16A34A/#EA580C/#DC2626 (zelena/oranzova/cervena pre spolahlivost)
    Typografia: cisty sans-serif, jasna hierarchia velkosti
    Rozlozenie: bocny panel (logo + krokovnik + info) + hlavna plocha
                s kartami v 2-stlpcovej mriezke (st.container(border=True))
    Responzivita: Streamlit stlpce sa AUTOMATICKY skladaju vertikalne na
                  uzsich obrazovkach (vstavane spravanie) - doplnkove CSS
                  media queries len doladujeme velkost pisma/odsadenia.

SPUSTENIE LOKALNE:
    streamlit run app.py
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


# ============================================================================
# ZAKLADNE NASTAVENIE STRANKY
# ============================================================================
st.set_page_config(page_title="PixelToProperty", page_icon="📐", layout="wide")


# ============================================================================
# CUSTOM CSS - karty, typografia, krokovnik, responzivita
# ============================================================================
st.markdown("""
<style>
/* ---- Typografia ---- */
h1 { font-weight: 800 !important; letter-spacing: -0.02em; }
h2, h3 { font-weight: 700 !important; }
.pp-subtitle { color: #64748B; font-size: 1.05rem; margin-top: -0.6rem; }
.pp-caption { color: #64748B; font-size: 0.85rem; }

/* ---- Karty (st.container(border=True)) - jemne zaoblenie + tien ---- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 14px !important;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06), 0 1px 2px rgba(15, 23, 42, 0.04);
}

/* ---- Metriky vacsie a vyraznejsie ---- */
div[data-testid="stMetricValue"] { font-size: 1.7rem; font-weight: 800; color: #1E293B; }
div[data-testid="stMetricLabel"] { color: #64748B; font-weight: 600; }

/* ---- Krokovnik v bocnom paneli ---- */
.pp-step { display: flex; align-items: center; gap: 0.6rem; padding: 0.35rem 0; }
.pp-step-badge {
    width: 24px; height: 24px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.8rem; font-weight: 700; flex-shrink: 0;
}
.pp-step-done .pp-step-badge { background: #16A34A; color: white; }
.pp-step-current .pp-step-badge { background: #2563EB; color: white; }
.pp-step-todo .pp-step-badge { background: #E2E8F0; color: #64748B; }
.pp-step-done .pp-step-label { color: #1E293B; }
.pp-step-current .pp-step-label { color: #1E293B; font-weight: 700; }
.pp-step-todo .pp-step-label { color: #94A3B8; }
.pp-step-label { font-size: 0.9rem; line-height: 1.2; }

/* ---- Info/tip bloky v bocnom paneli ---- */
.pp-side-box {
    background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px;
    padding: 0.8rem 0.9rem; font-size: 0.85rem; color: #475569; margin-top: 1rem;
}
.pp-side-box b { color: #1E293B; }

/* ---- Spolahlivost badge ---- */
.pp-confidence { font-weight: 800; font-size: 1.15rem; }

/* ---- Responzivita: mensie pisma/odsadenia na uzsich obrazovkach ----
   Streamlit stlpce (st.columns) sa uz AUTOMATICKY skladaju vertikalne pod
   ~640px sirky (vstavane spravanie) - tieto media queries len doladuju
   typografiu, aby to na mobile/tablete nebolo neprimerane velke. */
@media (max-width: 768px) {
    h1 { font-size: 1.6rem !important; }
    .pp-subtitle { font-size: 0.9rem; }
    div[data-testid="stMetricValue"] { font-size: 1.3rem; }
}
</style>
""", unsafe_allow_html=True)


def _resize_for_display(img_bgr: np.ndarray, max_dim: int = 700) -> np.ndarray:
    """Zmensi obrazok LEN PRE NAHLAD v prehliadaci (nie pre samotne spracovanie -
    to stale bezi na PLNOM rozliseni originalu, presnost sa tymto nemeni)."""
    h, w = img_bgr.shape[:2]
    scale = max_dim / max(h, w)
    if scale >= 1.0:
        return img_bgr
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ============================================================================
# SESSION STATE
# ============================================================================
for key in ["img_bgr", "frame_info", "calib", "strain", "stress", "color",
            "style", "props", "true_result"]:
    if key not in st.session_state:
        st.session_state[key] = None


def reset_downstream_state():
    for key in ["frame_info", "calib", "strain", "stress", "color", "style",
                "props", "true_result"]:
        st.session_state[key] = None


# ============================================================================
# BOCNY PANEL: logo, krokovnik, info boxy
# ============================================================================
def _step_status(step_index: int) -> str:
    """Vrati 'done'/'current'/'todo' pre krok podla stavu session_state."""
    completed = [
        st.session_state.img_bgr is not None,
        st.session_state.frame_info is not None,
        st.session_state.props is not None,
    ]
    if completed[step_index]:
        return "done"
    # current = prvy nedokonceny krok, pre ktory su predchadzajuce hotove
    if step_index == 0 or completed[step_index - 1]:
        return "current"
    return "todo"


with st.sidebar:
    st.markdown("### 📐 PixelToProperty")
    st.markdown('<div class="pp-caption">Digitalizácia ťahových skúšok</div>',
                unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**POSTUP**")

    step_labels = ["Nahrať obrázok\nPNG, JPG, JPEG", "Detegovať rám\na kalibrovať osi",
                   "Digitalizovať krivku\na vypočítať vlastnosti"]
    step_icons = {"done": "✓", "current": "●", "todo": "○"}
    for i, label in enumerate(step_labels):
        status = _step_status(i)
        label_html = label.replace("\n", "<br>")
        st.markdown(
            f'<div class="pp-step pp-step-{status}">'
            f'<div class="pp-step-badge">{step_icons[status]}</div>'
            f'<div class="pp-step-label">{label_html}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="pp-side-box"><b>ℹ️ Ako to funguje</b><br>'
        "Aplikácia automaticky nájde rám grafu, prečíta popisky osí (OCR), "
        "rozpozná krivku, digitalizuje ju a vypočíta mechanické vlastnosti "
        "aj Hollomonov fit.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="pp-side-box">💡 <b>Dobré vedieť</b><br>'
        "Nič sa neukladá natrvalo. Stiahni si CSV so surovými bodmi, ak "
        "chceš výsledky uchovať.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="pp-caption" style="margin-top:1.5rem;">PixelToProperty'
        '<br>Streamlit Community Cloud</div>',
        unsafe_allow_html=True,
    )


# ============================================================================
# HLAVICKA
# ============================================================================
st.markdown("# PixelToProperty — digitalizácia grafov ťahových skúšok")
st.markdown(
    '<div class="pp-subtitle">Nahraj obrázok grafu a získaj mechanické '
    "vlastnosti materiálu.</div>",
    unsafe_allow_html=True,
)
st.write("")

# ============================================================================
# RIADOK 1: Krok 0 (nahranie) | Nahlad
# ============================================================================
col_upload, col_preview = st.columns(2)

with col_upload:
    with st.container(border=True):
        st.subheader("Krok 0 — Nahrať obrázok grafu")
        uploaded_file = st.file_uploader("Nahraj obrázok grafu",
                                          type=["png", "jpg", "jpeg"],
                                          label_visibility="collapsed")
        if uploaded_file is not None:
            pil_img = Image.open(uploaded_file).convert("RGB")
            img_rgb = np.array(pil_img)
            new_img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            if st.session_state.img_bgr is None or not np.array_equal(
                new_img_bgr, st.session_state.img_bgr
            ):
                reset_downstream_state()
            st.session_state.img_bgr = new_img_bgr
            st.success(f"✓ {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")

with col_preview:
    with st.container(border=True):
        st.subheader("Náhľad nahraného obrázka")
        if st.session_state.img_bgr is not None:
            st.image(cv2.cvtColor(_resize_for_display(st.session_state.img_bgr),
                                   cv2.COLOR_BGR2RGB))
        else:
            st.caption("Obrázok sa zobrazí tu po nahraní.")

# ============================================================================
# RIADOK 2: Krok 1 (detekcia ramu) | Krok 2 (digitalizacia)
# ============================================================================
col_step1, col_step2 = st.columns(2)

with col_step1:
    with st.container(border=True):
        st.subheader("Krok 1 — Detegovať rám a kalibrovať osi")
        disabled_1 = st.session_state.img_bgr is None
        if st.button("Detegovať rám a kalibrovať osi", type="primary",
                      disabled=disabled_1, use_container_width=True):
            try:
                gray = cv2.cvtColor(st.session_state.img_bgr, cv2.COLOR_BGR2GRAY)
                frame = dig.detect_frame(gray)
                calib = dig.calibrate_axes(gray, frame)
                st.session_state.frame_info = frame
                st.session_state.calib = calib
                st.session_state.strain = None
                st.session_state.props = None
                st.session_state.true_result = None
            except Exception as e:
                st.error(f"Chyba pri detekcii rámu/osí: {e}")
                with st.expander("Technický detail"):
                    st.code(traceback.format_exc())

        if st.session_state.frame_info is not None:
            frame = st.session_state.frame_info
            calib = st.session_state.calib
            vis = st.session_state.img_bgr.copy()
            cv2.rectangle(vis, (frame.x_left, frame.y_top), (frame.x_right, frame.y_bottom),
                          (0, 255, 0), 3)
            st.image(cv2.cvtColor(_resize_for_display(vis, max_dim=500), cv2.COLOR_BGR2RGB))

            n_x_out = int(np.sum(~calib.x_inliers))
            n_y_out = int(np.sum(~calib.y_inliers))
            r1, r2 = st.columns(2)
            r1.metric("X os - kandidátov / použitých",
                      f"{len(calib.x_ticks_raw)} / {len(calib.x_ticks_raw) - n_x_out}")
            r2.metric("Y os - kandidátov / použitých",
                      f"{len(calib.y_ticks_raw)} / {len(calib.y_ticks_raw) - n_y_out}")
            if calib.x_is_percent_hint is not None:
                st.caption(f"Jednotka osi X (OCR): **{'%' if calib.x_is_percent_hint else 'zlomok'}**")
            st.success("✓ Kalibrácia úspešná. Skontroluj zelený rám. Ak nesedí, skús iný obrázok.")

with col_step2:
    with st.container(border=True):
        st.subheader("Krok 2 — Digitalizovať krivku a vypočítať vlastnosti")
        disabled_2 = st.session_state.frame_info is None
        if disabled_2:
            st.caption("Najprv dokonči Krok 1.")

        form_choice = st.radio(
            "Forma vstupnej krivky (vyber jednu možnosť)",
            options=["engineering", "true"],
            format_func=lambda v: "engineering (predvolené)" if v == "engineering" else "true (true stress–strain)",
            index=0, horizontal=True, disabled=disabled_2,
        )

        if st.button("Digitalizovať krivku a vypočítať vlastnosti", type="primary",
                      disabled=disabled_2, use_container_width=True):
            try:
                frame = st.session_state.frame_info
                calib = st.session_state.calib
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
                st.success("✓ Digitalizácia dokončená — výsledky nižšie.")
            except Exception as e:
                st.error(f"Chyba pri digitalizácii/výpočte: {e}")
                with st.expander("Technický detail"):
                    st.code(traceback.format_exc())

# ============================================================================
# RIADOK 3: 3.1 Vysledky (inzinierske vlastnosti) | 3.2 True krivka + Hollomon
# ============================================================================
col_res1, col_res2 = st.columns(2)

with col_res1:
    with st.container(border=True):
        st.subheader("3.1 Výsledky — inžinierske vlastnosti")
        if st.session_state.props is None:
            st.caption("Výsledky sa zobrazia po Kroku 2.")
        else:
            props = st.session_state.props
            strain = st.session_state.strain
            stress = st.session_state.stress

            m1, m2, m3 = st.columns(3)
            m1.metric("Štýl krivky", st.session_state.style)
            m2.metric("Počet bodov", f"{len(strain):,}".replace(",", " "))
            m3.metric("Farba krivky", st.session_state.color)

            conf_colors = {"HIGH": "#16A34A", "MEDIUM": "#EA580C", "LOW": "#DC2626"}
            conf_icons = {"HIGH": "🟢", "MEDIUM": "🟠", "LOW": "🔴"}
            c = props.confidence
            st.markdown(
                f'<div class="pp-confidence" style="color:{conf_colors.get(c, "#1E293B")};">'
                f"{conf_icons.get(c, '')} Spoľahlivosť odhadu: {c}</div>",
                unsafe_allow_html=True,
            )
            for msg in props.confidence_messages:
                st.caption(f"• {msg}")

            if props.confidence == "LOW" and (props.E_GPa != props.E_GPa):
                st.warning(
                    "Elastická oblasť sa nedá spoľahlivo určiť z tohto obrázka — "
                    "pravdepodobne fundamentálny limit rozlíšenia zdroja. E, Rp0.2 "
                    "a A sa nezobrazujú, bolo by to zavádzajúce číslo."
                )
            else:
                v1, v2, v3, v4 = st.columns(4)
                v1.metric("E", f"{props.E_GPa:.2f}", "GPa")
                v2.metric("Rp0.2", f"{props.Rp02_MPa:.1f}" if props.Rp02_MPa is not None else "N/F",
                          "MPa" if props.Rp02_MPa is not None else "")
                v3.metric("Rm", f"{props.Rm_MPa:.1f}", "MPa")
                v4.metric("A (korig.)", f"{props.A_percent:.2f}", "%")

                with st.expander("Podrobná diagnostika fitu"):
                    d1, d2, d3, d4, d5 = st.columns(5)
                    d1.metric("Body v okne", props.n_window)
                    d2.metric("R² (cez 0)", f"{props.elastic_r2:.3f}")
                    d3.metric("Sm(rel) ISO", f"{props.sm_rel_percent:.2f} %")
                    d4.metric("reach95", f"{props.reach95_ratio:.2f}")
                    d5.metric("Pokrytie rozsahu", f"{props.stress_span_fraction:.2f}")
                    if props.yield_ratio is not None:
                        st.caption(f"Yield ratio (Rp0.2/Rm): {props.yield_ratio:.3f}")

                fig, ax = plt.subplots(figsize=(6, 4.5))
                ax.plot(strain, stress, "o", markersize=2.5, color="#DC2626", alpha=0.5,
                        label="digitalizované body")
                a, b = props.elastic_window
                ax.plot(strain[a:b], stress[a:b], "-", linewidth=2.5, color="#2563EB",
                        label="elastické jadro (fit)")
                if props.Rp02_MPa is not None:
                    ax.plot(props.Rp02_strain, props.Rp02_MPa, "s", color="#7C3AED",
                            markersize=8, label="Rp0.2")
                ax.plot(props.Rm_strain, props.Rm_MPa, "o", color="#1E293B",
                        markersize=8, label="Rm")
                ax.set_xlabel("strain (%)" if props.strain_unit_percent else "strain")
                ax.set_ylabel("stress (MPa)")
                ax.legend(fontsize=8, loc="lower right")
                ax.grid(alpha=0.25)
                st.pyplot(fig, use_container_width=True)

            csv_data = "strain,stress_MPa\n" + "\n".join(
                f"{s:.6f},{t:.6f}" for s, t in zip(strain, stress)
            )
            st.download_button("⬇ Stiahnuť CSV", csv_data,
                                file_name="digitalizovana_krivka.csv", mime="text/csv",
                                use_container_width=True)

with col_res2:
    with st.container(border=True):
        st.subheader("3.2 True krivka + Hollomonov fit")
        if st.session_state.true_result is None:
            st.caption("Výsledky sa zobrazia po Kroku 2.")
        else:
            tr = st.session_state.true_result
            props = st.session_state.props

            agree = tr.classification.form_guess == tr.form_used
            g1, g2 = st.columns(2)
            g1.metric("Použitá forma (zadaná)", tr.form_used)
            g2.metric("Algoritmus navrhuje", tr.classification.form_guess)
            st.caption("Zhodujú sa ✓" if agree else "⚠️ POZOR: algoritmus by tipoval inú formu — over si to")

            holl = tr.hollomon
            true_c = tr.true_curve

            if not holl.applicable:
                st.warning(f"Hollomonov fit nie je dostupný: {holl.message}")
                fig2, ax2 = plt.subplots(figsize=(6, 4.5))
                ax2.plot(true_c.true_strain, true_c.true_stress, "o", markersize=2.5,
                          color="#DC2626", alpha=0.6, label="true krivka (po Rm)")
                ax2.set_xlabel("true strain")
                ax2.set_ylabel("true stress (MPa)")
                ax2.legend(fontsize=8, loc="lower right")
                ax2.grid(alpha=0.25)
                st.pyplot(fig2, use_container_width=True)
            else:
                h1, h2, h3 = st.columns(3)
                h1.metric("n (exponent spevnenia)", f"{holl.n:.3f}")
                h2.metric("K (MPa)", f"{holl.K_MPa:.0f}")
                h3.metric("R² fitu", f"{holl.r2:.3f}")

                fig2, ax2 = plt.subplots(figsize=(6, 4.5))
                ax2.plot(true_c.true_strain, true_c.true_stress, "o", markersize=2.5,
                          color="#DC2626", alpha=0.6, label="True krivka")
                eps_fit = np.linspace(max(holl.strain_range[0], 1e-6), holl.strain_range[1], 100)
                sigma_fit = holl.K_MPa * eps_fit ** holl.n
                eps_elastic_fit = sigma_fit / props.elastic_slope
                eps_fit_display = eps_fit + eps_elastic_fit + props.epsilon0
                ax2.plot(eps_fit_display, sigma_fit, "-", linewidth=2, color="#2563EB",
                          label="Hollomon fit")
                ax2.set_xlabel("true strain (-)")
                ax2.set_ylabel("true stress (MPa)")
                ax2.legend(fontsize=8, loc="lower right")
                ax2.grid(alpha=0.25)
                st.pyplot(fig2, use_container_width=True)

                st.info("ℹ️ Hollomonov fit je dostupný. Materiál má výraznú plastickú oblasť.")

st.write("")
st.divider()
st.caption(
    "Poznámka: appka nič neukladá natrvalo — obrázok aj výsledky existujú len počas "
    "tejto relácie v prehliadači."
)
