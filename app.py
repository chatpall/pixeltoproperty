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
# UVITACIE/BETA OKNO - modalne, zobrazi sa raz pri prvom nacitani relacie
# (st.session_state.welcome_dismissed=False), da sa kedykolvek znova otvorit
# tlacidlom v bocnom paneli ("ℹ️ About & feedback"). Vysvetluje aplikaciu
# populariznou formou + ziada o testovanie/spatnu vazbu cez ResearchGate.
# POZNAMKA: vyzaduje Streamlit >= 1.31 (st.dialog) - requirements.txt necha
# streamlit nepripnuty na verziu, takze Streamlit Cloud vzdy nainstaluje
# najnovsiu dostupnu, kde st.dialog uz existuje.
# ============================================================================
if "welcome_dismissed" not in st.session_state:
    st.session_state.welcome_dismissed = False


@st.dialog("Welcome to PixelToProperty 👋")
def show_welcome_dialog():
    st.markdown(
        "**Ever needed the exact numbers behind a stress–strain curve from a "
        "paper, report, or old lab printout — but only had the picture?**\n\n"
        "PixelToProperty reads the chart image, automatically finds the axes, "
        "calibrates them (OCR does the reading for you), traces the curve "
        "pixel by pixel, and turns it back into real (strain, stress) data "
        "points.\n\n"
        "From there it goes further: it computes the material's key "
        "mechanical properties directly from the digitized curve — Young's "
        "modulus (E), the 0.2% offset yield strength (Rp0.2), ultimate "
        "tensile strength (Rm), elongation (A), and, if the data supports "
        "it, the Hollomon strain-hardening exponent (n). You get the "
        "numbers, a downloadable CSV of the raw points, and a diagnostic "
        "view of how confident the fit is — no more measuring by eye with "
        "a ruler and a squint."
    )
    st.markdown("---")
    st.markdown(
        "**🧪 This is a testing version.** Source charts vary wildly — "
        "different fonts, colors, gridlines, resolutions — and the app "
        "won't get everything right yet. I'd be genuinely grateful if you "
        "tried it on your own curves and let me know what worked, what "
        "didn't, and what surprised you.\n\n"
        "**Share feedback:** via my ResearchGate profile → "
        "[Peter Oslanec](https://www.researchgate.net/profile/Peter-Oslanec-2)"
    )
    st.write("")
    if st.button("Got it, let's start", type="primary", use_container_width=True):
        st.session_state.welcome_dismissed = True
        st.rerun()


if not st.session_state.welcome_dismissed:
    show_welcome_dialog()


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
    st.markdown('<div class="pp-caption">Tensile test digitization</div>',
                unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**STEPS**")

    step_labels = ["Upload image\nPNG, JPG, JPEG", "Detect frame\nand calibrate axes",
                   "Digitize curve\nand compute properties"]
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
        '<div class="pp-side-box"><b>ℹ️ How it works</b><br>'
        "The app automatically finds the chart frame, reads the axis labels "
        "(OCR), recognizes the curve, digitizes it, and computes mechanical "
        "properties plus a Hollomon fit.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="pp-side-box">💡 <b>Good to know</b><br>'
        "Nothing is stored permanently. Download the CSV with the raw points "
        "if you want to keep the results.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="pp-caption" style="margin-top:1.5rem;">PixelToProperty'
        '<br>Streamlit Community Cloud</div>',
        unsafe_allow_html=True,
    )
    if st.button("ℹ️ About & feedback", use_container_width=True):
        st.session_state.welcome_dismissed = False
        st.rerun()


# ============================================================================
# HLAVICKA
# ============================================================================
st.markdown("# PixelToProperty — tensile test chart digitization")
st.markdown(
    '<div class="pp-subtitle">Upload a chart image and get the material\'s '
    "mechanical properties.</div>",
    unsafe_allow_html=True,
)
st.write("")

# ============================================================================
# RIADOK 1: Krok 0 (nahranie) | Nahlad
# ============================================================================
col_upload, col_preview = st.columns(2)

with col_upload:
    with st.container(border=True):
        st.subheader("Step 0 — Upload chart image")
        uploaded_file = st.file_uploader("Upload chart image",
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
        st.subheader("Uploaded image preview")
        if st.session_state.img_bgr is not None:
            st.image(cv2.cvtColor(_resize_for_display(st.session_state.img_bgr),
                                   cv2.COLOR_BGR2RGB))
        else:
            st.caption("The image will appear here after upload.")

# ============================================================================
# RIADOK 2: Krok 1 (detekcia ramu) | Krok 2 (digitalizacia)
# ============================================================================
col_step1, col_step2 = st.columns(2)

with col_step1:
    with st.container(border=True):
        st.subheader("Step 1 — Detect frame and calibrate axes")
        disabled_1 = st.session_state.img_bgr is None
        if st.button("Detect frame and calibrate axes", type="primary",
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
                st.error(f"Error detecting frame/axes: {e}")
                with st.expander("Technical details"):
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
            r1.metric("X axis - candidates / used",
                      f"{len(calib.x_ticks_raw)} / {len(calib.x_ticks_raw) - n_x_out}")
            r2.metric("Y axis - candidates / used",
                      f"{len(calib.y_ticks_raw)} / {len(calib.y_ticks_raw) - n_y_out}")
            if calib.x_is_percent_hint is not None:
                st.caption(f"X axis unit (OCR): **{'%' if calib.x_is_percent_hint else 'fraction'}**")
            st.success("✓ Calibration successful. Check the green frame. If it doesn't match, try a different image.")

with col_step2:
    with st.container(border=True):
        st.subheader("Step 2 — Digitize curve and compute properties")
        disabled_2 = st.session_state.frame_info is None
        if disabled_2:
            st.caption("Finish Step 1 first.")

        form_choice = st.radio(
            "Input curve form (choose one)",
            options=["engineering", "true"],
            format_func=lambda v: "engineering (default)" if v == "engineering" else "true (true stress–strain)",
            index=0, horizontal=True, disabled=disabled_2,
        )

        if st.button("Digitize curve and compute properties", type="primary",
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
                st.success("✓ Digitization complete — results below.")
            except Exception as e:
                st.error(f"Error during digitization/computation: {e}")
                with st.expander("Technical details"):
                    st.code(traceback.format_exc())

# ============================================================================
# RIADOK 3: 3.1 Vysledky (inzinierske vlastnosti) | 3.2 True krivka + Hollomon
# ============================================================================
col_res1, col_res2 = st.columns(2)

with col_res1:
    with st.container(border=True):
        st.subheader("3.1 Results — engineering properties")
        if st.session_state.props is None:
            st.caption("Results will appear after Step 2.")
        else:
            props = st.session_state.props
            strain = st.session_state.strain
            stress = st.session_state.stress

            m1, m2, m3 = st.columns(3)
            m1.metric("Curve style", st.session_state.style)
            m2.metric("Number of points", f"{len(strain):,}".replace(",", " "))
            m3.metric("Curve color", st.session_state.color)

            conf_colors = {"HIGH": "#16A34A", "MEDIUM": "#EA580C", "LOW": "#DC2626"}
            conf_icons = {"HIGH": "🟢", "MEDIUM": "🟠", "LOW": "🔴"}
            c = props.confidence
            st.markdown(
                f'<div class="pp-confidence" style="color:{conf_colors.get(c, "#1E293B")};">'
                f"{conf_icons.get(c, '')} Estimate confidence: {c}</div>",
                unsafe_allow_html=True,
            )
            for msg in props.confidence_messages:
                st.caption(f"• {msg}")

            if props.confidence == "LOW" and (props.E_GPa != props.E_GPa):
                st.warning(
                    "The elastic region cannot be reliably determined from this "
                    "image — likely a fundamental resolution limit of the source. "
                    "E, Rp0.2 and A are not shown, as that would be a misleading number."
                )
            else:
                v1, v2, v3, v4 = st.columns(4)
                v1.metric("E", f"{props.E_GPa:.2f}", "GPa")
                v2.metric("Rp0.2", f"{props.Rp02_MPa:.1f}" if props.Rp02_MPa is not None else "N/F",
                          "MPa" if props.Rp02_MPa is not None else "")
                v3.metric("Rm", f"{props.Rm_MPa:.1f}", "MPa")
                v4.metric("A (corrected)", f"{props.A_percent:.2f}", "%")

                with st.expander("Detailed fit diagnostics"):
                    d1, d2, d3, d4, d5 = st.columns(5)
                    d1.metric("Points in window", props.n_window)
                    d2.metric("R² (through origin)", f"{props.elastic_r2:.3f}")
                    d3.metric("Sm(rel) ISO", f"{props.sm_rel_percent:.2f} %")
                    d4.metric("reach95", f"{props.reach95_ratio:.2f}")
                    d5.metric("Range coverage", f"{props.stress_span_fraction:.2f}")
                    if props.yield_ratio is not None:
                        st.caption(f"Yield ratio (Rp0.2/Rm): {props.yield_ratio:.3f}")

                fig, ax = plt.subplots(figsize=(6, 4.5))
                ax.plot(strain, stress, "o", markersize=2.5, color="#DC2626", alpha=0.5,
                        label="digitized points")
                a, b = props.elastic_window
                ax.plot(strain[a:b], stress[a:b], "-", linewidth=2.5, color="#2563EB",
                        label="elastic core (fit)")

                # Dashed reference elastic line (E) extended across the whole
                # chart - visually shows EXACTLY what slope was chosen for the
                # linear (elastic) part, not just the segment used for the fit itself.
                x_max_plot = max(strain.max(), props.Rm_strain) * 1.05
                x_ref = np.array([0.0, x_max_plot])
                y_ref = props.elastic_slope * x_ref
                ax.plot(x_ref, y_ref, "--", linewidth=1.3, color="#2563EB", alpha=0.55,
                        label="reference slope E (extended)")

                if props.Rp02_MPa is not None:
                    # Dashed OFFSET line (parallel to E, shifted by 0.2% strain) -
                    # exactly the one used to determine the conventional (offset)
                    # yield strength Rp0.2. Extended to the point where it
                    # intersects the curve (Rp0.2), to clearly show the geometric
                    # principle.
                    offset = 0.2 if props.strain_unit_percent else 0.002
                    x_offset_end = props.Rp02_strain * 1.15
                    x_offset_line = np.array([offset, x_offset_end])
                    y_offset_line = props.elastic_slope * (x_offset_line - offset)
                    ax.plot(x_offset_line, y_offset_line, "--", linewidth=1.5, color="#7C3AED",
                            alpha=0.7, label="0.2% offset (yield strength)")
                    ax.plot(props.Rp02_strain, props.Rp02_MPa, "s", color="#7C3AED",
                            markersize=8, label="Rp0.2")
                ax.plot(props.Rm_strain, props.Rm_MPa, "o", color="#1E293B",
                        markersize=8, label="Rm")
                ax.set_xlim(0, x_max_plot)
                ax.set_ylim(0, max(stress.max(), props.Rm_MPa) * 1.1)
                ax.set_xlabel("strain (%)" if props.strain_unit_percent else "strain")
                ax.set_ylabel("stress (MPa)")
                ax.legend(fontsize=7.5, loc="lower right")
                ax.grid(alpha=0.25)
                st.pyplot(fig, use_container_width=True)

            csv_data = "strain,stress_MPa\n" + "\n".join(
                f"{s:.6f},{t:.6f}" for s, t in zip(strain, stress)
            )
            st.download_button("⬇ Download CSV", csv_data,
                                file_name="digitized_curve.csv", mime="text/csv",
                                use_container_width=True)

with col_res2:
    with st.container(border=True):
        st.subheader("3.2 True curve + Hollomon fit")
        if st.session_state.true_result is None:
            st.caption("Results will appear after Step 2.")
        else:
            tr = st.session_state.true_result
            props = st.session_state.props

            agree = tr.classification.form_guess == tr.form_used
            g1, g2 = st.columns(2)
            g1.metric("Form used (specified)", tr.form_used)
            g2.metric("Algorithm suggests", tr.classification.form_guess)
            st.caption("Match ✓" if agree else "⚠️ NOTE: the algorithm would guess a different form — please verify")

            holl = tr.hollomon
            true_c = tr.true_curve

            if not holl.applicable:
                st.warning(f"Hollomon fit not available: {holl.message}")
                fig2, ax2 = plt.subplots(figsize=(6, 4.5))
                ax2.plot(true_c.true_strain, true_c.true_stress, "o", markersize=2.5,
                          color="#DC2626", alpha=0.6, label="true curve (up to Rm)")
                ax2.set_xlabel("true strain")
                ax2.set_ylabel("true stress (MPa)")
                ax2.legend(fontsize=8, loc="lower right")
                ax2.grid(alpha=0.25)
                st.pyplot(fig2, use_container_width=True)
            else:
                h1, h2, h3 = st.columns(3)
                h1.metric("n (hardening exponent)", f"{holl.n:.3f}")
                h2.metric("K (MPa)", f"{holl.K_MPa:.0f}")
                h3.metric("Fit R²", f"{holl.r2:.3f}")

                fig2, ax2 = plt.subplots(figsize=(6, 4.5))
                ax2.plot(true_c.true_strain, true_c.true_stress, "o", markersize=2.5,
                          color="#DC2626", alpha=0.6, label="True curve")
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

                st.info("ℹ️ Hollomon fit is available. The material has a pronounced plastic region.")

st.write("")
st.divider()
st.caption(
    "Note: this app does not store anything permanently — the image and results only "
    "exist for the duration of this browser session."
)
