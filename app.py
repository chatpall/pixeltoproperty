"""
PixelToProperty - STREAMLIT WEB VERSION of tensile-test curve digitization.

This file uses the SAME modules (digitization.py, engineering_properties.py,
true_curve.py) without any change to their logic - only the presentation
layer changes (layout, style, responsiveness), not the computation logic.

DESIGN SYSTEM (derived from the project user's reference design):
    Colors: #2563EB (blue accent), #F8FAFC (card background), #E2E8F0
            (borders), #1E293B (text), #64748B (muted text),
            #16A34A/#EA580C/#DC2626 (green/orange/red for confidence)
    Typography: clean sans-serif, clear size hierarchy
    Layout: sidebar (logo + step tracker + info) + main area
            with cards in a 2-column grid (st.container(border=True))
    Responsiveness: Streamlit columns AUTOMATICALLY stack vertically on
                     narrower screens (built-in behavior) - the extra CSS
                     media queries only fine-tune font size/spacing.

RUN LOCALLY:
    streamlit run app.py
"""

import matplotlib
matplotlib.use("Agg")  # headless backend - required on a server without a display

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
# CONTACT INFO (shown in the welcome dialog and sidebar)
# ============================================================================
CONTACT_EMAIL = "chatpall+pixeltoproperty@gmail.com"


# ============================================================================
# BASIC PAGE SETUP
# ============================================================================
st.set_page_config(page_title="PixelToProperty", page_icon="📐", layout="wide")


# ============================================================================
# CUSTOM CSS - cards, typography, step tracker, responsiveness
# ============================================================================
st.markdown("""
<style>
/* ---- Typography ---- */
h1 { font-weight: 800 !important; letter-spacing: -0.02em; }
h2, h3 { font-weight: 700 !important; }
.pp-subtitle { color: #64748B; font-size: 1.05rem; margin-top: -0.6rem; }
.pp-caption { color: #64748B; font-size: 0.85rem; }

/* ---- Cards (st.container(border=True)) - subtle rounding + shadow ---- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 14px !important;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06), 0 1px 2px rgba(15, 23, 42, 0.04);
}

/* ---- Larger, bolder metrics ---- */
div[data-testid="stMetricValue"] { font-size: 1.7rem; font-weight: 800; color: #1E293B; }
div[data-testid="stMetricLabel"] { color: #64748B; font-weight: 600; }

/* ---- Step tracker in the sidebar ---- */
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

/* ---- Info/tip boxes in the sidebar ---- */
.pp-side-box {
    background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px;
    padding: 0.8rem 0.9rem; font-size: 0.85rem; color: #475569; margin-top: 1rem;
}
.pp-side-box b { color: #1E293B; }

/* ---- Confidence badge ---- */
.pp-confidence { font-weight: 800; font-size: 1.15rem; }

/* ---- Responsiveness: smaller fonts/spacing on narrower screens ----
   Streamlit columns (st.columns) already stack vertically AUTOMATICALLY
   below ~640px width (built-in behavior) - these media queries only
   fine-tune typography so it isn't oversized on mobile/tablet. */
@media (max-width: 768px) {
    h1 { font-size: 1.6rem !important; }
    .pp-subtitle { font-size: 0.9rem; }
    div[data-testid="stMetricValue"] { font-size: 1.3rem; }
}
</style>
""", unsafe_allow_html=True)


def _resize_for_display(img_bgr: np.ndarray, max_dim: int = 700) -> np.ndarray:
    """Shrinks the image ONLY FOR the browser preview (not for the actual
    processing - that still runs at the ORIGINAL's FULL resolution, accuracy
    is unaffected)."""
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
            "style", "props", "true_result", "welcome_seen"]:
    if key not in st.session_state:
        st.session_state[key] = None


def reset_downstream_state():
    for key in ["frame_info", "calib", "strain", "stress", "color", "style",
                "props", "true_result"]:
        st.session_state[key] = None


# ============================================================================
# WELCOME DIALOG - short guide + contact, shown once before first use
# ============================================================================
@st.dialog("Welcome to PixelToProperty")
def _welcome_dialog():
    st.markdown(
        "**PixelToProperty** turns a chart image of a tensile test "
        "(stress–strain curve) into digitized data points and mechanical "
        "properties (E, Rp0.2, Rm, elongation, Hollomon fit)."
    )
    st.markdown("**How to use it:**")
    st.markdown(
        "1. **Upload** an image of the chart (PNG/JPG).\n"
        "2. Click **Detect frame & calibrate axes** and check that the "
        "green rectangle matches the chart's plot area.\n"
        "3. Click **Digitize curve & compute properties** to get the "
        "results and download the raw data as CSV."
    )
    st.markdown(
        "For best results, use a clear, high-resolution image with "
        "sharp, readable axis labels."
    )
    st.info(
        "Nothing is stored permanently - the image and results exist only "
        "for the current browser session. Download the CSV if you want to "
        "keep the results."
    )
    st.caption(f"Questions or feedback? Contact: {CONTACT_EMAIL}")
    if st.button("Get started", type="primary", use_container_width=True):
        st.session_state.welcome_seen = True
        st.rerun()


if not st.session_state.welcome_seen:
    _welcome_dialog()


# ============================================================================
# SIDEBAR: logo, step tracker, info boxes
# ============================================================================
def _step_status(step_index: int) -> str:
    """Returns 'done'/'current'/'todo' for a step based on session_state."""
    completed = [
        st.session_state.img_bgr is not None,
        st.session_state.frame_info is not None,
        st.session_state.props is not None,
    ]
    if completed[step_index]:
        return "done"
    # current = first incomplete step, for which all previous steps are done
    if step_index == 0 or completed[step_index - 1]:
        return "current"
    return "todo"


with st.sidebar:
    st.markdown("### 📐 PixelToProperty")
    st.markdown('<div class="pp-caption">Tensile test digitization</div>',
                unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**PROGRESS**")

    step_labels = ["Upload image\nPNG, JPG, JPEG", "Detect frame\n& calibrate axes",
                   "Digitize curve\n& compute properties"]
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
        "The app automatically finds the chart frame, reads the axis "
        "labels (OCR), detects the curve, digitizes it, and computes "
        "mechanical properties plus the Hollomon fit.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="pp-side-box">💡 <b>Good to know</b><br>'
        "Nothing is stored permanently. Download the raw-points CSV if "
        "you want to keep the results.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="pp-side-box">✉️ <b>Questions?</b><br>'
        f'{CONTACT_EMAIL}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="pp-caption" style="margin-top:1.5rem;">PixelToProperty'
        '<br>Streamlit Community Cloud</div>',
        unsafe_allow_html=True,
    )


# ============================================================================
# HEADER
# ============================================================================
st.markdown("# PixelToProperty — tensile test chart digitization")
st.markdown(
    '<div class="pp-subtitle">Upload a chart image and get the material\'s '
    "mechanical properties.</div>",
    unsafe_allow_html=True,
)
st.write("")

# ============================================================================
# ROW 1: Step 0 (upload) | Preview
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
# ROW 2: Step 1 (frame detection) | Step 2 (digitization)
# ============================================================================
col_step1, col_step2 = st.columns(2)

with col_step1:
    with st.container(border=True):
        st.subheader("Step 1 — Detect frame & calibrate axes")
        disabled_1 = st.session_state.img_bgr is None
        if st.button("Detect frame & calibrate axes", type="primary",
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
                with st.expander("Technical detail"):
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
        st.subheader("Step 2 — Digitize curve & compute properties")
        disabled_2 = st.session_state.frame_info is None
        if disabled_2:
            st.caption("Complete Step 1 first.")

        form_choice = st.radio(
            "Input curve form (choose one)",
            options=["engineering", "true"],
            format_func=lambda v: "engineering (default)" if v == "engineering" else "true (true stress–strain)",
            index=0, horizontal=True, disabled=disabled_2,
        )

        if st.button("Digitize curve & compute properties", type="primary",
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
                with st.expander("Technical detail"):
                    st.code(traceback.format_exc())

# ============================================================================
# ROW 3: 3.1 Results (engineering properties) | 3.2 True curve + Hollomon
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
            m2.metric("Point count", f"{len(strain):,}".replace(",", " "))
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
                    "image — likely a fundamental limit of the source resolution. "
                    "E, Rp0.2 and A are not shown, as that would be a misleading number."
                )
            else:
                v1, v2, v3, v4 = st.columns(4)
                v1.metric("E", f"{props.E_GPa:.2f}", "GPa")
                v2.metric("Rp0.2", f"{props.Rp02_MPa:.1f}" if props.Rp02_MPa is not None else "N/F",
                          "MPa" if props.Rp02_MPa is not None else "")
                v3.metric("Rm", f"{props.Rm_MPa:.1f}", "MPa")
                v4.metric("A (corr.)", f"{props.A_percent:.2f}", "%")

                with st.expander("Detailed fit diagnostics"):
                    d1, d2, d3, d4, d5 = st.columns(5)
                    d1.metric("Points in window", props.n_window)
                    d2.metric("R² (through 0)", f"{props.elastic_r2:.3f}")
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
                # chart - visually shows the EXACT slope chosen for the linear
                # (elastic) part, not just the segment used for the actual fit.
                x_max_plot = max(strain.max(), props.Rm_strain) * 1.05
                x_ref = np.array([0.0, x_max_plot])
                y_ref = props.elastic_slope * x_ref
                ax.plot(x_ref, y_ref, "--", linewidth=1.3, color="#2563EB", alpha=0.55,
                        label="reference slope E (extended)")

                if props.Rp02_MPa is not None:
                    # Dashed OFFSET line (parallel to E, shifted by 0.2% strain)
                    # - the exact line used to determine the conventional yield
                    # strength Rp0.2. Extended to the point where it crosses the
                    # curve (Rp0.2), so the geometric principle is clearly visible.
                    offset = 0.2 if props.strain_unit_percent else 0.002
                    x_offset_end = props.Rp02_strain * 1.15
                    x_offset_line = np.array([offset, x_offset_end])
                    y_offset_line = props.elastic_slope * (x_offset_line - offset)
                    ax.plot(x_offset_line, y_offset_line, "--", linewidth=1.5, color="#7C3AED",
                            alpha=0.7, label="0.2% offset (conventional yield strength)")
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
            g1.metric("Form used (selected)", tr.form_used)
            g2.metric("Algorithm suggests", tr.classification.form_guess)
            st.caption("They match ✓" if agree else "⚠️ NOTE: the algorithm would guess a different form — double-check this")

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
    "Note: the app doesn't store anything permanently — the image and results "
    "only exist for the current browser session."
)
