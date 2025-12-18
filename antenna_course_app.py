import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import math


# ===========================
# Core computation functions
# ===========================

def compute_dipole_pattern(L_lambda=0.5, num_points=720):
    """
    Simple thin dipole pattern model.
    L_lambda = length in wavelengths (e.g. 0.5 for a half-wave dipole).
    Returns theta (rad) and normalized |E(θ)|.
    """
    theta = np.linspace(1e-3, np.pi - 1e-3, num_points)

    # Classical thin-dipole approximation:
    # E(θ) ∝ [cos(kL/2 cosθ) - cos(kL/2)] / sinθ
    kL_over_2 = np.pi * L_lambda / 2.0
    numerator = np.cos(kL_over_2 * np.cos(theta)) - np.cos(kL_over_2)
    denominator = np.sin(theta)
    E = numerator / denominator

    E_mag = np.abs(E)
    E_mag /= np.max(E_mag)  # normalize
    return theta, E_mag


def mom_charged_rod(length=1.0, N=20):
    """
    Very simple electrostatic Method of Moments toy problem:

    - Rod of length 'length' along z-axis, from -L/2 to +L/2.
    - Discretized into N segments.
    - Influence kernel ~ 1/|z_i - z_j| (crude).
    - Constant applied potential -> solve Z q = V.

    Returns segment centers (z) and charge density q.
    """
    z = np.linspace(-length / 2.0, length / 2.0, N + 1)
    centers = 0.5 * (z[:-1] + z[1:])
    dz = z[1] - z[0]

    Z = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            if i == j:
                # Crude self-term approximation to avoid singularity
                Z[i, j] = 2.0 / dz
            else:
                dist = abs(centers[i] - centers[j])
                Z[i, j] = 1.0 / dist

    # Constant potential (1 V) at all segments
    V = np.ones(N)
    q = np.linalg.solve(Z, V)

    return centers, q


def mom_wire_antenna(L_lambda=0.5, N=21, freq=1e9):
    """
    Very simplified thin-wire MoM demo:

    - Straight wire along z-axis, total length L_lambda * λ.
    - Center-fed (delta-gap) at the middle segment.
    - Kernel ~ exp(-jkR)/R (free-space Green's function).
    - Solve for segment currents I from Z I = V.
    - Compute a far-field pattern via a discrete array-factor-style sum.

    Returns:
        z_centers: segment centers (m)
        I: complex current on each segment
        theta: angles for far-field (rad)
        F_mag: normalized pattern magnitude
    """
    c0 = 3e8
    wavelength = c0 / freq
    k = 2.0 * np.pi / wavelength

    L = L_lambda * wavelength
    z = np.linspace(-L / 2.0, L / 2.0, N + 1)
    centers = 0.5 * (z[:-1] + z[1:])
    dz = z[1] - z[0]

    Z = np.zeros((N, N), dtype=complex)
    radius = 0.001 * L  # small radius just to regularize self term

    for i in range(N):
        for j in range(N):
            if i == j:
                R = radius
            else:
                R = abs(centers[i] - centers[j])
            Z[i, j] = np.exp(-1j * k * R) / R

    # Delta-gap excitation at the center segment
    V = np.zeros(N, dtype=complex)
    mid = N // 2
    V[mid] = 1.0  # 1 V excitation

    I = np.linalg.solve(Z, V)

    # Far-field pattern
    theta = np.linspace(1e-3, np.pi - 1e-3, 720)
    F = np.zeros_like(theta, dtype=complex)
    for n in range(N):
        F += I[n] * np.exp(1j * k * centers[n] * np.cos(theta))

    F_mag = np.abs(F)
    F_mag /= np.max(F_mag)

    return centers, I, theta, F_mag


# ===========================
# Streamlit UI helper
# ===========================

def plot_fig(fig):
    """Small helper to display a Matplotlib figure in Streamlit."""
    st.pyplot(fig)


# ===========================
# Module UIs
# ===========================

def module_radiation_wire():
    """
    Table 9.3 + Fig 9.17 style module:
    - Polar overlay plot of |F(θ)|^2 (Eq. 9.49), shown as "dB down" rings (0/10/20/30)
    - Dropdown selects the Table 9.3 row (total-length label)
    - ONE table-style current-distribution icon (single-sided silhouette + hatch), shown on the LEFT
    - Auto HPBW computed from Eq. 9.49 and compared to Table 9.3
    """


    # -------------------------
    # Presets (Table 9.3)
    # -------------------------
    presets = [
        {"name": "2ℓ << λ",   "ell_over_lambda": 0.02,  "hpbw_deg": 90.0},
        {"name": "2ℓ = λ/4",  "ell_over_lambda": 1/8,   "hpbw_deg": 87.0},
        {"name": "2ℓ = λ/2",  "ell_over_lambda": 1/4,   "hpbw_deg": 78.0},
        {"name": "2ℓ = 3λ/4", "ell_over_lambda": 3/8,   "hpbw_deg": 64.0},
        {"name": "2ℓ = λ",    "ell_over_lambda": 1/2,   "hpbw_deg": 47.8},
    ]

    # -------------------------
    # Eq. 9.49 helpers
    # -------------------------
    def F_theta(theta, ell_over_lambda):
        """
        F(θ) = [cos(β0 ℓ cosθ) - cos(β0 ℓ)] / sinθ
        where β0 = 2π/λ, and β0ℓ = 2π(ℓ/λ).
        """
        theta = np.clip(theta, 1e-9, np.pi - 1e-9)
        beta0_ell = 2.0 * np.pi * ell_over_lambda
        return (np.cos(beta0_ell * np.cos(theta)) - np.cos(beta0_ell)) / np.sin(theta)

    def power_pattern(theta, ell_over_lambda):
        P = np.abs(F_theta(theta, ell_over_lambda)) ** 2
        P /= np.max(P) if np.max(P) > 0 else 1.0
        return P

    def pattern_dbdown_fullcircle(ell_over_lambda, n=1440, floor_db=30):
        """
        Full 0..2π polar curve with symmetry about the dipole axis.
        We compute power on 0..π and mirror it.
        r is drawn so outer ring corresponds to 0 dB down.
        """
        theta_full = np.linspace(0, 2 * np.pi, n)
        theta_equiv = np.where(theta_full <= np.pi, theta_full, 2 * np.pi - theta_full)

        P = power_pattern(theta_equiv, ell_over_lambda)
        P_db = 10.0 * np.log10(np.maximum(P, 1e-15))
        P_db = np.clip(P_db, -floor_db, 0.0)

        dbdown = -P_db            # 0..floor_db
        r = floor_db - dbdown     # floor_db..0 (outer larger)
        return theta_full, r

    def compute_hpbw_deg(ell_over_lambda, n=20001):
        """
        HPBW from normalized power pattern: width between P=0.5 crossings around main lobe.
        Intended for Table 9.3 range (single major lobe).
        """
        theta = np.linspace(1e-6, np.pi - 1e-6, n)
        P = power_pattern(theta, ell_over_lambda)

        imax = int(np.argmax(P))

        left = None
        for i in range(imax, 0, -1):
            if P[i] >= 0.5 and P[i - 1] < 0.5:
                left = i
                break

        right = None
        for i in range(imax, len(P) - 1):
            if P[i] >= 0.5 and P[i + 1] < 0.5:
                right = i
                break

        if left is None or right is None:
            return None

        def interp(i_hi, i_lo):
            t1, t2 = theta[i_lo], theta[i_hi]
            p1, p2 = P[i_lo], P[i_hi]
            if p2 == p1:
                return t1
            return t1 + (0.5 - p1) * (t2 - t1) / (p2 - p1)

        th_left = interp(left, left - 1)
        th_right = interp(right, right + 1)
        return float(np.degrees(th_right - th_left))

    # -------------------------
    # Table icon (current distribution)
    # -------------------------
    def current_profile(ell_over_lambda, z_norm):
        """
        Envelope for the TABLE ICON (visual reference).
        - very short: approximately triangular
        - otherwise: sinusoidal standing-wave envelope
        """
        if ell_over_lambda < 0.06:
            I = 1.0 - np.abs(z_norm)
        else:
            beta0_ell = 2.0 * np.pi * ell_over_lambda
            I = np.sin(np.maximum(beta0_ell * (1.0 - np.abs(z_norm)), 0.0))

        I = np.abs(I)
        I /= np.max(I) if np.max(I) > 0 else 1.0

        # soften slightly so it looks more like the printed table sketches
        I = I ** 0.85
        return I

    def draw_table_current_icon(ax, ell_over_lambda, phase=0.0, title=None):
        """
        Draw like the textbook table:
        - dipole axis (vertical)
        - ONE-SIDED silhouette to the RIGHT (not a symmetric leaf)
        - light hatch lines
        - small arrow
        """
        z = np.linspace(-1.0, 1.0, 700)
        I = current_profile(ell_over_lambda, z)

        breathe = 0.9 + 0.1 * np.cos(phase)
        w = 0.55 * I * breathe

        # axis
        ax.plot([0, 0], [-1, 1], linewidth=2)

        # silhouette (right side)
        ax.plot(w, z, linewidth=2)
        ax.fill_betweenx(z, 0, w, alpha=0.18)

        # hatching
        z_stripes = np.linspace(-0.95, 0.95, 18)
        for zs in z_stripes:
            ws = float(np.interp(zs, z, w))
            ax.plot([0.02, ws], [zs, zs], linewidth=1, alpha=0.6)

        # arrow (simple)
        ax.plot([-0.14, -0.14], [0.40, 0.78], linewidth=2)
        ax.plot([-0.16, -0.14, -0.12], [0.70, 0.78, 0.70], linewidth=2)

        ax.set_xlim(-0.30, 0.75)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        if title:
            ax.set_title(title, fontsize=12, pad=6)

    # -------------------------
    # UI
    # -------------------------
    st.header("Radiation Pattern (Eq. 9.49) + Table 9.3 Reference")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("Select Table 9.3 row")

        names = [p["name"] for p in presets]
        choice = st.selectbox("Total length (legend label)", names, index=2)
        chosen = next(p for p in presets if p["name"] == choice)

        overlay_all = st.checkbox("Overlay all presets (Fig. 9.17 style)", value=True)

        # lock to the book style
        floor_db = 30

        st.markdown("---")
        st.subheader("HPBW check")
        st.write(f"Half-length ℓ/λ: **{chosen['ell_over_lambda']}**")
        st.write(f"Table 9.3 HPBW: **{chosen['hpbw_deg']}°**")

        computed = compute_hpbw_deg(chosen["ell_over_lambda"])
        if computed is None:
            st.warning("Computed HPBW not found (crossings missing).")
        else:
            err_pct = 100.0 * (computed - chosen["hpbw_deg"]) / chosen["hpbw_deg"]
            st.write(f"Computed HPBW: **{computed:.2f}°**")
            st.write(f"Percent error: **{err_pct:+.2f}%**")

        st.markdown("---")
        st.subheader("Current distribution (table reference icon)")
        animate = st.checkbox("Animate icon (optional)", value=False)
        phase = 0.0
        if animate:
            phase = st.slider("Animation phase", 0.0, float(2 * np.pi), 0.0, step=0.1)

        fig_icon, ax_icon = plt.subplots(figsize=(3.4, 3.4))
        draw_table_current_icon(
            ax_icon,
            ell_over_lambda=chosen["ell_over_lambda"],
            phase=phase,
            title=f"{chosen['name']}   (ℓ/λ = {chosen['ell_over_lambda']})"
        )
        st.pyplot(fig_icon, use_container_width=True)

    with col_right:
        st.subheader("Power pattern (polar, dB down)")

        fig = plt.figure(figsize=(7.6, 7.6))
        ax = fig.add_subplot(111, projection="polar")

        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

        tick_degs = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
        tick_labels = ["0°","30°","60°","90°","120°","150°","180°","150°","120°","90°","60°","30°"]
        ax.set_thetagrids(tick_degs, tick_labels)

        ax.set_rlim(0, floor_db)
        rings = [floor_db, floor_db - 10, floor_db - 20, floor_db - 30]
        rings = [r for r in rings if r >= 0]
        ax.set_yticks(rings)
        ax.set_yticklabels([str(int(floor_db - r)) for r in rings])  # 0,10,20,30 dB down
        ax.set_ylabel("Relative power (dB down)", labelpad=25)

        to_plot = presets if overlay_all else [chosen]
        for p in to_plot:
            th, r = pattern_dbdown_fullcircle(p["ell_over_lambda"], floor_db=floor_db)
            ax.plot(th, r, linewidth=2, label=p["name"])

        ax.set_title("Power pattern |F(θ)|² from Eq. 9.49 (Table 9.3 presets)", pad=18)
        ax.legend(loc="center left", bbox_to_anchor=(1.15, 0.5), frameon=False)

        try:
            plot_fig(fig)  # if your app defines plot_fig()
        except NameError:
            st.pyplot(fig)


    # =========================
    # Table-style current icon
    # =========================
def draw_table_current_icon(ax, ell_over_lambda, phase=0.0, title=None):
    # Make sure current_profile() exists above this function
    z = np.linspace(-1.0, 1.0, 600)
    I = current_profile(ell_over_lambda, z)

    # optional gentle “breathing”
    breathe = 0.9 + 0.1*np.cos(phase)
    width = 0.50 * I * breathe

    # Dipole axis
    ax.plot([0, 0], [-1, 1], linewidth=2)

    # One-sided outline (RIGHT side only)
    ax.plot(width, z, linewidth=2)

    # Fill from axis to outline
    ax.fill_betweenx(z, 0, width, alpha=0.18)

    # Hatch stripes (horizontal)
    z_stripes = np.linspace(-0.95, 0.95, 18)
    for zs in z_stripes:
        w = float(np.interp(zs, z, width))
        ax.plot([0.02, w], [zs, zs], linewidth=1, alpha=0.6)

    # Simple arrow (no annotate arrowprops)
    ax.plot([-0.12, -0.12], [0.40, 0.75], linewidth=2)
    ax.plot([-0.14, -0.12, -0.10], [0.68, 0.75, 0.68], linewidth=2)

    ax.set_xlim(-0.25, 0.65)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=12, pad=4)



    # =========================
    # Layout
    # =========================
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("Inputs")

        names = [p["name"] for p in presets]
        choice = st.selectbox("Select total length (matches Table 9.3 / Fig. 9.17 legend)", names, index=2)
        chosen = next(p for p in presets if p["name"] == choice)

        overlay_all = st.checkbox("Overlay all Table 9.3 patterns (like the book figure)", value=True)

        # If you want it to match the book exactly, keep this fixed at 30:
        floor_db = 30

        st.markdown("---")
        st.subheader("Table 9.3 values")
        st.write(f"Half-length ℓ/λ: **{chosen['ell_over_lambda']}**")
        st.write(f"Table HPBW: **{chosen['hpbw_deg']}°**")

        computed = compute_hpbw_deg(chosen["ell_over_lambda"])
        if computed is not None:
            err_pct = 100.0 * (computed - chosen["hpbw_deg"]) / chosen["hpbw_deg"]
            st.write(f"Computed HPBW: **{computed:.2f}°**")
            st.write(f"Percent error: **{err_pct:+.2f}%**")
        else:
            st.warning("Could not compute HPBW crossings for this selection.")

        st.markdown("---")
        st.subheader("Current distribution (Table icon)")
        animate = st.checkbox("Animate icon (optional)", value=False)
        phase = 0.0
        if animate:
            phase = st.slider("Animation phase", 0.0, float(2*np.pi), 0.0, step=0.1)

        fig_icon, ax_icon = plt.subplots(figsize=(3.2, 3.2))
        draw_table_current_icon(
            ax_icon,
            ell_over_lambda=chosen["ell_over_lambda"],
            phase=phase,
            title=f"{chosen['name']}   (ℓ/λ = {chosen['ell_over_lambda']})"
        )
        st.pyplot(fig_icon, use_container_width=True)

    with col_right:
        st.subheader("Power pattern (polar, dB down)")

        fig = plt.figure(figsize=(7.5, 7.5))
        ax = fig.add_subplot(111, projection="polar")

        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

        # Book-style symmetric angle labels
        tick_degs = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
        tick_labels = ["0°","30°","60°","90°","120°","150°","180°","150°","120°","90°","60°","30°"]
        ax.set_thetagrids(tick_degs, tick_labels)

        # Fix at 30 dB down to match the figure
        ax.set_rlim(0, floor_db)
        rings = [floor_db, floor_db-10, floor_db-20, floor_db-30]
        rings = [r for r in rings if r >= 0]
        ax.set_yticks(rings)
        ax.set_yticklabels([str(int(floor_db - r)) for r in rings])  # 0,10,20,30 dB down
        ax.set_ylabel("Relative power (dB down)", labelpad=25)

        to_plot = presets if overlay_all else [chosen]
        for p in to_plot:
            th, r = pattern_dbdown_fullcircle(p["ell_over_lambda"], floor_db=floor_db)
            ax.plot(th, r, linewidth=2, label=p["name"])

        ax.set_title("Power pattern |F(θ)|² from Eq. 9.49 (Table 9.3 presets)", pad=18)
        ax.legend(loc="center left", bbox_to_anchor=(1.15, 0.5), frameon=False)

        # Use your helper if you have it; else direct
        try:
            plot_fig(fig)
        except NameError:
            st.pyplot(fig)





def module_mom_charged_rod():
\

    st.header("Method of Moments: Charged Rod (Conductor at Constant Potential)")

    st.markdown(
        """
This module solves for the **charge distribution** on a thin cylindrical conductor (radius **a**, length **L**) held at a
constant potential **V0**, using a simple **Method of Moments** (collocation).

**Goal:** reproduce the classic “U-shaped” charge distribution with end effects (like the textbook Figure 4.28).
"""
    )

    col_controls, col_plot = st.columns([1, 2])

    with col_controls:
        st.subheader("Inputs")

        L = st.number_input("Rod length L (m)", min_value=0.01, value=1.00, step=0.10, format="%.3f")
        a = st.number_input("Rod radius a (m)", min_value=1e-6, value=1e-3, step=1e-4, format="%.6f")
        V0 = st.number_input("Applied potential V0 (V)", min_value=0.01, value=1.00, step=0.10, format="%.3f")

        eps_r = st.number_input("Relative permittivity εr of surrounding medium", min_value=1.0, value=1.0, step=0.1, format="%.2f")
        N = st.slider("Number of segments N", min_value=5, max_value=200, value=20, step=1)

        st.markdown("---")
        st.subheader("Plot options")
        use_0_to_L = st.checkbox("x-axis as 0 → L (match textbook)", value=True)
        show_units_pCpm = st.checkbox("Show charge density as pC/m (×1e12)", value=True)
        show_points = st.checkbox("Show segment centers (markers)", value=True)

        st.markdown("---")
        st.subheader("Model notes")
        st.write("• Collocation at segment centers.")
        st.write("• Off-diagonal kernel uses 1/sqrt((Δz)^2 + a^2).")
        st.write("• Self-term uses a finite log form (avoids singularity).")

    # -------------------------
    # MoM solve
    # -------------------------
    eps0 = 8.854187817e-12
    eps = eps0 * eps_r

    dz = L / N
    # segment centers: z in [0, L] or centered later
    zc = (np.arange(N) + 0.5) * dz

    # Build Z matrix such that V = (1/(4πϵ)) * Z * λ
    # Here λ is line charge density (C/m) assumed constant on each segment.
    Z = np.zeros((N, N), dtype=float)

    # Off-diagonal: approximate integral by midpoint (λ_j * dz / R_ij)
    # R_ij = sqrt((z_i - z_j)^2 + a^2)
    for i in range(N):
        for j in range(N):
            if i != j:
                Rij = np.sqrt((zc[i] - zc[j]) ** 2 + a ** 2)
                Z[i, j] = dz / Rij
            else:
                # Self-term: integral over segment length of 1/sqrt((z-z')^2 + a^2)
                # Approx closed form for segment of half-length h = dz/2:
                # ∫_{-h}^{h} 1/sqrt(u^2 + a^2) du = 2 * asinh(h/a)
                # which equals 2*ln((h + sqrt(h^2 + a^2))/a)
                h = dz / 2.0
                Z[i, j] = 2.0 * np.arcsinh(h / a)

    # Solve: V0 = (1/(4πϵ)) * Z * λ  =>  Z*λ = 4πϵ V0 * 1
    b = (4.0 * np.pi * eps * V0) * np.ones(N)
    lam = np.linalg.solve(Z, b)  # C/m

    # -------------------------
    # Format x-axis and y-axis like textbook
    # -------------------------
    if use_0_to_L:
        x = zc
        x_label = "Length (m)"
    else:
        x = zc - L / 2.0
        x_label = "z (m)"

    if show_units_pCpm:
        y = lam * 1e12
        y_label = "Charge density (pC/m)"
    else:
        y = lam
        y_label = "Charge density (C/m)"

    # -------------------------
    # Plot
    # -------------------------
    fig, ax = plt.subplots(figsize=(8, 4.8))
    if show_points:
        ax.plot(x, y, marker="o")
    else:
        ax.plot(x, y)

    ax.set_title(f"MoM Charge Distribution (L = {L:.2f} m, a = {a:.2e} m, N = {N})")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True)

    # If you have plot_fig helper, use it; otherwise st.pyplot
    try:
        plot_fig(fig)
    except NameError:
        st.pyplot(fig)

    # Quick “textbook check” hint
    st.markdown("### Textbook check")
    st.write("Try: **L = 1 m**, **a = 1 mm**, **V0 = 1 V**, **N = 20**, **εr = 1** — you should see the classic U-shape with end spikes.")



def module_mom_wire_antenna():
    st.header("Method of Moments: Thin Wire Antenna")

    col_controls, col_plots = st.columns([1, 2])

    with col_controls:
        st.subheader("Inputs")

        L_lambda = st.slider(
            "Wire length L (in wavelengths λ)",
            min_value=0.2,
            max_value=2.0,
            value=0.5,
            step=0.05
        )
        N = st.slider(
            "Number of segments N",
            min_value=11,
            max_value=101,
            value=21,
            step=2,
            help="Use odd N so the feed is at the exact center segment."
        )
        freq_GHz = st.number_input(
            "Frequency (GHz)",
            min_value=0.1,
            max_value=20.0,
            value=1.0,
            step=0.1
        )
        freq = freq_GHz * 1e9

        st.markdown(
            """
            **What this model does:**
            - Builds a Z-matrix using a simple exp(-jkR)/R kernel.  
            - Applies a **delta-gap feed** at the center segment.  
            - Solves **Z I = V** for the current distribution.  
            - Uses the resulting currents to build a far-field pattern.
            """
        )

    with col_plots:
        zc, I, theta, F_mag = mom_wire_antenna(
            L_lambda=L_lambda, N=N, freq=freq
        )

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7))
        fig.tight_layout(pad=3.0)

        # Current magnitude
        ax1.plot(zc, np.abs(I), marker="o")
        ax1.set_xlabel("z (m)")
        ax1.set_ylabel("|I(z)| (arb. A)")
        ax1.set_title("MoM Current Magnitude Along the Wire")
        ax1.grid(True)

        # Far-field pattern
        ax2.plot(np.degrees(theta), F_mag)
        ax2.set_xlabel("θ (degrees)")
        ax2.set_ylabel("|F(θ)| (normalized)")
        ax2.set_title("Computed Far-Field Pattern (Broadside cut)")
        ax2.grid(True)

        plot_fig(fig)


def module_notes_theory():
    st.header("Notes & Theory – ECE474 Antennas Recap")

    st.markdown(
        """
        This tab is meant for explanation purposes for the computer programs.  
        You can scroll through and use these as reference points.

        ---
        """
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Dipole / Wire Radiation", "Charged Rod MoM", "Wire Antenna MoM", "Explanation"]
    )

    with tab1:
        st.subheader("Dipole / Wire Radiation Pattern")

        st.markdown(
            """
            **Key ideas:**

            - A thin straight wire or dipole has a radiation pattern that depends on its **electrical length** \(L/\\lambda\).  
            - For a **half-wave dipole** \((L \\approx 0.5\\lambda)\), the classic pattern has:
              - Maximum radiation broadside to the wire.
              - Nulls along the axis of the wire.
            - As the length increases \(L > \\lambda\), **additional lobes** appear.

            **Model used in this app:**

            We approximate the field as:

            """
        )
        st.latex(
            r"E(\theta) \propto \frac{\cos\!\big(\tfrac{kL}{2}\cos\theta\big) - \cos\!\big(\tfrac{kL}{2}\big)}{\sin\theta}"
        )
        st.markdown(
            """
            where:

            - \(k = 2\\pi / \\lambda\) is the wavenumber  
            - \(L\) is the physical length of the wire  
            - \\(\\theta\\) is the angle measured from the axis of the wire  

            In the GUI, the plot is **normalized**, so you mainly see the **shape** of the pattern.
            """
        )

    with tab2:
        st.subheader("Charged Rod – Method of Moments (MoM) Model")

        st.markdown(
            """
            **Conceptual goal:**

            - Show how **Method of Moments** turns a continuous problem into a **matrix equation**.  
            - Here, we treat a charged conducting rod held at a constant potential.

            **Steps:**

            1. Divide the rod into **N short segments** (basis functions).  
            2. Assume each segment has an **unknown charge** \(q_n\).  
            3. Write the potential at segment \(i\) as a sum of contributions from all segments:
               
               """
        )
        st.latex(
            r"V_i = \sum_{j=1}^N Z_{ij} q_j"
        )
        st.markdown(
            """
            4. For our model, we use a simple kernel:

               """
        )
        st.latex(
            r"Z_{ij} \approx \frac{1}{|z_i - z_j|} \quad (i \neq j)"
        )
        st.markdown(
            """
            and a special **self-term** when \(i = j\) to avoid a singularity.  
            5. We set the potential at every segment to **1 V**, so:

               """
        )
        st.latex(
            r"Z \mathbf{q} = \mathbf{V}"
        )
        st.markdown(
            """
            and solve for the vector of charges \(\\mathbf{q}\).

            **What to say in the exam:**

            > “I used a simple 1/|zᵢ−zⱼ| kernel as an example of how MoM works.
            > The important part is setting up Z and solving Zq = V, not having a perfect
            > physical model.”
            """
        )

    with tab3:
        st.subheader("Wire Antenna – Method of Moments")

        st.markdown(
            """
            **Physical problem:**

            - Thin straight wire antenna along the z-axis.
            - Center-fed by a voltage source (delta-gap excitation).
            - Surrounded by free space.

            **MoM steps here:**

            1. Divide the wire into **N segments** and assume an unknown current \(I_n\) on each.  
            2. Use a simple free-space Green’s function:

               """
        )
        st.latex(
            r"Z_{ij} \approx \frac{e^{-jkR_{ij}}}{R_{ij}}"
        )
        st.markdown(
            """
            where \(R_{ij}\) is the distance between segment centers \(i\) and \(j\).

            3. Apply a **delta-gap voltage** \(V\) at the center segment, set all other entries
               of the excitation vector to zero.  
            4. Solve:

               """
        )
        st.latex(
            r"Z \mathbf{I} = \mathbf{V}"
        )
        st.markdown(
            """
            to obtain the complex current distribution \(\\mathbf{I}\).

            5. Compute the far-field pattern using a discrete sum over segments:

               """
        )
        st.latex(
            r"F(\theta) = \sum_{n=1}^N I_n e^{j k z_n \cos\theta}"
        )
        st.markdown(
            """
            and then plot \|F(θ)\|, normalized.

            **How this ties to class:**

            - This is the same **MoM philosophy** used in more advanced antenna codes.  
            - Even though the kernel is simplified, the structure ZI = V and the
              idea of basis/testing functions are the same.
            """
        )

    with tab4:
        st.subheader("My overall explanation")

        st.markdown(
            """

            **1. Overall app purpose**

            > “I built a small antenna learning module in Python/Streamlit that bundles
            > our main computational assignments:
            > thin-wire radiation, a charged-rod MoM toy problem, and a thin-wire
            > antenna MoM solver. Each module lets me adjust parameters and see how
            > the physics responds in real time.”

            **2. Radiation tab**

            > “The first tab visualizes the normalized radiation pattern of a straight
            > wire/dipole as a function of its electrical length L/λ. It uses a standard
            > thin-dipole formula for E(θ), and I can show how increasing L creates
            > additional lobes.”

            **3. Charged rod MoM tab**

            > “The second tab is a Method of Moments example for a charged rod.
            > I discretize the rod into N segments, approximate the influence between
            > segments with a 1/|zᵢ−zⱼ| kernel, assemble the Z-matrix, and solve Zq = V
            > for the charge distribution. It’s a simple but clear illustration of how
            > MoM turns an integral equation into a matrix equation.”

            **4. Wire antenna MoM tab**

            > “The third tab uses a thin-wire MoM formulation. I build a complex Z-matrix
            > using an exp(-jkR)/R kernel, apply a delta-gap feed in the center, solve for
            > the segment currents, and then compute the far-field pattern as a sum of
            > contributions from each segment. This connects nicely to the idea of
            > current distribution and array factor from class.”

            **5. Tie back to course work**

            > “Together, these modules pull together the theory from Homework 1–3,
            > the MoM programming assignments, and our discussion of radiation patterns
            > and current distributions. The GUI makes it easy to demo these concepts
            > during the final.”
            """
        )

def module_design_spirals():


    c = 299_792_458.0

    st.header("Design Projects: Spiral Antennas (from class handout)")

    st.markdown(
        """
This page implements the **two spiral design assignments** using the equations and sizing rules shown in the professor handout:
- **Equiangular (log) spiral:**  r = r0 * exp(a * φ)  (angle-based frequency independent idea)
- **Archimedean spiral:** two-arm form and frequency limits using inner/outer diameter rules
"""
    )

    colL, colR = st.columns([1, 2])

    with colL:
        st.subheader("Choose design")
        design_type = st.selectbox(
            "Design type",
            ["Equiangular Spiral", "Archimedean Spiral"],
            index=0
        )

        st.subheader("Frequency band (spec)")
        fL_MHz = st.number_input("fL (MHz)", min_value=1.0, value=500.0, step=10.0)
        fH_MHz = st.number_input("fH (MHz)", min_value=1.0, value=3000.0, step=10.0)

        fL = fL_MHz * 1e6
        fH = fH_MHz * 1e6
        if fH <= fL:
            st.error("Need fH > fL")
            return

        st.markdown("---")
        st.subheader("Bandwidth")
        Br = fH / fL
        fc = 0.5 * (fH + fL)
        Bpct = (fH - fL) / fc * 100.0
        st.write(f"Ratio bandwidth  fH/fL = **{Br:.2f} : 1**")
        st.write(f"Percent bandwidth = **{Bpct:.1f}%** (about fc = {fc/1e9:.3f} GHz)")

        st.markdown("---")
        st.subheader("Geometry controls")
        turns = st.slider("Turns", 0.5, 10.0, 4.0, step=0.5)
        pts_per_turn = st.slider("Points per turn", 200, 2000, 900, step=100)
        two_arm = st.checkbox("Two-arm spiral", value=True)

        if design_type == "Equiangular Spiral":
            st.markdown("Handout equation: **r = r0 · e^(aφ)**")
            r0 = st.number_input("r0 (m) (inner radius seed)", min_value=1e-6, value=0.002, step=0.001, format="%.6f")
            a = st.number_input("a (growth rate, 1/rad)", min_value=0.01, value=0.20, step=0.01, format="%.3f")

            phi_max = 2*np.pi*turns
            r_out = r0 * np.exp(a * phi_max)

            st.markdown("---")
            st.subheader("Quick size numbers (geometry-based)")
            st.write(f"Outer radius from chosen parameters: **r_out ≈ {r_out:.4f} m**")
            # optional “rule-of-thumb” using circumference ~ λ
            r_out_rule = (c/fL) / (2*np.pi)
            r_in_rule = (c/fH) / (2*np.pi)
            st.write(f"Rule-of-thumb from band edges (2πr≈λ):")
            st.write(f"• r_out ≈ λL/(2π) = **{r_out_rule:.4f} m**")
            st.write(f"• r_in  ≈ λH/(2π) = **{r_in_rule:.4f} m**")

        else:
            st.markdown("Handout equation: **r = bφ** and **r = b(φ − π)** (two-arm)")
            b = st.number_input("b (m/rad)", min_value=1e-6, value=0.003, step=0.0005, format="%.6f")
            r_inner_offset = st.number_input("Inner radius offset (m)", min_value=0.0, value=0.0, step=0.001, format="%.4f")

            lamL = c/fL
            lamH = c/fH

            # From handout frequency limits:
            # inner diameter ≈ λH/2  => r_in ≈ λH/4
            # outer diameter ≈ λL/2  => r_out ≈ λL/4
            r_in_req = lamH / 4.0
            r_out_req = lamL / 4.0

            st.markdown("---")
            st.subheader("Frequency-limit sizing (matches handout)")
            st.write(f"λL = {lamL:.4f} m,  λH = {lamH:.4f} m")
            st.write(f"Inner diameter ≈ λH/2  ⇒ r_in ≈ λH/4 = **{r_in_req:.4f} m**")
            st.write(f"Outer diameter ≈ λL/2  ⇒ r_out ≈ λL/4 = **{r_out_req:.4f} m**")

            # choose φ span so that r(φ_max) ≈ r_out_req (rough)
            phi_max = 2*np.pi*turns
            r_out_geom = r_inner_offset + b * phi_max
            st.write(f"(With your b/turns) outer radius ≈ **{r_out_geom:.4f} m**")

            spacing = 2*np.pi*b
            st.write(f"Arm spacing (Archimedean): s = 2πb ≈ **{spacing:.4f} m**")

    with colR:
        st.subheader("Geometry plot (top view)")

        n = int(max(200, pts_per_turn * turns))
        phi = np.linspace(0.0, 2*np.pi*turns, n)

        if design_type == "Equiangular Spiral":
            # r = r0 e^(a φ)
            r = r0 * np.exp(a * phi)
        else:
            # r = r_offset + b φ
            r = r_inner_offset + b * phi

        x = r * np.cos(phi)
        y = r * np.sin(phi)

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(x, y, linewidth=2)

        if two_arm:
            ax.plot(-x, -y, linewidth=2)

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.grid(True)

        title = "Equiangular Spiral" if design_type == "Equiangular Spiral" else "Archimedean Spiral"
        ax.set_title(f"{title} (two-arm = {two_arm})")

        try:
            plot_fig(fig)
        except NameError:
            st.pyplot(fig)

    st.markdown("---")
    st.subheader("Key Takeaways/ Important things to note")
    if design_type == "Equiangular Spiral":
        st.write(
            "• The equiangular spiral is frequency independent because the geometry is angle-based.\n"
            "• I used r = r0 e^(aφ) and varied r0, a, and turns to cover the band.\n"
            "• Bandwidth is reported as fH/fL and percent BW."
        )
    else:
        st.write(
            "• The Archimedean spiral uses r = bφ (and the second arm shifted by π).\n"
            "• The handout gives frequency limits: inner diameter ~ λH/2 and outer diameter ~ λL/2.\n"
            "• I used those to compute required inner/outer radii and picked b and turns to meet the size."
        )

def module_homeworks():

    st.header("Homeworks 1–3 (worked + explained + plots)")

    # ============================================================
    # Helper functions (kept INSIDE this module so nothing else needed)
    # ============================================================

    def db10(x):
        return 10.0 * np.log10(np.maximum(x, 1e-15))

    # ---------- HW1 #1: Hertzian dipole fields ----------
    def hertzian_fields_theta90(I0, dl, f, r):
        """
        Fields of a short dipole (current element) at θ = 90° in free space.
        Returns magnitudes of Eθ and Hφ (phasor magnitudes).
        Uses standard expressions including 1/r, 1/r^2, 1/r^3 terms.
        """
        c = 299_792_458.0
        lam = c / f
        k = 2.0 * np.pi / lam
        eta0 = 377.0

        # Common factor (magnitude) ignoring e^{-jkr} because |e^{-jkr}|=1
        A = (I0 * dl) / (4.0 * np.pi)

        # Complex bracket terms:
        # Eθ = j η A [ k/r + 1/(j r^2) - 1/(k r^3) ]
        # Hφ = j A [ k/r + 1/(j r^2) ]
        term_far = k / r
        term_ind = 1.0 / (1j * r**2)     # = -j / r^2
        term_es  = -1.0 / (k * r**3)     # real negative

        E = 1j * eta0 * A * (term_far + term_ind + term_es)
        H = 1j * A * (term_far + term_ind)

        return abs(E), abs(H), k

    # ---------- HW1 #3: Pattern integrals ----------
    def hw1_prob3_results():
        # U = r^2 P = K sinθ cosφ, over θ∈[0,π], φ∈[-π/2,π/2]
        # HPBW: set U = K/2.
        # Az plane at θ=π/2: cosφ=1/2 -> φ=±60° => HPBW=120°
        # El plane at φ=0: sinθ=1/2 -> θ=30°,150° => HPBW=120°
        # FNBW: az null at cosφ=0 => ±90° => 180° ; el null at sinθ=0 => 0°,180° => 180°
        # Prad = ∫∫ U dΩ = ∫∫ K sinθ cosφ (sinθ dθ dφ) = K * (∫cosφ dφ)* (∫sin^2θ dθ)
        # = K*(2)*(π/2)=Kπ
        # D=4πUmax/Prad = 4πK/(Kπ)=4
        return {
            "HPBW_az_deg": 120.0,
            "HPBW_el_deg": 120.0,
            "FNBW_az_deg": 180.0,
            "FNBW_el_deg": 180.0,
            "Directivity": 4.0,
            "Directivity_dBi": 10.0 * math.log10(4.0)
        }

    # ---------- HW1 #5: Thin-wire dipole pattern factor ----------
    def dipole_F_theta(theta, k_ell):
        """
        F(θ) = [cos(kℓ cosθ) - cos(kℓ)] / sinθ
        where ℓ is HALF-LENGTH (textbook convention in your screenshot earlier).
        """
        th = np.clip(theta, 1e-9, np.pi - 1e-9)
        return (np.cos(k_ell * np.cos(th)) - np.cos(k_ell)) / np.sin(th)

    def dipole_power_pattern(theta, k_ell):
        P = np.abs(dipole_F_theta(theta, k_ell)) ** 2
        P /= np.max(P) if np.max(P) > 0 else 1.0
        return P

    def compute_hpbw(theta, Pnorm):
        imax = int(np.argmax(Pnorm))

        left = None
        for i in range(imax, 0, -1):
            if Pnorm[i] >= 0.5 and Pnorm[i - 1] < 0.5:
                left = i
                break

        right = None
        for i in range(imax, len(Pnorm) - 1):
            if Pnorm[i] >= 0.5 and Pnorm[i + 1] < 0.5:
                right = i
                break

        if left is None or right is None:
            return None, None, None

        def interp(i_hi, i_lo):
            t1, t2 = theta[i_lo], theta[i_hi]
            p1, p2 = Pnorm[i_lo], Pnorm[i_hi]
            if p2 == p1:
                return t1
            return t1 + (0.5 - p1) * (t2 - t1) / (p2 - p1)

        th1 = interp(left, left - 1)
        th2 = interp(right, right + 1)

        return float(np.degrees(th1)), float(np.degrees(th2)), float(np.degrees(th2 - th1))

    def radiation_resistance_from_F(k_ell, eta=377.0, Rref=73.13, ntheta=20001):
        """
        Radiation resistance computed via pattern integral, then calibrated
        against known half-wave dipole value Rref at kℓ_ref = π/2.
        """
        eta0 = 377.0
        theta = np.linspace(1e-9, np.pi - 1e-9, ntheta)

        def I_of(kell):
            F = dipole_F_theta(theta, kell)
            return float(np.trapz((np.abs(F) ** 2) * np.sin(theta), theta))

        I_target = I_of(k_ell)
        I_ref = I_of(np.pi / 2.0)

        return (eta / eta0) * Rref * (I_target / I_ref)

    # ---------- HW2: Array Factor ----------
    def array_factor_mag(N, d_over_lambda, psi, theta):
        """
        Uniform linear array AF magnitude.
        u = k d cosθ + ψ, with kd = 2π d/λ
        """
        kd = 2.0 * np.pi * d_over_lambda
        u = kd * np.cos(theta) + psi

        num = np.sin(N * u / 2.0)
        den = np.sin(u / 2.0)

        # handle den ~ 0 with safe replacement
        den = np.where(np.abs(den) < 1e-12, 1e-12, den)
        AF = np.abs(num / den)
        AF /= np.max(AF) if np.max(AF) > 0 else 1.0
        return AF

    def polar_plot_af(theta, AF, title=""):
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="polar")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

        # Use θ as angle on polar; many texts use θ from +z.
        ax.plot(theta, AF, linewidth=2)
        ax.set_title(title, pad=18)
        st.pyplot(fig)

    # ============================================================
    # UI: Choose Homework
    # ============================================================
    hw = st.sidebar.selectbox("Choose homework", ["Homework 1", "Homework 2", "Homework 3"], index=0)

    # ============================================================
    # Homework 1
    # ============================================================
    if hw == "Homework 1":
        st.subheader("Homework 1")

        prob = st.selectbox("Choose problem", ["Problem 1", "Problem 3", "Problem 5"], index=0)

        # ------------------ HW1 P1 ------------------
        if prob == "Problem 1":
            st.markdown("## Problem 1 — short current element (Hertzian dipole)")
            st.write("Given: dℓ = 0.1 m, I = 1∠0° A, f = 10 MHz. Evaluate at θ = 90° and r = 1, 5, 10 m.")

            I0 = 1.0
            dl = 0.1
            f = 10e6

            cols = st.columns(3)
            rs = [1.0, 5.0, 10.0]
            results = []
            for r in rs:
                E, H, k = hertzian_fields_theta90(I0, dl, f, r)
                results.append((r, E, H))

            st.markdown("### (a) Field magnitudes at θ = 90°")
            for (r, E, H) in results:
                st.write(f"r = {r:.0f} m:  |Eθ| = {E:.4f} V/m,   |Hφ| = {H:.6e} A/m,   |E|/|H| = {E/H:.1f} Ω")

            st.markdown("### (b) Near vs far comparison (why r=5 m is borderline)")
            st.write("Compare term sizes in Eθ bracket: k/r (far), 1/r² (induction), 1/(k r³) (electrostatic).")
            c = 299_792_458.0
            lam = c / f
            k = 2*np.pi / lam
            for r in rs:
                far = k/r
                ind = 1/(r**2)
                es  = 1/(k*r**3)
                st.write(f"r={r:.0f} m: k/r={far:.4f}, 1/r²={ind:.4f}, 1/(k r³)={es:.4f}")

            st.markdown("### (c) Far-field plane-wave behavior")
            st.write("As r increases, 1/r dominates, and |E|/|H| approaches η0 ≈ 377 Ω.")

        # ------------------ HW1 P3 ------------------
        elif prob == "Problem 3":
            with st.expander("Problem 3 — Beamwidths & Directivity from Given Power Density", expanded=False):
                st.markdown("""
                ### Given
                The far-field power density is:

                P(θ, φ) = K sinθ cosφ / r²  

                Valid over:
                - 0 ≤ θ ≤ π  
                - −π/2 ≤ φ ≤ π/2  

                In the far field, the **radiation intensity** is:

                U(θ, φ) = r² P(θ, φ) = K sinθ cosφ

                The maximum radiation occurs at:
                - θ = 90°
                - φ = 0°

                So:
                Uₘₐₓ = K

                ---

                ### (a) Half-Power Beamwidths (HPBW)

                **Azimuth plane (θ = 90°):**

                U(φ) = K cosφ  

                Half-power condition:

                U / Uₘₐₓ = 1/2  
                cosφ = 1/2 → φ = ±60°

                Azimuth HPBW = **120°**

                ---

                **Elevation plane (φ = 0°):**

                U(θ) = K sinθ  

                sinθ = 1/2 → θ = 30°, 150°

                Elevation HPBW = **120°**

                ---

                ### (b) First-Null Beamwidths (FNBW)

                Azimuth nulls: φ = ±90° → **FNBW = 180°**  
                Elevation nulls: θ = 0°, 180° → **FNBW = 180°**

                ---

                ### (c) Directivity

                Total radiated power:

                Pᵣₐd = ∬ U(θ, φ) dΩ  

                Pᵣₐd = K ∫ cosφ dφ ∫ sin²θ dθ  
                Pᵣₐd = Kπ

                Directivity:

                D = 4πUₘₐₓ / Pᵣₐd = 4  

                D(dBi) = **6.02 dBi**

                ---

                ### Final Answers
                - Azimuth HPBW = **120°**
                - Elevation HPBW = **120°**
                - Azimuth FNBW = **180°**
                - Elevation FNBW = **180°**
                - Directivity = **4 (6.02 dBi)**
                """)
            st.markdown("## Problem 3 — beamwidths + directivity from given power density")
            st.write("Given: P = K sinθ cosφ / r² over 0≤θ≤π and -π/2≤φ≤π/2.")
            ans = hw1_prob3_results()

            st.markdown("### (a) Half-power beamwidths")
            st.write(f"Azimuth HPBW = {ans['HPBW_az_deg']}°")
            st.write(f"Elevation HPBW = {ans['HPBW_el_deg']}°")

            st.markdown("### (b) First-null beamwidths")
            st.write(f"Azimuth FNBW = {ans['FNBW_az_deg']}°")
            st.write(f"Elevation FNBW = {ans['FNBW_el_deg']}°")

            st.markdown("### (c) Directivity")
            st.write(f"Directivity D = {ans['Directivity']:.2f} (linear)")
            st.write(f"Directivity ≈ {ans['Directivity_dBi']:.2f} dBi (assuming ideal efficiency)")

        # ------------------ HW1 P5 ------------------
        if prob == "Problem 5":
            st.markdown("## HW1 Problem 5 — Thin-wire dipole (textbook model)")

            ell_over_lambda = 0.056
            beta_ell = 2*np.pi*ell_over_lambda
            eta0 = 377.0

            theta = np.linspace(1e-6, np.pi-1e-6, 40001)

            F = (np.cos(beta_ell*np.cos(theta)) - np.cos(beta_ell)) / np.sin(theta)
            P = np.abs(F)**2
            P /= np.max(P)

            # Half-power points
            idx = np.where(P >= 0.5)[0]
            th_hp = np.degrees(theta[idx[0]])
            BW = 2*(90 - th_hp)

            # Radiation resistance (professor's formula)
            Ra = (120*np.pi/6)*(2*np.pi)**2*(ell_over_lambda**2)

            st.write(f"Half-power point ≈ {th_hp:.2f}°")
            st.write(f"HPBW ≈ {BW:.1f}°")
            st.write(f"Radiation resistance Rₐ ≈ {Ra:.2f} Ω")

            fig, ax = plt.subplots()
            ax.plot(np.degrees(theta), P)
            ax.axhline(0.5, linestyle="--")
            ax.set_xlabel("θ (deg)")
            ax.set_ylabel("|F(θ)|² (normalized)")
            ax.grid(True)
            st.pyplot(fig)

            st.markdown("### Water case (εᵣ = 81)")
            ell_over_lambda_water = ell_over_lambda * 9
            st.write(f"ℓ/λ (water) = {ell_over_lambda_water:.3f}")
            with st.expander("Problem 5 — Short Dipole Beamwidth & Radiation Resistance", expanded=False):
                st.markdown("""
                    ### Given
                    Short dipole with electrical length:

                    ℓ / λ₀ = 0.056

                    Radiation pattern factor:

                    F(θ) = [cos(β₀ℓ cosθ) − cos(β₀ℓ)] / sinθ

                    ---

                    ### (a) Dipole in Air

                    From |F(θ)|², half-power points occur at:

                    θ_HP = 45.3°

                    Beamwidth:

                    HPBW = 2(90° − 45.3°) = **89.4°**

                    ---

                    ### Radiation Resistance

                    Rₐ = (120π / 6)(2π)² (ℓ / λ₀)²  

                    Rₐ ≈ **7.78 Ω**

                    ---

                    ### (b) Dipole in Water (εᵣ = 81)

                    Wavelength:

                    λ = λ₀ / √εᵣ = λ₀ / 9  

                    New electrical length:

                    ℓ / λ = **0.504**

                    Half-power points from |F(θ)|²:

                    θ_HP ≈ 66.34°

                    Beamwidth:

                    HPBW = **47.32°**

                    ---

                    ### Final Answers

                    **Air**
                    - HPBW = **89.4°**
                    - Radiation resistance = **7.78 Ω**

                    **Water**
                    - ℓ / λ = **0.504**
                    - HPBW = **47.32°**
                    """)

    # ============================================================
    # Homework 2
    # ============================================================
    elif hw == "Homework 2":
        st.subheader("Homework 2")

        prob = st.selectbox("Choose problem", ["Problem 8", "Problem 9", "Problem 22 (Yagi-Uda)"], index=0)

        # ------------------ HW2 P8 ------------------
        if prob == "Problem 8":
            st.markdown("## Problem 8 — radiation pattern via graphical procedure (array factor)")
            st.write("We plot normalized |AF(θ)| for each specified case. (Uniform excitation, linear array)")

            case = st.selectbox(
                "Choose case",
                ["(a) 5-element broadside, d=λ/2",
                 "(b) 5-element broadside, d=λ",
                 "(c) 5-element end-fire, d/λ=0.8"],
                index=0
            )

            N = 5
            theta = np.linspace(0, np.pi, 2001)

            if case.startswith("(a)"):
                d_over_lambda = 0.5
                psi = 0.0
                AF = array_factor_mag(N, d_over_lambda, psi, theta)
                polar_plot_af(theta, AF, title="N=5 broadside, d=λ/2 (ψ=0)")
                st.write("Broadside: main lobe at θ=90°; spacing λ/2 avoids grating lobes.")

            elif case.startswith("(b)"):
                d_over_lambda = 1.0
                psi = 0.0
                AF = array_factor_mag(N, d_over_lambda, psi, theta)
                polar_plot_af(theta, AF, title="N=5 broadside, d=λ (ψ=0)")
                st.write("d=λ causes grating lobes (multiple strong lobes).")

            else:
                d_over_lambda = 0.8
                kd = 2*np.pi*d_over_lambda
                psi = -kd  # end-fire toward θ=0°
                AF = array_factor_mag(N, d_over_lambda, psi, theta)
                polar_plot_af(theta, AF, title="N=5 end-fire, d/λ=0.8 (ψ=−kd)")
                st.write("End-fire: choose ψ≈−kd so the phase adds toward θ=0°.")

        # ------------------ HW2 P9 ------------------
        elif prob == "Problem 9":
            st.markdown("## Problem 9 — sketch array factor for 4 elements")
            st.write("We plot normalized |AF(θ)| to guide your sketch.")

            case = st.selectbox(
                "Choose case",
                ["(a) broadside, d=λ/2 (ψ=0)",
                 "(b) d=λ/4 and ψ=π/2"],
                index=0
            )

            N = 4
            theta = np.linspace(0, np.pi, 2001)

            if case.startswith("(a)"):
                d_over_lambda = 0.5
                psi = 0.0
                AF = array_factor_mag(N, d_over_lambda, psi, theta)
                polar_plot_af(theta, AF, title="N=4 broadside, d=λ/2 (ψ=0)")
                st.write("Sketch notes: symmetric about θ=90°, main lobe at 90°, nulls near ~60° and ~120°.")

            else:
                d_over_lambda = 0.25
                psi = np.pi/2
                AF = array_factor_mag(N, d_over_lambda, psi, theta)
                polar_plot_af(theta, AF, title="N=4, d=λ/4, ψ=π/2")
                st.write("Sketch notes: beam is steered away from broadside (toward θ≈180° direction in this convention).")

        # ------------------ HW2 P22 ------------------
        elif prob == "Problem 22 (Yagi-Uda)":
            st.markdown("## HW2 Problem 22 — 3-element Yagi-Uda (mutual coupling → currents → pattern)")

            

            st.markdown("### Where the coupling values come from (textbook figures)")
            st.write(
                "Mutual impedances are read from Fig. 9.37 at the given spacing d/λ, "
                "using the curve corresponding to element length ℓ/λ (0.55, ~0.5, 0.4). "
                "Then Zₘₙ = Rₘₙ + jXₘₙ."
            )

            # Show Figure 9.37 for citation / oral explanation
            st.image(
                "fig_9_37.png",
                caption="Figure 9.37 — Mutual impedance of side-by-side coupled linear antennas (R and X vs d/λ)",
                use_container_width=True
            )

            st.markdown("### Given geometry (from the problem)")
            st.write("- Reflector length: ℓ/λ = 0.55")
            st.write("- Driven length:    ℓ/λ = 0.50 (center-fed)")
            st.write("- Director length:  ℓ/λ = 0.40")
            st.write("- Spacing: d = λ/4  ⇒  d/λ = 0.25")
            st.write("- Excitation: V₂ = 1∠0° V, with V₁ = V₃ = 0 (only the driven element is fed)")

            st.markdown("### Step 1 — Build the impedance matrix")
            st.write(
                "We assemble a 3×3 Z-matrix using self impedances Z₁₁, Z₂₂, Z₃₃ and mutual impedances "
                "Z₁₂, Z₁₃, Z₂₃ (symmetric for side-by-side: Z₁₂=Z₂₁, etc.)."
            )

            st.latex(r"\mathbf{V}=\mathbf{Z}\mathbf{I}")
            st.latex(r"\begin{bmatrix}V_1\\V_2\\V_3\end{bmatrix}="
                    r"\begin{bmatrix}Z_{11}&Z_{12}&Z_{13}\\Z_{21}&Z_{22}&Z_{23}\\Z_{31}&Z_{32}&Z_{33}\end{bmatrix}"
                    r"\begin{bmatrix}I_1\\I_2\\I_3\end{bmatrix}")

            st.markdown("### (Values used in our solution — approximated from Figures 9.35–9.37)")
            st.write("If your professor wants: point to Fig 9.37 for mutual R and X, then explain Z = R + jX.")

            # Default values from your handwritten solution screenshot (editable)
            colA, colB, colC = st.columns(3)
            with colA:
                Z11 = st.text_input("Z11 (Ω)", "75 + j100")
                Z12 = st.text_input("Z12 (Ω)", "60 - j30")
                Z13 = st.text_input("Z13 (Ω)", "-10 - j25")
            with colB:
                Z21 = st.text_input("Z21 (Ω)", "60 - j30")
                Z22 = st.text_input("Z22 (Ω)", "73 + j43")
                Z23 = st.text_input("Z23 (Ω)", "35 - j20")
            with colC:
                Z31 = st.text_input("Z31 (Ω)", "-10 - j25")
                Z32 = st.text_input("Z32 (Ω)", "35 - j20")
                Z33 = st.text_input("Z33 (Ω)", "70 - j100")

            def parse_Z(s: str) -> complex:
                """
                Accepts: '75 + j100', '60 - j30', '-10 - j25', '73+j43', '70-j100'
                Also accepts: '75+100j', '75+1j*100'
                """
                s = s.strip().lower().replace(" ", "").replace("i", "j")

                # If already valid python complex like "75+100j", just use it
                try:
                    return complex(s)
                except Exception:
                    pass

                # Convert "j100" -> "100j" and "-j30" -> "-30j"
                # Handle +j... and -j...
                s = s.replace("+j", "+")  # temporarily remove j marker
                s = s.replace("-j", "-")

                # Now we need to put 'j' at the end of the imag number.
                # We assume the format is a + b where b is the imag magnitude.
                # Example: "75+100" should become "75+100j"
                # Example: "60-30"  should become "60-30j"
                # Example: "-10-25" should become "-10-25j"
                # If there is no '+' or '-' (imag part), this will fail, which is fine.

                # Find the split between real and imag by scanning from index 1
                split = None
                for i in range(1, len(s)):
                    if s[i] in ["+", "-"]:
                        split = i
                        break

                if split is None:
                    raise ValueError(f"Could not parse impedance: '{s}'")

                real_part = s[:split]
                imag_part = s[split:]  # includes sign

                # add j to imag part
                return complex(real_part + imag_part + "j")


            Z = np.array([
                [parse_Z(Z11), parse_Z(Z12), parse_Z(Z13)],
                [parse_Z(Z21), parse_Z(Z22), parse_Z(Z23)],
                [parse_Z(Z31), parse_Z(Z32), parse_Z(Z33)],
            ], dtype=complex)

            V = np.array([0+0j, 1+0j, 0+0j], dtype=complex)

            st.markdown("### Step 2 — Solve currents from the matrix equation")
            st.latex(r"\mathbf{I}=\mathbf{Z}^{-1}\mathbf{V}")

            if st.button("Solve currents and plot pattern"):
                I = np.linalg.solve(Z, V)

                st.markdown("#### Currents (phasors)")
                for idx, Ik in enumerate(I, start=1):
                    mag = abs(Ik)
                    ang = np.degrees(np.angle(Ik))
                    st.write(f"I{idx} = {mag:.3f} ∠ {ang:.1f}° A")

                st.markdown("### Step 3 — Compute radiation pattern (using currents as complex weights)")
                st.write("We model the three elements at positions y = −d, 0, +d (spacing d = λ/4).")
                st.latex(r"d=\lambda/4\Rightarrow kd=2\pi(d/\lambda)=\pi/2")
                st.latex(r"AF(\theta)=I_1 e^{-jkd\cos\theta}+I_2+I_3 e^{+jkd\cos\theta}")
                st.write("We plot normalized |AF(θ)| as the Yagi-Uda pattern proxy (good for class).")

                theta = np.linspace(0, np.pi, 2000)
                kd = np.pi / 2
                AF = I[0]*np.exp(-1j*kd*np.cos(theta)) + I[1] + I[2]*np.exp(1j*kd*np.cos(theta))
                AF = np.abs(AF)
                AF /= np.max(AF) if np.max(AF) > 0 else 1.0

                fig = plt.figure(figsize=(7, 7))
                ax = fig.add_subplot(111, projection="polar")
                ax.set_theta_zero_location("N")
                ax.set_theta_direction(-1)
                ax.plot(theta, AF, linewidth=2)
                ax.set_title("Yagi-Uda Pattern (normalized |AF(θ)|)", pad=18)
                st.pyplot(fig)

                st.info(
                    "explanation line: 'I used Fig 9.37 to estimate mutual R and X at d/λ = 0.25, "
                    "formed Zmn = Rmn + jXmn, assembled the Z-matrix, solved for induced currents, "
                    "then used those complex currents as weights in AF(θ) to get the radiation pattern.'"
                )



    # ============================================================
    # Homework 3
    # ============================================================
    elif hw == "Homework 3":
        st.subheader("Homework 3 — Antennas in Communications + Radar Applications")

        st.info("This section includes quick calculators + worked steps.")

        tab1, tab2 = st.tabs(["Comms / Link Budget", "Radar Equation (from handout)"])

        # =========================
        # TAB 1 — COMMS / LINK
        # =========================
        with tab1:
            st.markdown("### Link / Coverage Calculator (far-field S, E-field, received power, received V)")

            c = 3e8
            eta0 = 377.0

            col1, col2, col3 = st.columns(3)
            with col1:
                f_GHz = st.number_input("Frequency f (GHz)", value=2.4, min_value=0.01, step=0.1, key="hw3_link_f_GHz")
                Pt_W = st.number_input("Transmit power Pt (W)", value=10.0, min_value=0.0, step=1.0, key="hw3_link_Pt_W")
                Gt_dBi = st.number_input("Tx gain Gt (dBi)", value=5.0, step=0.5, key="hw3_link_Gt_dBi")

            with col2:
                Gr_lin = st.number_input("Rx gain Gr (linear)", value=1.5, min_value=0.0, step=0.1, key="hw3_link_Gr_lin")
                r_m = st.number_input("Range r (m)", value=500.0, min_value=0.1, step=50.0, key="hw3_link_r_m")
                RL = st.number_input("Load impedance R_L (Ω)", value=50.0, min_value=1.0, step=5.0, key="hw3_link_RL")

            with col3:
                V_sens_mV = st.number_input("Receiver sensitivity (mV rms)", value=25.0, min_value=0.0, step=1.0, key="hw3_link_Vsens")
                show_safety = st.checkbox("Compare to safety limit 10 mW/cm²", value=True, key="hw3_link_safety")


            f = f_GHz * 1e9
            lam = c / f
            Gt_lin = 10 ** (Gt_dBi / 10)

            # Far-field power density
            S = (Pt_W * Gt_lin) / (4 * np.pi * r_m**2)  # W/m^2

            # E-field (rms)
            E_rms = np.sqrt(S * eta0)

            # Effective aperture
            Ae = (Gr_lin * lam**2) / (4 * np.pi)

            # Received power
            Pr = S * Ae

            # Received voltage (matched load approximation)
            Vrms = np.sqrt(max(Pr, 0.0) * RL)

            st.markdown("#### Results")
            st.write(f"λ = {lam:.4f} m")
            st.write(f"S(r) = {S:.3e} W/m²")
            st.write(f"E_rms(r) = {E_rms:.4f} V/m")
            st.write(f"Ae = {Ae:.6f} m²")
            st.write(f"Pr = {Pr:.3e} W")
            st.write(f"Vrms ≈ {Vrms*1e3:.3f} mV (assuming matched load)")

            if show_safety:
                safety_W_m2 = 100.0  # 10 mW/cm^2 = 100 W/m^2
                st.write(f"Safety limit: 10 mW/cm² = {safety_W_m2:.1f} W/m²")
                if S < safety_W_m2:
                    st.success("Safety check: BELOW limit ✅")
                else:
                    st.error("Safety check: ABOVE limit ⚠️")

            V_sens = V_sens_mV * 1e-3
            if Vrms >= V_sens and V_sens > 0:
                st.success("Meets sensitivity ✅")
            elif V_sens > 0:
                st.warning("Does NOT meet sensitivity ❌ (increase Pt, gain, or reduce range)")

            st.markdown("#### Equations used")
            st.markdown(
                "- Compute power density: S = Pt*Gt / (4πr²)\n"
                "- Convert to E-field: S = E²/η0\n"
                "- Convert to received power: Pr = S*Ae, where Ae = Gr*λ²/(4π)\n"
                "- Convert to received Vrms using Pr = Vrms²/RL"
            )

        # =========================
        # TAB 2 — RADAR
        # =========================
        with tab2:
            st.markdown("### Radar Equation Calculator (matches the handout chain)")

            st.caption(
                "The handout shows: P_target = (Pt/(4πr²))Gt, then multiply by σ, then spread again, "
                "then multiply by AeR to get received power, with G = (4π/λ²)Ae. "
                "We use that same flow here."
            )
            st.markdown("Source: class handout ‘Antennas in Radar Applications’. :contentReference[oaicite:1]{index=1}")

            c = 3e8

            colA, colB, colC = st.columns(3)
            with colA:
                f_radar_GHz = st.number_input("Radar frequency f (GHz)", value=10.0, min_value=0.1, step=0.5, key="hw3_radar_f_GHz")
                Pt_radar_W = st.number_input("Transmit power Pt (W)", value=10.0, min_value=0.0, step=1.0, key="hw3_radar_Pt_W")

            with colB:
                Gt_dBi_r = st.number_input("Tx gain Gt (dBi)", value=30.0, step=1.0, key="hw3_radar_Gt_dBi")
                Gr_dBi_r = st.number_input("Rx gain Gr (dBi)", value=30.0, step=1.0, key="hw3_radar_Gr_dBi")

            with colC:
                sigma = st.number_input("RCS σ (m²)", value=0.01, min_value=0.0, step=0.01, format="%.4f", key="hw3_radar_sigma")
                r_radar = st.number_input("Range r (m)", value=100.0, min_value=0.1, step=10.0, key="hw3_radar_r_m")

            RL_radar = st.number_input("Receiver load (Ω) for Vrms estimate", value=50.0, min_value=1.0, step=5.0, key="hw3_radar_RL")

            RL_radar = st.number_input("Receiver load (Ω) for Vrms estimate", value=50.0, min_value=1.0, step=5.0)

            fR = f_radar_GHz * 1e9
            lamR = c / fR
            GtR = 10 ** (Gt_dBi_r / 10)
            GrR = 10 ** (Gr_dBi_r / 10)

            # Effective apertures from G = (4π/λ²) Ae  -> Ae = G λ² /(4π)
            AeT = (GtR * lamR**2) / (4 * np.pi)
            AeR = (GrR * lamR**2) / (4 * np.pi)

                    # Handout chain:
                    # P_target = Pt/(4π r^2) * Gt
            P_target = (Pt_radar_W * GtR) / (4 * np.pi * r_radar**2)

                    # P_i-target = σ * P_target
            P_i_target = sigma * P_target

                    # P_scat = P_i-target / (4π r^2)
            P_scat = P_i_target / (4 * np.pi * r_radar**2)

                    # P_R = P_scat * AeR
            Pr_radar = P_scat * AeR

                    # Also show compact monostatic form equivalent (same result)
            Pr_compact = (Pt_radar_W * GtR * GrR * lamR**2 * sigma) / (((4*np.pi)**3) * (r_radar**4))

            Vrms_radar = np.sqrt(max(Pr_radar, 0.0) * RL_radar)

            st.markdown("#### Results")
            st.write(f"λ = {lamR:.4f} m")
            st.write(
                                f"AeT = {AeT:.6e} m^2, AeR = {AeR:.6e} m^2 "
                                "(using G = (4π / λ^2) Ae)"
                        )

            st.write(f"P_target = {P_target:.3e} W/m^2 (handout step)")
            st.caption("Source: Antennas in Radar Applications handout")

            st.write(f"P_i-target = σ·P_target = {P_i_target:.3e}")
            st.write(f"P_scat = {P_scat:.3e}")
            st.write(f"Received power P_R = {Pr_radar:.3e} W")
            st.write(f"Compact check P_R = {Pr_compact:.3e} W (should match)")
            st.write(f"Estimated Vrms (50Ω) ≈ {Vrms_radar*1e3:.3f} mV")

            st.markdown("#### Explanation")
            st.markdown(
                            "- Use handout flow: Tx spreads to target → multiply by σ → spreads back → multiply by AeR\n"
                            "- Use G ↔ Ae relation: G = (4π/λ²)Ae\n"
                            "- Then compare PR (or Vrms) to receiver threshold if given"
                        )

            st.markdown("#### The equations you are literally using (matches handout)")
            st.latex(r"P_{target}=\frac{P_T}{4\pi r^2}G_T")
            st.latex(r"P_{i-target}=\sigma P_{target}")
            st.latex(r"P_{scat}=\frac{P_{i-target}}{4\pi r^2}")
            st.latex(r"P_R=P_{scat}A_{eR}")
            st.latex(r"G=\frac{4\pi}{\lambda^2}A_e")
def module_mom_quiz():
    import streamlit as st

    st.header("Method of Moments (MoM) — Quiz Answers")
    st.caption("Concise, correct answers written in my own words")

    qnum = st.selectbox(
        "Choose a quiz question",
        list(range(1, 16)),
        format_func=lambda x: f"Question {x}",
        key="mom_quiz_select"
    )

    answers = {
        1: """
        **Why the MoM integral equation is different from those we learned in Math courses?**
        
        In Method of Moments, we start from a **physics-based integral equation**
        where the unknown (charge or current) appears inside the integral.
        Unlike standard math problems, we cannot solve this analytically.
        Instead, we approximate the unknown using basis functions and convert
        the integral equation into a **matrix equation**.
        """,

        2: """
        **What is the main difference between the computation domain in FD and MoM?**

        Finite-difference methods discretize the **entire computation domain**
        (the full space where fields exist).
        MoM only discretizes the **sources or boundaries**, such as wires or
        conductors. This greatly reduces the number of unknowns, but results
        in a dense matrix.
        """,

        3: """
        **In MoM, we digitize the source into N segments, creating N unknowns. How do we generate N equations?**
        After dividing the source into N segments, we generate N equations by
        **enforcing the integral equation at N testing points**.
        Each enforcement condition produces one equation, giving a solvable
        system for the N unknown coefficients.
        """,

        4: """
        **After digitizing the rod into N segments, how did we avoid integrating the charge over each segment?**
        
        Instead of integrating the charge continuously over each segment,
        we approximate the charge using a **simple basis function** (such as
        a pulse or point approximation). This allows the integral equation
        to reduce to a summation over segment centers.
        """,

        5: """
        **What caused singular diagonal elements in the MoM matrix?**
        
        Singular diagonal terms occur because the kernel contains a **1/R**
        dependence. When the observation point coincides with the source
        segment, R approaches zero, causing the integral to become singular.
        """,

        6: """
        The diagonal terms are treated separately using a **finite-radius or
        finite-segment approximation**. This replaces the zero-distance
        interaction with an effective distance so the self-term remains finite.
        """,

        7: """
        Yes, all matrix elements could be treated using more exact integration.
        However, this increases computational cost significantly.
        The trade-off is **higher accuracy versus longer runtime and complexity**.
        """,

        8: """
        Segment length should be small enough to accurately model the source
        variation, but not so small that the matrix becomes unnecessarily large.
        The segment length must also be much larger than the wire radius to
        satisfy the thin-wire assumption.
        """,

        9: """
        Yes. Available data assumes a **thin wire**, meaning the radius must be
        much smaller than the segment length.
        This can be improved by refining the model to include radius effects
        in the self-terms or by using more advanced kernels.
        """,

        10: """
        Basis functions.
        """,

        11: """
        Testing (or weighting) functions.
        """,

        12: """
        Point-matching technique.
        """,

        13: """
        Galerkin method.
        """,

        14: """
        Entire-domain basis functions are functions that span the entire structure
        and better represent the global behavior of the solution.
        """,

        15: """
        No. Singular diagonal elements are not inherent to MoM.
        With proper finite-radius modeling or self-term correction,
        the diagonal elements remain finite.
        """
    }

    st.markdown(f"### Question {qnum}")
    st.markdown(answers[qnum])

def module_antenna_measurements_lab():
    st.header("Antenna Measurements Lab")

    st.markdown(
        """
        This lab documents hands-on antenna measurements performed during the course using
        a **Vector Network Analyzer (VNA)** and an **indoor antenna range (anechoic chamber)**.
        The goal was to connect antenna theory with real measurements of
        **impedance matching, transmission, radiation patterns, and broadband behavior**.
        """
    )

    st.markdown("---")

    # =====================================================
    # PART 1 — VNA LAB (LINEAR WIRE ANTENNAS)
    # =====================================================
    st.subheader("1. VNA Measurements: Linear Wire Antennas")

    st.image(
        "IMG_8553.jpg",
        caption="VNA bench setup used to measure S-parameters between two linear wire antennas",
        use_container_width=True
    )

    st.markdown(
        """
        In this experiment, a **two-port Vector Network Analyzer (VNA)** was used to measure
        S-parameters between two **linear wire antennas**.

        Before taking measurements, a **Short–Open–Load (SOL) calibration** was performed
        using a **50 Ω reference**, ensuring that cable and connector effects were removed.

        The VNA display showed multiple traces:
        - The **purple trace was labeled “21”**, indicating **S21 (forward transmission)** between the antennas
        - The **blue and yellow traces showed deep dips**, which is typical of **reflection measurements**
          (commonly S11 and S22)

        The **dips in the blue and yellow traces** indicate frequencies where the antennas were
        **well matched**, meaning less power was reflected and more power was accepted by the antenna.
        This corresponds to **resonant behavior** of the linear wire antennas.
        """
    )

    st.markdown(
        """
        **Key takeaway:**  
        Reflection coefficients (S11, S22) reveal antenna resonance and impedance matching,
        while S21 shows how efficiently power is transmitted between antennas.
        """
    )

    st.markdown("---")

    # =====================================================
    # PART 2 — INDOOR ANTENNA RANGE (HORN ANTENNAS)
    # =====================================================
    st.subheader("2. Indoor Antenna Range: Horn Antennas")

    st.image(
        "IMG_8554(1).jpg",
        caption="Indoor antenna range (anechoic chamber) lined with RF absorber material",
        use_container_width=True
    )

    st.markdown(
        """
        Radiation pattern measurements were performed in the **indoor antenna range
        (anechoic chamber)** located in **Holmes 453**.

        The chamber is lined with RF absorber foam to:
        - Minimize reflections
        - Reduce multipath interference
        - Simulate **free-space radiation conditions**

        **Horn antennas** (sometimes informally referred to as “bell antennas”) were used
        as both transmitting and receiving antennas.
        Horn antennas are commonly used in antenna ranges because they have:
        - Well-defined radiation patterns
        - Good impedance matching
        - Broad bandwidth
        """
    )

    st.markdown("---")

    # =====================================================
    # PART 3 — FREE-SPACE MEASUREMENT & SOFTWARE
    # =====================================================
    st.subheader("3. Free-Space Transmission and Radiation Pattern Measurement")

    st.image(
        "IMG_8555(1).jpg",
        caption="Antenna alignment and measurement display inside the indoor antenna range",
        use_container_width=True
    )

    st.markdown(
        """
        Using the **antenna range software** provided in the lab handout,
        free-space transmission measurements were performed.

        The received signal level (**S21**) was measured as a function of:
        - Frequency
        - Antenna alignment
        - Orientation inside the chamber

        This allowed observation of fundamental antenna characteristics such as:
        - Directionality
        - Beamwidth
        - Effects of alignment and polarization
        """
    )

    st.markdown("---")

    # =====================================================
    # PART 4 — LOG-PERIODIC ANTENNA (LPDA)
    # =====================================================
    st.subheader("4. Broadband Directional Antenna: Log-Periodic Dipole Array (LPDA)")

    st.image(
        "IMG_8941.jpg",
        caption="Log-periodic dipole array (LPDA) mounted inside the indoor antenna range",
        use_container_width=True
    )

    st.markdown(
        """
        In a later lab, a **log-periodic dipole array (LPDA)** was measured.
        The LPDA is a **directional broadband antenna** composed of multiple dipole elements
        whose **lengths and spacing gradually change** along the boom.

        In the indoor antenna range, the LPDA was used for **radiation pattern measurements**
        to observe:
        - Main beam direction
        - Beamwidth
        - Front-to-back behavior

        After radiation pattern measurements, the LPDA was taken to the
        **Network Analysis Lab (Holmes 491)**, where a VNA was used to examine
        **bandwidth and broadband behavior** using S-parameter measurements.
        """
    )

    st.markdown("---")

    st.markdown(
        """
        **Overall conclusion:**  
        These experiments demonstrated how antenna theory translates into real measurements.
        Using a VNA and an anechoic chamber, we observed resonance, impedance matching,
        transmission behavior, and radiation patterns for narrowband and broadband antennas.
        """
    )


# ===========================
# Main Streamlit app
# ===========================

def main():
    st.set_page_config(
        page_title="ECE474 Antennas Recap – Mikayla Shankles",
        layout="wide"
    )

    st.sidebar.title("ECE474 Antennas coursework - Mikayla Shankles")
    st.sidebar.markdown("Use this menu to switch between modules:")

    module = st.sidebar.radio(
        "Select module",
        (
            "Radiation: Wire / Dipole",
            "MoM: Charged Rod",
            "MoM: Wire Antenna",
            "Notes & Theory",
            "Design Projects: Spirals",
            "Homeworks",
            "MoM Quiz Answers",
            "Antenna Measurements Lab"


        )
    )

    if module == "Radiation: Wire / Dipole":
        module_radiation_wire()
    elif module == "MoM: Charged Rod":
        module_mom_charged_rod()
    elif module == "MoM: Wire Antenna":
        module_mom_wire_antenna()
    elif module == "Notes & Theory":
        module_notes_theory()
    elif module == "Design Projects: Spirals":
        module_design_spirals()
    elif module == "Homeworks":
        module_homeworks()
    elif module == "MoM Quiz Answers":
        module_mom_quiz()
    elif module == "Antenna Measurements Lab":
        module_antenna_measurements_lab()





if __name__ == "__main__":
    main()



