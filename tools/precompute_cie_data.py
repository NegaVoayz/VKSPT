#!/usr/bin/env python3
"""
Precompute CIE 1931 2° standard observer data and pseudo-inverse basis
for RGB-to-spectrum estimation.

Output: GLSL const float arrays for CIE_X, CIE_Y, CIE_Z (CMFs)
        and BASIS_R, BASIS_G, BASIS_B (pseudo-inverse spectrum basis).
"""

import numpy as np

# =============================================================================
# CIE 1931 2° Standard Observer Color Matching Functions
# Wavelengths 380–780 nm, 5 nm step.  Values from CIE standard tables.
# =============================================================================

WAVELENGTHS = list(range(380, 785, 5))  # 81 values
N = len(WAVELENGTHS)
DLAMBDA = 5.0  # nm

# CIE 1931 2° CMF data: x_bar, y_bar, z_bar
# Source: CIE 015:2018 Table T.4 (rounded to 6 decimal places)
cie_data = [
    # lam,  x_bar,      y_bar,      z_bar
    (380, 0.001368, 0.000039, 0.006450),
    (385, 0.002236, 0.000064, 0.010550),
    (390, 0.004243, 0.000120, 0.020050),
    (395, 0.007650, 0.000217, 0.036210),
    (400, 0.014310, 0.000396, 0.067850),
    (405, 0.023190, 0.000640, 0.110200),
    (410, 0.043510, 0.001210, 0.207400),
    (415, 0.077630, 0.002180, 0.371300),
    (420, 0.134380, 0.004000, 0.645600),
    (425, 0.214770, 0.007300, 1.039050),
    (430, 0.283900, 0.011600, 1.385600),
    (435, 0.328500, 0.016840, 1.622960),
    (440, 0.348280, 0.023000, 1.747060),
    (445, 0.348060, 0.029800, 1.782600),
    (450, 0.336200, 0.038000, 1.772110),
    (455, 0.318700, 0.048000, 1.744100),
    (460, 0.290800, 0.060000, 1.669200),
    (465, 0.251100, 0.073900, 1.528100),
    (470, 0.195360, 0.090980, 1.287640),
    (475, 0.142100, 0.112600, 1.041900),
    (480, 0.095640, 0.139020, 0.812950),
    (485, 0.058000, 0.169300, 0.616200),
    (490, 0.032000, 0.208020, 0.465180),
    (495, 0.014700, 0.258600, 0.353300),
    (500, 0.004900, 0.323000, 0.272000),
    (505, 0.002400, 0.407300, 0.212300),
    (510, 0.009300, 0.503000, 0.158200),
    (515, 0.029100, 0.608200, 0.111700),
    (520, 0.063270, 0.710000, 0.078250),
    (525, 0.109600, 0.793200, 0.057250),
    (530, 0.165500, 0.862000, 0.042160),
    (535, 0.225750, 0.914850, 0.029840),
    (540, 0.290400, 0.954000, 0.020300),
    (545, 0.359700, 0.980300, 0.013400),
    (550, 0.433450, 0.994950, 0.008750),
    (555, 0.512050, 1.000000, 0.005750),
    (560, 0.594500, 0.995000, 0.003900),
    (565, 0.678400, 0.978600, 0.002750),
    (570, 0.762100, 0.952000, 0.002100),
    (575, 0.842500, 0.915400, 0.001800),
    (580, 0.916300, 0.870000, 0.001650),
    (585, 0.978600, 0.816300, 0.001400),
    (590, 1.026300, 0.757000, 0.001100),
    (595, 1.056700, 0.694900, 0.001000),
    (600, 1.062200, 0.631000, 0.000800),
    (605, 1.045600, 0.566800, 0.000600),
    (610, 1.002600, 0.503000, 0.000340),
    (615, 0.938400, 0.441200, 0.000240),
    (620, 0.854450, 0.381000, 0.000190),
    (625, 0.751400, 0.321000, 0.000100),
    (630, 0.642400, 0.265000, 0.000050),
    (635, 0.541900, 0.217000, 0.000030),
    (640, 0.447900, 0.175000, 0.000020),
    (645, 0.360800, 0.138200, 0.000010),
    (650, 0.283500, 0.107000, 0.000000),
    (655, 0.218700, 0.081600, 0.000000),
    (660, 0.164900, 0.061000, 0.000000),
    (665, 0.121200, 0.044580, 0.000000),
    (670, 0.087400, 0.032000, 0.000000),
    (675, 0.063600, 0.023200, 0.000000),
    (680, 0.046770, 0.017000, 0.000000),
    (685, 0.032900, 0.011920, 0.000000),
    (690, 0.022700, 0.008210, 0.000000),
    (695, 0.015840, 0.005723, 0.000000),
    (700, 0.011359, 0.004102, 0.000000),
    (705, 0.008111, 0.002929, 0.000000),
    (710, 0.005790, 0.002091, 0.000000),
    (715, 0.004109, 0.001484, 0.000000),
    (720, 0.002899, 0.001047, 0.000000),
    (725, 0.002049, 0.000740, 0.000000),
    (730, 0.001440, 0.000520, 0.000000),
    (735, 0.001000, 0.000361, 0.000000),
    (740, 0.000690, 0.000249, 0.000000),
    (745, 0.000476, 0.000172, 0.000000),
    (750, 0.000332, 0.000120, 0.000000),
    (755, 0.000235, 0.000085, 0.000000),
    (760, 0.000166, 0.000060, 0.000000),
    (765, 0.000117, 0.000042, 0.000000),
    (770, 0.000083, 0.000030, 0.000000),
    (775, 0.000059, 0.000021, 0.000000),
    (780, 0.000042, 0.000015, 0.000000),
]
assert len(cie_data) == N

x_bar = np.array([d[1] for d in cie_data])
y_bar = np.array([d[2] for d in cie_data])
z_bar = np.array([d[3] for d in cie_data])

# =============================================================================
# XYZ -> linear sRGB matrix (D65 white point)
# =============================================================================
XYZ_TO_SRGB = np.array([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252],
])
SRGB_TO_XYZ = np.linalg.inv(XYZ_TO_SRGB)

# =============================================================================
# Display-referred CMFs: directly map spectrum → linear sRGB
# Then normalize each channel so equal-energy S(λ)=1 → sRGB (1,1,1)
# =============================================================================

# Per-wavelength linear sRGB response
srgb_R = np.zeros(N)
srgb_G = np.zeros(N)
srgb_B = np.zeros(N)
for i in range(N):
    xyz = np.array([x_bar[i], y_bar[i], z_bar[i]])
    srgb = XYZ_TO_SRGB @ xyz
    srgb_R[i] = srgb[0]
    srgb_G[i] = srgb[1]
    srgb_B[i] = srgb[2]

# Equal-energy response: integrate over full visible spectrum
ee_R = np.sum(srgb_R) * DLAMBDA
ee_G = np.sum(srgb_G) * DLAMBDA
ee_B = np.sum(srgb_B) * DLAMBDA
print(f"Equal-energy raw → linear sRGB: R={ee_R:.4f} G={ee_G:.4f} B={ee_B:.4f}")

# Per-channel normalization: display-referred CMFs
disp_R = srgb_R / ee_R
disp_G = srgb_G / ee_G
disp_B = srgb_B / ee_B

# Verify
ee_R_norm = np.sum(disp_R) * DLAMBDA
ee_G_norm = np.sum(disp_G) * DLAMBDA
ee_B_norm = np.sum(disp_B) * DLAMBDA
print(f"Equal-energy normalized → sRGB: R={ee_R_norm:.4f} G={ee_G_norm:.4f} B={ee_B_norm:.4f}")
print(f"  (should be 1,1,1)")

# =============================================================================
# System matrix M_srgb: maps 81-dim spectrum → linear sRGB
# M_srgb[i] = [disp_R[i]*DLAMBDA, disp_G[i]*DLAMBDA, disp_B[i]*DLAMBDA]^T
# But we need it as 3×81 for pseudo-inverse
# =============================================================================
M_srgb_norm = np.zeros((3, N))
M_srgb_norm[0, :] = disp_R * DLAMBDA
M_srgb_norm[1, :] = disp_G * DLAMBDA
M_srgb_norm[2, :] = disp_B * DLAMBDA

# Pseudo-inverse for RGB → spectrum
MMT_norm = M_srgb_norm @ M_srgb_norm.T  # 3x3
MMT_inv_norm = np.linalg.inv(MMT_norm)
M_pinv_norm = M_srgb_norm.T @ MMT_inv_norm  # 81×3

basis_R_norm = M_pinv_norm[:, 0]
basis_G_norm = M_pinv_norm[:, 1]
basis_B_norm = M_pinv_norm[:, 2]

# Verify basis functions
basis_R_srgb = M_srgb_norm @ basis_R_norm
basis_G_srgb = M_srgb_norm @ basis_G_norm
basis_B_srgb = M_srgb_norm @ basis_B_norm
print(f"Basis R → linear sRGB: {basis_R_srgb}")
print(f"Basis G → linear sRGB: {basis_G_srgb}")
print(f"Basis B → linear sRGB: {basis_B_srgb}")
print(f"  (should be approx [1,0,0], [0,1,0], [0,0,1])")

# =============================================================================
# Output: GLSL const float arrays
# =============================================================================

def format_float_array(name, values, per_line=6):
    """Format a float array as GLSL const float[]."""
    lines = [f"const float {name}[{len(values)}] = {{"]
    for i in range(0, len(values), per_line):
        chunk = values[i:i+per_line]
        line = "    " + ", ".join(f"{v:.8f}" for v in chunk)
        if i + per_line < len(values):
            line += ","
        lines.append(line)
    lines.append("};")
    return "\n".join(lines)

print("\n// =============================================================================")
print("// Display-referred Color Matching Functions (direct to linear sRGB)")
print(f"// {N} samples, {WAVELENGTHS[0]}–{WAVELENGTHS[-1]} nm, step {DLAMBDA} nm")
print("// Per-channel normalized: equal-energy S(λ)=1 → linear sRGB (1,1,1)")
print("// =============================================================================")
print()
print(format_float_array("CMF_R", disp_R))
print()
print(format_float_array("CMF_G", disp_G))
print()
print(format_float_array("CMF_B", disp_B))
print()
print("// =============================================================================")
print("// Pseudo-inverse basis spectra for linear RGB → spectrum estimation")
print("// S(λ) = lin_R × BASIS_R[λ] + lin_G × BASIS_G[λ] + lin_B × BASIS_B[λ]")
print("// =============================================================================")
print()
print(format_float_array("BASIS_R", basis_R_norm))
print()
print(format_float_array("BASIS_G", basis_G_norm))
print()
print(format_float_array("BASIS_B", basis_B_norm))
