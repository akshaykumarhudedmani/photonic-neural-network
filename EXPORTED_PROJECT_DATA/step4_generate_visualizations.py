import json
import matplotlib.pyplot as plt
import numpy as np

# Load Datasets
with open(r"C:\CSE\quantum\hardware_database.json") as f:
    hw_db = json.load(f)

with open(r"C:\CSE\quantum\training_results.json") as f:
    train_res = json.load(f)

# Set up publication-style plot
plt.style.use('default')
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor('#ffffff')

# -------------------------------------------------------------
# Plot 1: Waveguide Dispersion Curve (Lumerical FDE)
# -------------------------------------------------------------
disp = hw_db["waveguide_nominal"]["dispersion_sweep"]
wls = [d["wavelength_um"] for d in disp]
neffs = [d["neff"] for d in disp]
ngs = [d["group_index"] for d in disp]

ax1 = axs[0, 0]
color1 = '#1f77b4'
ax1.set_xlabel('Wavelength (µm)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Effective Index ($n_{eff}$)', color=color1, fontsize=11, fontweight='bold')
line1 = ax1.plot(wls, neffs, 'o-', color=color1, linewidth=2, label='$n_{eff}$')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, linestyle='--', alpha=0.6)

ax1_twin = ax1.twinx()
color2 = '#d62728'
ax1_twin.set_ylabel('Group Index ($n_g$)', color=color2, fontsize=11, fontweight='bold')
line2 = ax1_twin.plot(wls, ngs, 's--', color=color2, linewidth=2, label='$n_g$')
ax1_twin.tick_params(axis='y', labelcolor=color2)
ax1.set_title('1. Silicon Waveguide Dispersion (Lumerical FDE)', fontsize=12, fontweight='bold')

# -------------------------------------------------------------
# Plot 2: Directional Coupler Gap vs 3dB Coupling Length
# -------------------------------------------------------------
coupler_data = hw_db["directional_coupler"]["gap_sweep"]
gaps = [c["gap_nm"] for c in coupler_data]
l3db = [c["L_3dB_um"] for c in coupler_data]
lpi = [c["L_pi_um"] for c in coupler_data]

ax2 = axs[0, 1]
ax2.plot(gaps, l3db, 'o-', color='#2ca02c', linewidth=2.5, label='50:50 3dB Split ($L_{3dB}$)')
ax2.plot(gaps, lpi, 'v--', color='#9467bd', linewidth=2, label='100% Cross-Over ($L_\\pi$)')
ax2.axvline(200, color='gray', linestyle=':', label='Nominal Gap (200 nm)')
ax2.scatter(200, hw_db["directional_coupler"]["nominal_L_3dB_um"], color='red', s=100, zorder=5)
ax2.set_xlabel('Waveguide Gap (nm)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Coupling Length (µm)', fontsize=11, fontweight='bold')
ax2.set_title('2. Coupler Supermode Coupling Length vs Gap', fontsize=12, fontweight='bold')
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(loc='upper left', fontsize=9)

# -------------------------------------------------------------
# Plot 3: Clements 4x4 Optical Mesh Layout
# -------------------------------------------------------------
ax3 = axs[1, 0]
ax3.set_xlim(-0.5, 5.5)
ax3.set_ylim(-0.5, 3.5)

# Draw 4 waveguides
for y in range(4):
    ax3.plot([-0.2, 5.2], [y, y], color='#333333', linewidth=2.5, zorder=1)
    ax3.text(-0.4, y, f'In {4-y}', va='center', ha='right', fontsize=10, fontweight='bold')
    ax3.text(5.4, y, f'Out {4-y}', va='center', ha='left', fontsize=10, fontweight='bold')

# Draw 6 MZIs as rounded bounding boxes
mzi_boxes = [
    (1.0, 2.5, 'MZI 1 (θ₁, φ₁)'), # Ch 0-1 (y=3, y=2)
    (1.0, 0.5, 'MZI 2 (θ₂, φ₂)'), # Ch 2-3 (y=1, y=0)
    (2.3, 1.5, 'MZI 3 (θ₃, φ₃)'), # Ch 1-2 (y=2, y=1)
    (3.6, 2.5, 'MZI 4 (θ₄, φ₄)'), # Ch 0-1 (y=3, y=2)
    (3.6, 0.5, 'MZI 5 (θ₅, φ₅)'), # Ch 2-3 (y=1, y=0)
    (4.7, 1.5, 'MZI 6 (θ₆, φ₆)'), # Ch 1-2 (y=2, y=1)
]

for x, y, label in mzi_boxes:
    rect = plt.Rectangle((x-0.4, y-0.4), 0.8, 0.8, facecolor='#ffbb78', edgecolor='#d62728', linewidth=1.5, zorder=2)
    ax3.add_patch(rect)
    ax3.text(x, y, label, ha='center', va='center', fontsize=8, fontweight='bold')

ax3.set_title('3. Clements 4x4 Unitary MZI Mesh (Hardware Layout)', fontsize=12, fontweight='bold')
ax3.axis('off')

# -------------------------------------------------------------
# Plot 4: Training Loss Curves & Benchmark Accuracies
# -------------------------------------------------------------
ax4 = axs[1, 1]
ax4.plot(train_res["loss_ideal"], color='#1f77b4', linewidth=2, label='Ideal FP32 Training Loss')
ax4.plot(train_res["loss_hat"], color='#ff7f0e', linewidth=2, linestyle='--', label='Hardware-Aware Loss (with Noise)')
ax4.set_xlabel('Epochs', fontsize=11, fontweight='bold')
ax4.set_ylabel('Cross-Entropy Loss', fontsize=11, fontweight='bold')
ax4.set_title('4. Hardware-Aware Training vs Ideal Software', fontsize=12, fontweight='bold')
ax4.grid(True, linestyle='--', alpha=0.6)
ax4.legend(loc='upper right', fontsize=9)

# Add text box with accuracies
acc_text = (
    f"Ideal Software Acc:          {train_res['ideal_accuracy']*100:.1f}%\n"
    f"Uncalibrated Noisy Chip:     {train_res['noisy_uncalibrated_accuracy']*100:.1f}%\n"
    f"Hardware-Aware Trained Acc:  {train_res['hardware_aware_accuracy']*100:.1f}%"
)
ax4.text(0.05, 0.25, acc_text, transform=ax4.transAxes, fontsize=10,
         bbox=dict(boxstyle='round', facecolor='#e6f2ff', edgecolor='#1f77b4', alpha=0.9),
         family='monospace')

plt.tight_layout()
output_img = r"C:\CSE\quantum\results_visualizations.png"
plt.savefig(output_img, dpi=300)
print(f"Publication-quality visualization saved to: {output_img}")
