"""
Export All Simulation & Hardware Datasets for Future Photonic Projects
======================================================================
Consolidates and exports all FDE, INTERCONNECT, Touchstone (.s4p),
Clements compiler, and MNIST neural benchmark data into an offline archive.
"""

import os
import shutil
import json
import numpy as np

export_dir = os.path.abspath('EXPORTED_PROJECT_DATA')
os.makedirs(export_dir, exist_ok=True)

# 1. Generate Touchstone .s4p File (Standard Industry Format for 4-Port RF/Photonic S-parameters)
print("1. Generating 4-Port Touchstone S-parameter file (.s4p)...")
from step5_interconnect_circuit import simulate_optical_circuit, load_hardware_and_layer

hw_db, compiled_layer = load_hardware_and_layer()
wavelengths, S_matrix_lambda, _ = simulate_optical_circuit(hw_db, compiled_layer)

freqs_Hz = 3e8 / wavelengths
# Sort frequencies in ascending order (Touchstone requirement)
sort_idx = np.argsort(freqs_Hz)
freqs_sorted = freqs_Hz[sort_idx]
S_sorted = S_matrix_lambda[sort_idx]

s4p_path = os.path.join(export_dir, 'clements_mesh_4x4.s4p')
with open(s4p_path, 'w') as f:
    f.write("! 4-Channel Clements Photonic Neural Network Unitary Mesh\n")
    f.write("! Platform: 220nm SOI Silicon Photonics, Center Wavelength: 1550 nm\n")
    f.write("! Format: Frequency (Hz), S-parameters: Magnitude / Angle (Degrees)\n")
    f.write("# HZ S MA R 50\n")
    
    for f_val, S_mat in zip(freqs_sorted, S_sorted):
        # Format: f S11 S12 S13 S14 \n S21 S22 S23 S24 \n S31 S32 S33 S34 \n S41 S42 S43 S44
        f.write(f"{f_val:.6e} ")
        for i in range(4):
            if i > 0:
                f.write("                 ")
            for j in range(4):
                val = S_mat[i, j]
                mag = np.abs(val)
                ang = np.degrees(np.angle(val))
                f.write(f" {mag:.6f} {ang:.2f}")
            f.write("\n")

print(f"   [OK] Touchstone file generated: {s4p_path}")

# Also copy into C:\CSE\quantum
shutil.copy(s4p_path, 'clements_mesh_4x4.s4p')

# 2. Copy all project databases and models into EXPORTED_PROJECT_DATA
files_to_export = [
    'hardware_database.json',
    'hardware_specs.json',
    'coupler_gap_sweep.csv',
    'compiled_optical_layer.json',
    'clements_mesh_4x4.icp',
    'build_interconnect_mesh.lsf',
    'interconnect_sparams.json',
    'mnist_benchmark_results.json',
    'results_visualizations.png',
    'interconnect_simulation_results.png',
    'mnist_photonic_benchmark.png',
    '2nd.lms',
    'step1_hardware_sweeps.py',
    'step2_clements_compiler.py',
    'step3_photonic_neural_network.py',
    'step4_generate_visualizations.py',
    'step5_interconnect_circuit.py',
    'step6_mnist_photonic_benchmark.py'
]

print("\n2. Archiving all core project files into EXPORTED_PROJECT_DATA...")
manifest = []
for fname in files_to_export:
    if os.path.exists(fname):
        dest = os.path.join(export_dir, fname)
        shutil.copy(fname, dest)
        size_kb = os.path.getsize(dest) / 1024.0
        manifest.append({"file": fname, "size_KB": round(size_kb, 1)})
        print(f"   [COPIED] {fname} ({size_kb:.1f} KB)")
    else:
        print(f"   [MISSING] {fname}")

with open(os.path.join(export_dir, 'export_manifest.json'), 'w') as f:
    json.dump(manifest, f, indent=4)

print(f"\n[COMPLETE] All {len(manifest)} project datasets securely archived in:")
print(f"           {export_dir}")
