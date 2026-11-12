"""
Step 2: Clements Unitary Mesh Compiler & SVD Phase Shift Decomposer
====================================================================
Joint Authors:
  - Akshay Kumar Hudedmani (Computer Science & Engineering - CSE)
    GitHub: https://github.com/akshaykumarhudedmani
  - Taniya Sana (Electronics & Communication Engineering - ECE)
    GitHub: https://github.com/tanyasana
"""

import json
import numpy as np

def clements_decompose(U):
    """
    Implements the Clements decomposition for an N x N unitary matrix.
    Decomposes U into a rectangular mesh of beam splitters / MZIs and phase shifters.
    Reference: Clements et al., Optica Vol. 3, Issue 12, pp. 1460-1465 (2016).
    
    Returns:
    - mzi_list: list of dicts with (m, n, theta, phi) for each MZI in the Clements mesh
    - diag_phases: final diagonal phase shifts on output channels
    """
    N = U.shape[0]
    U_curr = U.copy().astype(complex)
    mzi_list = []
    
    # Clements rectangular mesh elimination pattern
    for i in range(N - 1):
        if i % 2 == 0:
            # Bottom-left to top-right eliminations
            for j in range(i + 1):
                col = j
                row = N - 1 - i + j
                m = row - 1
                n = row
                
                # Null out U_curr[row, col] using row m and row n
                u1 = U_curr[m, col]
                u2 = U_curr[n, col]
                
                if np.abs(u2) < 1e-12:
                    theta = 0.0
                    phi = 0.0
                else:
                    theta = 2.0 * np.arctan2(np.abs(u2), np.abs(u1))
                    phi = np.angle(u1) - np.angle(u2)
                
                # 2x2 MZI transformation
                # T_inv @ [u1; u2] -> [u1'; 0]
                cos_t = np.cos(theta / 2.0)
                sin_t = np.sin(theta / 2.0)
                
                T_inv = np.array([
                    [np.exp(-1j * phi) * cos_t, sin_t],
                    [-np.exp(-1j * phi) * sin_t, cos_t]
                ], dtype=complex)
                
                # Apply T_inv to rows m and n
                U_curr[[m, n], :] = T_inv @ U_curr[[m, n], :]
                
                mzi_list.append({
                    "layer": i + 1,
                    "channel_m": m,
                    "channel_n": n,
                    "theta_rad": float(theta % (2 * np.pi)),
                    "phi_rad": float(phi % (2 * np.pi))
                })
        else:
            # Top-right to bottom-left eliminations
            for j in range(i + 1):
                col = N - 1 - i + j
                row = j
                m = col - 1
                n = col
                
                u1 = U_curr[row, m]
                u2 = U_curr[row, n]
                
                if np.abs(u1) < 1e-12:
                    theta = np.pi
                    phi = 0.0
                else:
                    theta = 2.0 * np.arctan2(np.abs(u1), np.abs(u2))
                    phi = np.angle(u2) - np.angle(u1)
                
                cos_t = np.cos(theta / 2.0)
                sin_t = np.sin(theta / 2.0)
                
                T = np.array([
                    [np.exp(1j * phi) * cos_t, -sin_t],
                    [np.exp(1j * phi) * sin_t, cos_t]
                ], dtype=complex)
                
                # Apply T to columns m and n
                U_curr[:, [m, n]] = U_curr[:, [m, n]] @ T
                
                mzi_list.append({
                    "layer": i + 1,
                    "channel_m": m,
                    "channel_n": n,
                    "theta_rad": float(theta % (2 * np.pi)),
                    "phi_rad": float(phi % (2 * np.pi))
                })
                
    diag_phases = np.angle(np.diag(U_curr)) % (2 * np.pi)
    return mzi_list, diag_phases.tolist()

def compile_weight_matrix_to_hardware(W, hardware_db):
    """
    Takes an arbitrary N x N weight matrix W from deep learning and compiles it
    into physical micro-heater temperatures, voltages, and optical attenuations.
    """
    N = W.shape[0]
    
    # 1. Singular Value Decomposition: W = U @ diag(Sigma) @ Vh
    U, S, Vh = np.linalg.svd(W)
    
    # 2. Decompose U and V into Clements MZI meshes
    mzi_U, diag_U = clements_decompose(U)
    mzi_V, diag_V = clements_decompose(Vh.conj().T)
    
    # 3. Map Phase Angles to Physical Hardware Controls
    delta_T_pi = hardware_db["phase_shifter"]["delta_T_pi_K"]
    R_heater = hardware_db["phase_shifter"]["heater_resistance_ohms"]
    mW_per_pi = hardware_db["phase_shifter"]["tuning_efficiency_mW_per_pi"]
    
    def annotate_mzi_with_hardware(mzi_list):
        for m in mzi_list:
            th = m["theta_rad"]
            ph = m["phi_rad"]
            
            # Temperatures
            m["heater_theta_delta_T_C"] = round((th / np.pi) * delta_T_pi, 2)
            m["heater_phi_delta_T_C"]   = round((ph / np.pi) * delta_T_pi, 2)
            
            # Electrical Voltages: V = sqrt(P * R) where P = (theta / pi) * P_pi
            p_th = (th / np.pi) * mW_per_pi * 1e-3
            p_ph = (ph / np.pi) * mW_per_pi * 1e-3
            m["heater_theta_voltage_V"] = round(float(np.sqrt(p_th * R_heater)), 3)
            m["heater_phi_voltage_V"]   = round(float(np.sqrt(p_ph * R_heater)), 3)
            
        return mzi_list
        
    mzi_U = annotate_mzi_with_hardware(mzi_U)
    mzi_V = annotate_mzi_with_hardware(mzi_V)
    
    compiled_plan = {
        "matrix_dimension": N,
        "singular_values_sigma": [round(float(s), 4) for s in S],
        "optical_attenuations_dB": [round(float(-20.0 * np.log10(s + 1e-12)), 2) for s in S],
        "mzi_mesh_U": mzi_U,
        "mzi_mesh_V_dagger": mzi_V,
        "output_phase_shifts_rad": [round(float(p), 4) for p in diag_U]
    }
    
    return compiled_plan

if __name__ == "__main__":
    with open(r"C:\CSE\quantum\hardware_database.json") as f:
        hw_db = json.load(f)
        
    print("=" * 75)
    print("STEP 2: TESTING CLEMENTS SVD COMPILER ENGINE FOR 4x4 AI MATRIX")
    print("=" * 75)
    
    # Example 4x4 AI Layer Weight Matrix
    np.random.seed(42)
    W_test = np.array([
        [ 0.65, -0.42,  0.18,  0.88],
        [ 0.12,  0.95, -0.31,  0.05],
        [-0.78,  0.22,  0.64, -0.19],
        [ 0.33, -0.15,  0.82,  0.47]
    ])
    
    compiled = compile_weight_matrix_to_hardware(W_test, hw_db)
    
    with open(r"C:\CSE\quantum\compiled_optical_layer.json", "w") as f:
        json.dump(compiled, f, indent=4)
        
    print(f"Matrix Dimension: {compiled['matrix_dimension']} x {compiled['matrix_dimension']}")
    print(f"Singular values Sigma: {compiled['singular_values_sigma']}")
    print(f"Total MZIs in Mesh U:  {len(compiled['mzi_mesh_U'])}")
    print(f"Total MZIs in Mesh V:  {len(compiled['mzi_mesh_V_dagger'])}")
    print("\nSample Physical Controls for First 3 MZIs in Mesh U:")
    for idx, m in enumerate(compiled["mzi_mesh_U"][:3]):
        print(f"  MZI #{idx+1} (Ch {m['channel_m']}-{m['channel_n']}): "
              f"theta={m['theta_rad']:.3f} rad ({m['heater_theta_voltage_V']:.2f}V, +{m['heater_theta_delta_T_C']:.1f}°C) | "
              f"phi={m['phi_rad']:.3f} rad ({m['heater_phi_voltage_V']:.2f}V, +{m['heater_phi_delta_T_C']:.1f}°C)")
              
    print("\n" + "=" * 75)
    print("STEP 2 COMPLETE: Compiler successfully mapped weight matrix to physical voltages!")
    print(r"Compiled Plan saved to: C:\CSE\quantum\compiled_optical_layer.json")
    print("=" * 75)
