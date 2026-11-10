import os
import sys
import json
import numpy as np

# Add Lumerical Python API to path
sys.path.append(r"C:\Program Files\Lumerical\v261\api\python")
import lumapi

def run_ece_characterization():
    print("=" * 60)
    print("Starting ECE Photonic Hardware Characterization in Lumerical...")
    print("=" * 60)
    
    workspace_dir = r"C:\CSE\quantum"
    base_file = os.path.join(workspace_dir, "2nd.lms")
    
    mode = lumapi.MODE(hide=False)
    try:
        mode.load(base_file)
        print("[1/4] Loaded base waveguide model:", base_file)
        
        # 1. Single Waveguide Characterization (Fundamental TE mode)
        mode.eval("""
            switchtolayout;
            setanalysis('wavelength', 1.55e-6);
            setanalysis('number of trial modes', 5);
            findmodes;
            
            # Extract fundamental mode properties
            neff_single = real(getdata('FDE::data::mode1', 'neff'));
            loss_single = getdata('FDE::data::mode1', 'loss'); # dB/m
            ng_single = real(getdata('FDE::data::mode1', 'ng'));
            
            write('ece_run_log.txt', 'Single waveguide solved:');
            write('ece_run_log.txt', 'neff = ' + num2str(neff_single));
            write('ece_run_log.txt', 'loss (dB/m) = ' + num2str(loss_single));
            write('ece_run_log.txt', 'ng = ' + num2str(ng_single));
        """)
        
        # Read intermediate results
        mode.eval("neff_val = neff_single; loss_val = loss_single; ng_val = ng_single;")
        neff_single = float(np.real(mode.getv("neff_val")))
        loss_single = float(np.real(mode.getv("loss_val")))
        ng_single = float(np.real(mode.getv("ng_val")))
        print(f"      -> Fundamental TE neff @ 1550 nm: {neff_single:.4f}")
        print(f"      -> Propagation loss: {loss_single / 100.0:.3e} dB/cm")
        print(f"      -> Group index ng: {ng_single:.4f}")

        # 2. Coupled Waveguide Characterization for 50:50 Directional Coupler
        print("[2/4] Simulating Directional Coupler Supermodes (Symmetric vs Anti-Symmetric)...")
        # Add a second waveguide next to the core to simulate supermode splitting
        mode.eval("""
            switchtolayout;
            # Let's inspect waveguide geometry
            select('core');
            w_core = get('y span');
            h_core = get('z span');
            x_pos = get('x');
            y_pos = get('y');
            z_pos = get('z');
            mat_core = get('material');
            
            # Nominal gap between waveguides
            gap_nom = 0.20e-6; # 200 nm
            
            # Duplicate core to create symmetric 2-waveguide coupler
            copy(0, w_core + gap_nom, 0);
            set('name', 'core_coupled');
            
            # Adjust FDE simulation region to cover both waveguides
            select('FDE');
            y_span_fde = get('y span');
            set('y min', y_pos - w_core*2);
            set('y max', y_pos + w_core*3 + gap_nom);
            
            findmodes;
            
            # Supermode 1 (Even/Symmetric) and Supermode 2 (Odd/Anti-Symmetric)
            n_even = real(getdata('FDE::data::mode1', 'neff'));
            n_odd  = real(getdata('FDE::data::mode2', 'neff'));
            delta_n = abs(n_even - n_odd);
            
            lambda0 = 1.55e-6;
            # 3dB coupling length (exact 50:50 power split): L_3dB = lambda0 / (4 * delta_n)
            L_3dB = lambda0 / (4.0 * delta_n);
            # Full cross-over coupling length (100% transfer): L_pi = lambda0 / (2 * delta_n)
            L_pi = lambda0 / (2.0 * delta_n);
        """)
        
        n_even = float(np.real(mode.getv("n_even")))
        n_odd = float(np.real(mode.getv("n_odd")))
        delta_n = float(np.real(mode.getv("delta_n")))
        L_3dB = float(np.real(mode.getv("L_3dB")))
        L_pi = float(np.real(mode.getv("L_pi")))
        
        print(f"      -> Symmetric Mode (Even) neff: {n_even:.4f}")
        print(f"      -> Anti-Symmetric Mode (Odd) neff: {n_odd:.4f}")
        print(f"      -> Supermode Index Splitting (Delta_n): {delta_n:.5f}")
        print(f"      -> 50:50 3dB Coupling Length (L_3dB): {L_3dB * 1e6:.2f} um")
        print(f"      -> Full Cross Coupling Length (L_pi): {L_pi * 1e6:.2f} um")
        
        # 3. Thermo-Optic Phase Shifter Calibration
        print("[3/4] Calibrating Thermo-Optic Phase Shifter...")
        # Silicon thermo-optic coefficient: dn/dT = 1.86e-4 K^-1
        # Heater length typical L_heater = 50 to 100 um
        L_heater = 100e-6 # 100 um
        # Phase shift: delta_phi = (2*pi/lambda) * (dn/dT * Delta_T) * L_heater
        # For 2*pi phase shift: Delta_T_2pi = lambda / (dn/dT * L_heater)
        dn_dT = 1.86e-4 # 1/K for Si
        delta_T_2pi = 1.55e-6 / (dn_dT * L_heater)
        print(f"      -> Heater length: {L_heater*1e6:.1f} um")
        print(f"      -> Temperature increase for 2*pi phase shift: {delta_T_2pi:.1f} K")
        
        # 4. Packaging specs for CSE / PyTorch Software Stack
        print("[4/4] Packaging specifications into 'hardware_specs.json'...")
        
        # Physical S-matrix for a 50:50 Directional Coupler:
        # T_coupler = [ [t, -j*k], [-j*k, t] ] where t = cos(pi*L/(2*L_pi)), k = sin(pi*L/(2*L_pi))
        # At L = L_3dB, t = k = 1/sqrt(2)
        coupling_ratio = 0.50
        insertion_loss_dB = 0.05 # Typical 0.05 dB per coupler
        insertion_loss_amp = float(10.0 ** (-insertion_loss_dB / 20.0))
        
        hardware_specs = {
            "technology": "Silicon Photonics (SOI) 220nm Platform",
            "operating_wavelength_m": 1.55e-6,
            "waveguide": {
                "neff_fundamental_TE": neff_single,
                "group_index_ng": ng_single,
                "propagation_loss_dB_per_cm": loss_single / 100.0,
                "confinement_factor": 0.85
            },
            "directional_coupler_50_50": {
                "gap_nm": 200.0,
                "n_even": n_even,
                "n_odd": n_odd,
                "delta_neff": delta_n,
                "L_3dB_um": float(L_3dB * 1e6),
                "L_pi_um": float(L_pi * 1e6),
                "nominal_split_ratio": coupling_ratio,
                "coupler_insertion_loss_dB": insertion_loss_dB,
                "fab_imbalance_std": 0.02 # 2% std variation from fab tolerance
            },
            "phase_shifter": {
                "mechanism": "Thermo-optic micro-heater",
                "length_um": float(L_heater * 1e6),
                "dn_dT": dn_dT,
                "delta_T_pi_K": float(delta_T_2pi / 2.0),
                "delta_T_2pi_K": float(delta_T_2pi),
                "phase_noise_std_rad": 0.035, # ~2 degrees thermal phase noise
                "thermal_crosstalk_factor": 0.05 # 5% crosstalk to adjacent waveguide
            },
            "mzi_unit_cell": {
                "architecture": "Push-pull balanced MZI with 2 phase shifters (theta, phi)",
                "transfer_matrix_formula": "U_mzi(theta, phi) = Coupler @ [[exp(j*theta), 0], [0, 1]] @ Coupler @ [[exp(j*phi), 0], [0, 1]]",
                "matrix_order": "Clements rectangular mesh"
            }
        }
        
        output_json_path = os.path.join(workspace_dir, "hardware_specs.json")
        with open(output_json_path, "w") as f:
            json.dump(hardware_specs, f, indent=4)
        
        print("\n" + "=" * 60)
        print("SUCCESS! ECE Hardware Specs exported to:")
        print(output_json_path)
        print("=" * 60)
        
    finally:
        mode.close()

if __name__ == "__main__":
    run_ece_characterization()
