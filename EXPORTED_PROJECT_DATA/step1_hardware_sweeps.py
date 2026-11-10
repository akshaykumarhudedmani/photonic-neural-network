import os
import sys
import json
import csv
import numpy as np

os.environ['ANSYSLMD_LICENSE_FILE'] = '1055@10.22.8.151'
sys.path.append(r"C:\Program Files\Lumerical\v261\api\python")
import lumapi

def run_hardware_sweeps():
    print("=" * 75)
    print("STEP 1: AUTOMATED ECE HARDWARE PARAMETRIC SWEEPS IN LUMERICAL")
    print("=" * 75)
    
    workspace = r"C:\CSE\quantum"
    
    mode = lumapi.MODE(hide=False)
    try:
        # -------------------------------------------------------------
        # Part A: Single Waveguide Dispersion Sweep (1500 nm - 1600 nm)
        # -------------------------------------------------------------
        print("\n[A] Running Single Waveguide Wavelength Sweep (1500 nm -> 1600 nm)...")
        base_file = os.path.join(workspace, "2nd.lms")
        mode.load(base_file)
        
        wavelengths = np.linspace(1.50e-6, 1.60e-6, 11)
        dispersion_data = []
        
        for wl in wavelengths:
            mode.eval(f"""
                switchtolayout;
                setanalysis('wavelength', {wl});
                findmodes;
                neff_wl = real(getdata('FDE::data::mode1', 'neff'));
                ng_wl   = real(getdata('FDE::data::mode1', 'ng'));
                loss_wl = getdata('FDE::data::mode1', 'loss');
            """)
            n_eff = float(np.real(mode.getv("neff_wl")))
            n_g   = float(np.real(mode.getv("ng_wl")))
            loss  = float(np.real(mode.getv("loss_wl")))
            dispersion_data.append({
                "wavelength_um": round(wl * 1e6, 3),
                "neff": round(n_eff, 5),
                "group_index": round(n_g, 4),
                "loss_dB_per_cm": round(loss / 100.0, 4)
            })
            print(f"    -> lambda = {wl*1e6:.3f} um | neff = {n_eff:.5f} | ng = {n_g:.4f}")
            
        # -------------------------------------------------------------
        # Part B: Directional Coupler Gap Sweep (160 nm -> 240 nm)
        # -------------------------------------------------------------
        print("\n[B] Running Directional Coupler Gap Sweep (160 nm -> 240 nm)...")
        gaps = [0.16e-6, 0.18e-6, 0.20e-6, 0.22e-6, 0.24e-6]
        coupler_results = []
        
        for gap in gaps:
            # Shift wg1 and wg2 symmetrically around 0
            # wg center = +/- (w/2 + gap/2) = +/- (0.25 + gap/2)
            y_center = 0.25e-6 + gap / 2.0
            mode.eval(f"""
                newproject;
                addrect; set('name', 'substrate'); set('x', 0); set('x span', 10e-6); set('y', 0); set('y span', 8e-6); set('z min', -2e-6); set('z max', 0); set('material', 'SiO2 (Glass) - Palik');
                addrect; set('name', 'wg1'); set('x', 0); set('x span', 10e-6); set('y', {-y_center}); set('y span', 0.50e-6); set('z min', 0); set('z max', 0.22e-6); set('material', 'Si (Silicon) - Palik');
                addrect; set('name', 'wg2'); set('x', 0); set('x span', 10e-6); set('y', {y_center}); set('y span', 0.50e-6); set('z min', 0); set('z max', 0.22e-6); set('material', 'Si (Silicon) - Palik');
                addfde; set('solver type', '2D X normal'); set('x', 0); set('y', 0); set('y span', 4.5e-6); set('z', 0.11e-6); set('z span', 2.5e-6); set('wavelength', 1.55e-6); set('number of trial modes', 5);
                findmodes;
                n_even = real(getdata('mode1', 'neff'));
                n_odd  = real(getdata('mode2', 'neff'));
                delta_n = abs(n_even - n_odd);
                L_3dB = 1.55e-6 / (4.0 * delta_n);
                L_pi  = 1.55e-6 / (2.0 * delta_n);
            """)
            ne = float(np.real(mode.getv("n_even")))
            no = float(np.real(mode.getv("n_odd")))
            dn = float(np.real(mode.getv("delta_n")))
            l3 = float(np.real(mode.getv("L_3dB")))
            lp = float(np.real(mode.getv("L_pi")))
            
            coupler_results.append({
                "gap_nm": round(gap * 1e9, 1),
                "n_even": round(ne, 5),
                "n_odd": round(no, 5),
                "delta_n": round(dn, 5),
                "L_3dB_um": round(l3 * 1e6, 2),
                "L_pi_um": round(lp * 1e6, 2)
            })
            print(f"    -> Gap = {gap*1e9:.0f} nm | Delta_n = {dn:.5f} | L_3dB = {l3*1e6:.2f} um | L_pi = {lp*1e6:.2f} um")
        
        # -------------------------------------------------------------
        # Part C: Save Master Hardware Database
        # -------------------------------------------------------------
        nominal_coupler = [c for c in coupler_results if c["gap_nm"] == 200.0][0]
        
        # Thermal Phase Shifter Specs:
        L_heater = 100e-6
        dn_dT = 1.86e-4
        delta_T_pi = 1.55e-6 / (2.0 * dn_dT * L_heater)
        delta_T_2pi = delta_T_pi * 2.0
        
        master_db = {
            "platform": "Silicon-on-Insulator (SOI) 220nm",
            "operating_wavelength_um": 1.55,
            "waveguide_nominal": {
                "width_nm": 500,
                "height_nm": 220,
                "neff_1550nm": dispersion_data[5]["neff"], # at 1.55 um
                "ng_1550nm": dispersion_data[5]["group_index"],
                "propagation_loss_dB_per_cm": 0.0,
                "dispersion_sweep": dispersion_data
            },
            "directional_coupler": {
                "nominal_gap_nm": 200.0,
                "nominal_L_3dB_um": nominal_coupler["L_3dB_um"],
                "nominal_L_pi_um": nominal_coupler["L_pi_um"],
                "gap_sweep": coupler_results,
                "split_ratio": 0.50,
                "split_tolerance_std": 0.02, # 2% std variation
                "insertion_loss_dB": 0.05
            },
            "phase_shifter": {
                "type": "Thermo-optic micro-heater",
                "length_um": round(L_heater * 1e6, 1),
                "delta_T_pi_K": round(delta_T_pi, 1),
                "delta_T_2pi_K": round(delta_T_2pi, 1),
                "tuning_efficiency_mW_per_pi": 20.0,
                "heater_resistance_ohms": 100.0,
                "thermal_phase_noise_std_rad": 0.035,
                "thermal_crosstalk_factor": 0.05
            }
        }
        
        db_path = os.path.join(workspace, "hardware_database.json")
        with open(db_path, "w") as f:
            json.dump(master_db, f, indent=4)
            
        csv_path = os.path.join(workspace, "coupler_gap_sweep.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["gap_nm", "n_even", "n_odd", "delta_n", "L_3dB_um", "L_pi_um"])
            writer.writeheader()
            writer.writerows(coupler_results)
            
        print("\n" + "=" * 75)
        print("STEP 1 COMPLETE: All hardware parametric datasets extracted & saved!")
        print(f"Master Database: {db_path}")
        print(f"Coupler CSV:     {csv_path}")
        print("=" * 75)
        
    finally:
        mode.close()

if __name__ == "__main__":
    run_hardware_sweeps()
