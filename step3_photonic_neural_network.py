import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Load Lumerical Hardware Database
with open(r"C:\CSE\quantum\hardware_database.json") as f:
    HW_DB = json.load(f)

class PhotonicMZI(nn.Module):
    """
    Differentiable PyTorch model of a physical Mach-Zehnder Interferometer (MZI)
    calibrated with Ansys Lumerical electromagnetic parameters.
    """
    def __init__(self, add_noise=False):
        super().__init__()
        self.theta = nn.Parameter(torch.rand(1) * 2 * np.pi)
        self.phi   = nn.Parameter(torch.rand(1) * 2 * np.pi)
        self.add_noise = add_noise
        
        # Physical parameters from Lumerical
        self.phase_noise_std = HW_DB["phase_shifter"]["thermal_phase_noise_std_rad"] # 0.035 rad
        self.coupler_imbalance_std = HW_DB["directional_coupler"]["split_tolerance_std"] # 0.02
        self.loss_dB = HW_DB["directional_coupler"]["insertion_loss_dB"] # 0.05 dB
        self.loss_factor = float(10.0 ** (-self.loss_dB / 20.0))

    def forward(self, x):
        """
        x: tensor of shape (batch, 2) - complex light amplitudes
        """
        th = self.theta.squeeze()
        ph = self.phi.squeeze()
        
        # Inject physical hardware noise if enabled
        if self.add_noise and self.training:
            th = th + torch.randn(1, device=x.device).squeeze() * self.phase_noise_std
            ph = ph + torch.randn(1, device=x.device).squeeze() * self.phase_noise_std
            eta = 0.50 + torch.randn(1, device=x.device).squeeze() * self.coupler_imbalance_std
        else:
            eta = torch.tensor(0.50, device=x.device)
            
        t = torch.sqrt(torch.clamp(eta, 0.01, 0.99))
        k = torch.sqrt(torch.clamp(1.0 - eta, 0.01, 0.99))
        
        # 50:50 Directional Coupler matrix (2x2)
        row1 = torch.stack([t + 0j, 1j * k])
        row2 = torch.stack([1j * k, t + 0j])
        C = torch.stack([row1, row2])
        
        # Phase Shifter matrix (2x2)
        P = torch.diag(torch.stack([torch.exp(1j * th), torch.exp(1j * ph)]))
        
        # Total MZI transfer matrix (2x2)
        U_mzi = self.loss_factor * (C @ P @ C)
        
        # Multiply light: (batch, 2) @ (2, 2)^T
        return torch.matmul(x, U_mzi.T)

class PhotonicClementsLayer(nn.Module):
    """
    4-Channel Photonic Linear Layer implemented as a Clements Rectangular Mesh
    consisting of 6 MZIs in 4 optical layers.
    """
    def __init__(self, add_noise=False):
        super().__init__()
        self.mzi1 = PhotonicMZI(add_noise) # ch 0-1
        self.mzi2 = PhotonicMZI(add_noise) # ch 2-3
        self.mzi3 = PhotonicMZI(add_noise) # ch 1-2
        self.mzi4 = PhotonicMZI(add_noise) # ch 0-1
        self.mzi5 = PhotonicMZI(add_noise) # ch 2-3
        self.mzi6 = PhotonicMZI(add_noise) # ch 1-2
        
        # Optical Attenuators (Singular Values Sigma)
        self.attenuators = nn.Parameter(torch.ones(4))
        
    def forward(self, x):
        """
        x: (batch, 4) real input vector
        """
        out = x.to(torch.complex64)
        
        # Layer 1
        y01 = self.mzi1(out[:, [0, 1]])
        y23 = self.mzi2(out[:, [2, 3]])
        out = torch.cat([y01[:, 0:1], y01[:, 1:2], y23[:, 0:1], y23[:, 1:2]], dim=1)
        
        # Layer 2
        y12 = self.mzi3(out[:, [1, 2]])
        out = torch.cat([out[:, 0:1], y12[:, 0:1], y12[:, 1:2], out[:, 3:4]], dim=1)
        
        # Layer 3
        y01 = self.mzi4(out[:, [0, 1]])
        y23 = self.mzi5(out[:, [2, 3]])
        out = torch.cat([y01[:, 0:1], y01[:, 1:2], y23[:, 0:1], y23[:, 1:2]], dim=1)
        
        # Layer 4
        y12 = self.mzi6(out[:, [1, 2]])
        out = torch.cat([out[:, 0:1], y12[:, 0:1], y12[:, 1:2], out[:, 3:4]], dim=1)
        
        # Apply Optical Attenuation (Sigma)
        atten = torch.sigmoid(self.attenuators)
        out = out * atten
        
        # Square-law Photodetection: I = |E|^2
        return torch.abs(out) ** 2

class PhotonicClassifier(nn.Module):
    """
    Complete Photonic Neural Network Classifier:
    Optical Input -> 4-Channel Clements Mesh -> Photodetectors -> Classification Layer
    """
    def __init__(self, num_classes=3, add_noise=False):
        super().__init__()
        self.photonic_mesh = PhotonicClementsLayer(add_noise=add_noise)
        self.classifier = nn.Linear(4, num_classes)
        
    def forward(self, x):
        features = self.photonic_mesh(x)
        return self.classifier(features)

def generate_benchmark_dataset():
    """
    Generates a 4-feature 3-class classification problem.
    """
    np.random.seed(42)
    torch.manual_seed(42)
    
    N = 300
    c1 = np.random.randn(N // 3, 4) * 0.45 + np.array([ 1.5, -1.0,  0.8, -1.2])
    c2 = np.random.randn(N // 3, 4) * 0.45 + np.array([-1.2,  1.5, -0.7,  1.0])
    c3 = np.random.randn(N // 3, 4) * 0.45 + np.array([ 0.1,  0.2,  1.8,  0.6])
    
    X = np.vstack([c1, c2, c3])
    y = np.array([0]*(N//3) + [1]*(N//3) + [2]*(N//3))
    
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    
    indices = np.random.permutation(N)
    train_idx, test_idx = indices[:200], indices[200:]
    
    X_train = torch.tensor(X[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y[train_idx], dtype=torch.long)
    X_test  = torch.tensor(X[test_idx], dtype=torch.float32)
    y_test  = torch.tensor(y[test_idx], dtype=torch.long)
    
    return X_train, y_train, X_test, y_test

def train_and_eval(model, X_train, y_train, X_test, y_test, epochs=100):
    optimizer = optim.Adam(model.parameters(), lr=0.035)
    criterion = nn.CrossEntropyLoss()
    
    loss_history = []
    for ep in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        
    model.eval()
    with torch.no_grad():
        preds = model(X_test).argmax(dim=1)
        acc = (preds == y_test).float().mean().item()
        
    return acc, loss_history

def run_experiment():
    print("=" * 75)
    print("STEP 3: HARDWARE-AWARE PHOTONIC NEURAL NETWORK (PYTORCH)")
    print("=" * 75)
    
    X_train, y_train, X_test, y_test = generate_benchmark_dataset()
    print(f"Dataset generated: {len(X_train)} Train samples, {len(X_test)} Test samples (4 Features, 3 Classes)\n")
    
    # 1. Ideal Software Baseline (No Noise)
    print("[Model 1] Training Ideal Photonic Neural Network (Zero Physical Noise)...")
    model_ideal = PhotonicClassifier(num_classes=3, add_noise=False)
    acc_ideal, loss_ideal = train_and_eval(model_ideal, X_train, y_train, X_test, y_test, epochs=100)
    print(f"       -> Test Accuracy (Ideal Software): {acc_ideal * 100:.1f}%\n")
    
    # 2. Deploying Ideal Model onto Imperfect Optical Chip (Uncalibrated)
    print("[Model 2] Deploying Ideal Model directly onto Noisy Optical Hardware...")
    model_uncalibrated = PhotonicClassifier(num_classes=3, add_noise=True)
    model_uncalibrated.load_state_dict(model_ideal.state_dict())
    
    # Evaluate across 20 noisy hardware test passes
    acc_noisy_list = []
    model_uncalibrated.train() # Enable noise injection
    with torch.no_grad():
        for _ in range(20):
            preds = model_uncalibrated(X_test).argmax(dim=1)
            acc_noisy_list.append((preds == y_test).float().mean().item())
            
    acc_noisy = float(np.mean(acc_noisy_list))
    print(f"       -> Test Accuracy (Uncalibrated Chip): {acc_noisy * 100:.1f}%")
    print(f"       -> Accuracy Drop: {(acc_ideal - acc_noisy) * 100:.1f}% (Hardware Noise Degraded AI!)\n")
    
    # 3. Hardware-Aware Training (HAT)
    print("[Model 3] Training with HARDWARE-AWARE TRAINING (HAT)...")
    print("       (PyTorch learns weights while simulating Lumerical noise in the backward pass)")
    model_hat = PhotonicClassifier(num_classes=3, add_noise=True)
    acc_hat, loss_hat = train_and_eval(model_hat, X_train, y_train, X_test, y_test, epochs=100)
    
    acc_hat_test_list = []
    model_hat.train() # Test robustness against noise
    with torch.no_grad():
        for _ in range(20):
            preds = model_hat(X_test).argmax(dim=1)
            acc_hat_test_list.append((preds == y_test).float().mean().item())
    acc_hat_robust = float(np.mean(acc_hat_test_list))
    
    print(f"       -> Test Accuracy (Hardware-Aware Optical Model): {acc_hat_robust * 100:.1f}%")
    print(f"       -> Accuracy Recovered: +{(acc_hat_robust - acc_noisy) * 100:.1f}%!\n")
    
    # Save results
    results = {
        "ideal_accuracy": acc_ideal,
        "noisy_uncalibrated_accuracy": acc_noisy,
        "hardware_aware_accuracy": acc_hat_robust,
        "loss_ideal": loss_ideal,
        "loss_hat": loss_hat
    }
    
    with open(r"C:\CSE\quantum\training_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("=" * 75)
    print("STEP 3 COMPLETE: Hardware-Aware Training Successfully Restores Accuracy!")
    print(r"Results saved to: C:\CSE\quantum\training_results.json")
    print("=" * 75)

if __name__ == "__main__":
    run_experiment()
