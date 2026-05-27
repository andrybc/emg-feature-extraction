import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------
# PARAMETERS
# These define the shape of your simulated recording session
# -----------------------------------------------------------------------

fs = 1000           # Sampling rate in Hz (1000 samples per second)
duration = 4.0      # Total recording duration in seconds
n_channels = 8      # Number of EMG electrodes (channels) on the armband
n_samples = int(fs * duration)  # Total number of samples: 4000

# -----------------------------------------------------------------------
# TIME AXIS
# np.linspace(start, stop, num) creates num evenly spaced values
# We create 4000 time points from 0 to 4 seconds
# endpoint=False means we stop just before 4.0, not including it
# This is standard for discrete signal processing
# -----------------------------------------------------------------------

t = np.linspace(0, duration, n_samples, endpoint=False)

# -----------------------------------------------------------------------
# AMPLITUDE ENVELOPE
# This controls when the muscle is "active" versus "at rest"
# We use a smooth sigmoid-based on/off pattern
# Seconds 0-1: rest, Seconds 1-3: contraction, Seconds 3-4: rest
# -----------------------------------------------------------------------

def smooth_step(t, t_start, t_end, rise=0.1):
    """
    Creates a smooth on/off envelope using a logistic (sigmoid) function.
    
    Parameters
    ----------
    t       : np.ndarray  -- time axis
    t_start : float       -- time when activation begins
    t_end   : float       -- time when activation ends
    rise    : float       -- controls how quickly the envelope rises/falls
                            smaller = sharper transition
    
    Returns
    -------
    envelope : np.ndarray -- values between 0 (rest) and 1 (full activation)
    """
    # Sigmoid rising edge: smoothly goes from 0 to 1 around t_start
    rise_edge = 1 / (1 + np.exp(-(t - t_start) / rise))
    
    # Sigmoid falling edge: smoothly goes from 1 to 0 around t_end
    fall_edge = 1 / (1 + np.exp(-(t_end - t) / rise))
    
    # Multiply them: only high where BOTH are high (between t_start and t_end)
    return rise_edge * fall_edge

# Build the envelope: muscle is active between t=1.0 and t=3.0
envelope = smooth_step(t, t_start=1.0, t_end=3.0, rise=0.08)

# Scale the envelope: EMG amplitude during contraction ~ 1.0 mV peak
# At rest, the noise floor is much smaller ~ 0.05 mV
rest_noise_level = 0.05
contraction_amplitude = 1.0

# Final amplitude at each time point
amplitude = rest_noise_level + (contraction_amplitude - rest_noise_level) * envelope

# -----------------------------------------------------------------------
# GENERATE MULTI-CHANNEL EMG
# Each channel gets its own independent noise, but shares the same envelope
# shape=(n_samples, n_channels) means rows are time steps, columns are channels
# This is the standard convention in EMG research
# -----------------------------------------------------------------------

np.random.seed(42)  # Seed for reproducibility: same result every run

# np.random.randn returns standard normal random numbers
# We multiply each sample by the amplitude at that time point
# amplitude[:, np.newaxis] reshapes (4000,) to (4000,1) so it broadcasts
# across all 8 channels correctly
emg_raw = np.random.randn(n_samples, n_channels) * amplitude[:, np.newaxis]

# Add small per-channel scaling variation to simulate different electrode contacts
# Each channel gets a random scaling factor between 0.7 and 1.3
channel_gains = np.random.uniform(0.7, 1.3, size=n_channels)
emg_raw = emg_raw * channel_gains[np.newaxis, :]  # broadcast across time axis

print(f"EMG array shape: {emg_raw.shape}")
print(f"  {emg_raw.shape[0]} samples (time steps)")
print(f"  {emg_raw.shape[1]} channels (electrodes)")
print(f"Sampling rate: {fs} Hz")
print(f"Duration: {duration} seconds")
print(f"Time resolution: {1/fs*1000:.2f} ms per sample")

# -----------------------------------------------------------------------
# PLOTTING
# Good visualization is a core research skill
# -----------------------------------------------------------------------

fig, axes = plt.subplots(
    nrows=n_channels,   # One row per channel
    ncols=1,            # One column
    figsize=(12, 10),   # Width=12 inches, Height=10 inches
    sharex=True         # All subplots share the same x-axis (time)
)

# Color map: use a perceptually uniform colormap for channel colors
colors = plt.cm.tab10(np.linspace(0, 0.8, n_channels))

for ch_idx in range(n_channels):
    ax = axes[ch_idx]
    
    # Plot the raw EMG for this channel
    ax.plot(
        t,                        # x-axis: time in seconds
        emg_raw[:, ch_idx],       # y-axis: voltage for channel ch_idx
        color=colors[ch_idx],     # different color per channel
        linewidth=0.6,            # thin line to show detail
        alpha=0.85                # slight transparency
    )
    
    # Label each channel on the y-axis
    ax.set_ylabel(f"CH {ch_idx + 1}\n(mV)", fontsize=8, rotation=0, labelpad=40)
    
    # Add a horizontal line at zero for reference
    ax.axhline(0, color='gray', linewidth=0.4, linestyle='--')
    
    # Draw vertical shaded region showing activation period
    ax.axvspan(1.0, 3.0, alpha=0.08, color='green', label='Contraction' if ch_idx == 0 else "")
    
    # Clean up tick marks
    ax.tick_params(axis='y', labelsize=7)
    ax.set_ylim(-2.0, 2.0)  # Fixed y-axis scale for comparison across channels

# Only show x-axis label on the bottom plot
axes[-1].set_xlabel("Time (seconds)", fontsize=11)

# Overall title
fig.suptitle(
    "Simulated 8-Channel Surface EMG\n(Gaussian noise with smooth activation envelope)",
    fontsize=13,
    fontweight='bold',
    y=1.01
)

# Add a legend on the first plot
axes[0].legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig("../data/simulated/emg_raw_8ch.png", dpi=150, bbox_inches='tight')
plt.show()

print("\nPlot saved to data/simulated/emg_raw_8ch.png")