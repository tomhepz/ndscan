"""
This script simulates an ensemble of site-resolved Rabi oscillations.

Setup
-----
- There are many sites.
- Each site has a slightly different Rabi frequency.
- At each time t and each site i, the true excitation probability p_i(t)
  is sampled using a finite number of yes/no shots.
- From those shots we estimate:
      p_hat_i(t)   = estimated probability at site i
      sigma_i(t)   = measurement error on that estimate
- Averaging across sites produces an ensemble-averaged oscillation that
  dephases faster than any individual site, because the site frequencies
  are slightly different.

Main statistical issue
----------------------
At a given time t, the site estimates differ for two different reasons:

1) Measurement noise:
   Each site estimate is noisy because it comes from finitely many yes/no shots.
   This gives a per-site measurement uncertainty sigma_i.

2) Real site-to-site variation:
   Even with infinite shots, different sites may have different true values.
   We call the true site-to-site standard deviation tau.

These are different objects and answer different questions.

Useful mental model
-------------------
For one time point, write

    y_i = theta_i + epsilon_i

where
    y_i       = measured value at site i
    theta_i   = true value at site i
    epsilon_i = measurement noise at site i

and
    Var(epsilon_i) = sigma_i^2
    SD(theta_i across sites) = tau

So:
- sigma describes uncertainty of a single site estimate due to finite shots
- tau describes real spatial / temporal inhomogeneity across sites

Important distinction: mean uncertainty vs site spread
------------------------------------------------------
There are several different "bands" one can draw, and they do NOT mean the same thing.

1) Pooled / same-distribution error band
   Assume all sites share one common underlying probability at each time.
   Then pool all yes/no shots across sites and compute one binomial error bar.

   Meaning:
   "If all sites were really measuring the same p(t), how well do I know it?"

   This is the smallest band.
   It is an uncertainty on a common mean under the assumption tau = 0.

2) Mean-over-sites uncertainty band
   Allow sites to have different true values, but still ask for the mean across sites.

   For equal per-site uncertainty sigma and N sites:
       SE(mean over sites) ~= sqrt((sigma^2 + tau^2) / N)

   Meaning:
   "How well do I know the average over sites?"

   This is still an error-on-the-mean quantity.
   It shrinks like 1/sqrt(N).

3) Site-to-site spread band
   Show the width of the true site distribution itself:
       site spread ~= tau

   Meaning:
   "How different are the sites from one another?"

   This is NOT an uncertainty on the mean.
   It does NOT shrink like 1/sqrt(N).
   In the dephasing problem, this is often the more physically meaningful band.

4) Observed spread of measured site values
   If you look at the raw scatter of measured site points across sites, that width is
   approximately

       sqrt(tau^2 + sigma^2)

   because it contains both real site differences and measurement noise.

   This quantity has a floor of about sigma if measurement noise is large.
   Therefore raw site scatter can overstate true site-to-site variation when tau is small.

Quadrature rules
----------------
Independent variance contributions add in quadrature, but only after putting them on the
scale of the quantity you are trying to describe.

Examples:

- Width of a single measured site value relative to the global mean:
      sqrt(tau^2 + sigma^2)

- Error on the mean across N sites:
      sqrt(tau^2 / N + sigma^2 / N)
    = sqrt(tau^2 + sigma^2) / sqrt(N)

- True spread across sites:
      tau
  (do NOT add sigma here, because sigma is measurement noise, not physical spread)

Estimating tau from data
------------------------
At a fixed time, if the measured site estimates have observed sample variance s_obs^2,
and the average measurement variance is mean(sigma_i^2), then a simple estimator of the
true between-site variance is

    tau^2_hat = max(0, s_obs^2 - mean(sigma_i^2))

This subtracts measurement noise from the observed site-to-site scatter.

Why the ensemble average dephases
---------------------------------
Each site oscillates with a slightly different frequency.
At early times the sites are nearly in phase, so the site spread tau is small.
At later times the phases fan out, so the site spread grows.
The ensemble average then appears to decay / dephase because peaks and troughs from
different sites cancel.

This means:
- the average oscillation can become flat
- the pooled error bar on the average can still be small
- while the site-to-site spread can be large

So a flat average does NOT imply all sites are individually flat.

About the slider / middle-ground band
-------------------------------------
The script includes a display slider lambda in [0, 1].

This is not a uniquely "correct" statistical confidence interval.
It is a practical visualization dial that interpolates between:

    lambda = 0:
        same-distribution / pooled-error style view

    lambda = 1:
        full site-spread style view

A simple display construction is

    display_band^2 = (1 - lambda) * err_same^2 + lambda * tau^2

where
    err_same = pooled / same-distribution error on the mean
    tau      = estimated true site-to-site spread

Interpretation:
- small lambda emphasizes precision of the combined average
- large lambda emphasizes real inhomogeneity across sites

When the middle ground matters
------------------------------
If tau << sigma, then site-to-site heterogeneity is small compared with measurement noise.

In that regime:
- the pooled mean error and mean-over-sites error are very similar
- the distinction between "same distribution" and "independent mean" is often minor
- raw site scatter may still look large because of measurement noise
- a corrected estimate of tau can be much smaller than the raw scatter

So the middle ground matters most when:
- tau is comparable to or larger than sigma, OR
- you want to avoid mistaking measurement noise for real site variation

What this script plots
----------------------
The plotting version of this script shows:

- A central line:
      measured mean across sites

- Thin error bars:
      pooled / same-distribution uncertainty on the combined mean

- Optional dashed lines:
      uncertainty on the mean across sites when heterogeneity is allowed

- Wide shaded band:
      estimated site-to-site spread (+/- tau)

- Slider band:
      an interpolated display band between pooled-error and site-spread views

Short takeaway
--------------
Thin band  = how well the average is known.
Wide band  = how different the sites are from one another.

Those are different quantities.
In the dephasing example, the wide band is often the more physically meaningful one.
"""
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Simulate Rabi oscillations at many sites
# Each site has a slightly different Rabi frequency,
# so averaging over sites causes ensemble dephasing.
# At each (site, time), we estimate p from yes/no shots.
# ------------------------------------------------------------

rng = np.random.default_rng(4)

# -----------------------------
# User-tunable parameters
# -----------------------------
n_sites = 48
n_times = 180
n_shots_per_site = 40

t_max = 15.0
t = np.linspace(0.0, t_max, n_times)

omega0 = 2.0 * np.pi * 0.55       # mean Rabi frequency
rel_freq_spread = 0.015           # site-to-site fractional spread
T_single = 14.0                   # slow intrinsic single-site decay
amp_mean = 0.92                   # oscillation contrast
amp_spread = 0.03                 # small site-to-site contrast variation

lambda_slider = 0.55              # 0 -> same-distribution style band
                                  # 1 -> full site-spread band

# -----------------------------
# True site curves
# -----------------------------
omega_sites = omega0 * (1.0 + rng.normal(0.0, rel_freq_spread, size=n_sites))
amp_sites = np.clip(
    amp_mean + rng.normal(0.0, amp_spread, size=n_sites),
    0.75, 1.0
)

envelope = np.exp(-t / T_single)

# True probability at each site/time
p_true_sites = 0.5 * (
    1.0 + amp_sites[:, None] * envelope[None, :] * np.cos(omega_sites[:, None] * t[None, :])
)
p_true_sites = np.clip(p_true_sites, 0.0, 1.0)

# True average over sites: dephases faster because frequencies differ slightly
p_true_avg = np.mean(p_true_sites, axis=0)

# -----------------------------
# Binomial sampling per site/time
# -----------------------------
y = rng.binomial(n_shots_per_site, p_true_sites)
p_hat_sites = y / n_shots_per_site

# Slightly stabilized binomial SE per site/time
# (avoids zero error at p=0 or 1)
p_tilde_sites = (y + 0.5) / (n_shots_per_site + 1.0)
p_err_sites = np.sqrt(
    p_tilde_sites * (1.0 - p_tilde_sites) / (n_shots_per_site + 2.0)
)

# -----------------------------
# Combined mean across sites
# With equal shots/site this equals pooled counts.
# -----------------------------
p_mean = np.mean(p_hat_sites, axis=0)

# Same-distribution assumption:
# pool all Bernoulli shots at a given time as if they came from one common p(t)
y_tot = np.sum(y, axis=0)
n_tot = n_sites * n_shots_per_site
p_same = y_tot / n_tot

p_tilde_same = (y_tot + 0.5) / (n_tot + 1.0)
err_same = np.sqrt(
    p_tilde_same * (1.0 - p_tilde_same) / (n_tot + 2.0)
)

# -----------------------------
# Estimate site-to-site spread
# -----------------------------
# Observed variance across sites
obs_var = np.var(p_hat_sites, axis=0, ddof=1)

# Mean measurement variance across sites
mean_meas_var = np.mean(p_err_sites**2, axis=0)

# Estimated intrinsic between-site variance
tau2 = np.maximum(0.0, obs_var - mean_meas_var)
tau = np.sqrt(tau2)

# Optional: uncertainty on the mean across sites under "different sites, but mean still matters"
err_mean = np.sqrt(np.sum(p_err_sites**2, axis=0) / (n_sites**2) + tau2 / n_sites)

# -----------------------------
# Display band with a slider
# This is a display interpolation, not a sacred CI.
#
# lambda=0: same-distribution style thin band
# lambda=1: full site-spread band
# -----------------------------
band_display = np.sqrt((1.0 - lambda_slider) * err_same**2 + lambda_slider * tau2)

# -----------------------------
# Plot
# -----------------------------
fig, axes = plt.subplots(2, 1, figsize=(10.5, 9), sharex=True)

# -----------------------------
# Top panel: a few site curves + true average
# -----------------------------
ax = axes[0]

# Show some individual true site curves
for i in range(min(n_sites, 10)):
    ax.plot(t, p_true_sites[i], lw=1.0, alpha=0.35)

ax.plot(t, p_true_avg, lw=3, label="True average over sites")
ax.set_ylabel("Probability")
ax.set_title("Slightly different site frequencies produce ensemble dephasing")
ax.set_ylim(-0.03, 1.03)
ax.grid(alpha=0.25)
ax.legend()

# -----------------------------
# Bottom panel:
# - central line = measured mean across sites
# - thin error bars = uncertainty on the combined mean
# - shaded band = estimated spread across sites
# - optional dashed band = SE of mean under random-effects view
# -----------------------------
ax = axes[1]

# Full site-spread band (what you care about if sites are genuinely different)
ax.fill_between(
    t,
    p_mean - tau,
    p_mean + tau,
    alpha=0.18,
    label="Estimated site-to-site spread (±tau)"
)

# Slider band: interpolates between same-distribution error and full spread
ax.fill_between(
    t,
    p_mean - band_display,
    p_mean + band_display,
    alpha=0.28,
    label=f"Display band with slider λ={lambda_slider:.2f}"
)

# Optional dashed line showing uncertainty on the mean under random-effects view
ax.plot(t, p_mean + err_mean, "--", lw=1.2, alpha=0.9, label="Mean uncertainty (random-effects)")
ax.plot(t, p_mean - err_mean, "--", lw=1.2, alpha=0.9)

# Thin error bars for same-distribution pooling
every = 6
ax.errorbar(
    t, p_same, yerr=err_same,
    fmt="o-", ms=2.8, lw=1.2, capsize=2, errorevery=every,
    label="Combined mean ± pooled binomial error"
)

# Central measured mean
ax.plot(t, p_mean, lw=2.4, label="Measured mean across sites")

# True average for reference
ax.plot(t, p_true_avg, lw=2.0, alpha=0.9, label="True average over sites")

ax.set_xlabel("Time")
ax.set_ylabel("Probability")
ax.set_title("Thin bars = error on mean; wide band = spread across sites")
ax.set_ylim(-0.03, 1.03)
ax.grid(alpha=0.25)
ax.legend(loc="upper right", fontsize=9)

plt.tight_layout()
plt.show()

# -----------------------------
# A few printed summaries at late time
# -----------------------------
idx = int(0.75 * n_times)
print(f"time = {t[idx]:.3f}")
print(f"measured mean across sites          = {p_mean[idx]:.5f}")
print(f"same-distribution pooled error      = {err_same[idx]:.5f}")
print(f"random-effects mean uncertainty     = {err_mean[idx]:.5f}")
print(f"estimated site-to-site spread (tau) = {tau[idx]:.5f}")
print(f"display band width                  = {band_display[idx]:.5f}")