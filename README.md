# PureSymbolic

Ahhh — now I see exactly what you mean 😎

You’re not talking about the *musical instrument architecture*, but the **mathematical unification** of synthesis under *a single difference equation* — where oscillators, partials, and even filters all come from manipulating **delay terms** (powers of ( Z^{-1} )).

Let’s dive into that properly, because what you’re describing is *the purest form of a digital synthesizer kernel*.

---

## 🧩 1. Everything Is a Difference Equation

At its core, every linear time-invariant (LTI) discrete-time system is:

[
y[n] = \sum_{k=1}^{P} a_k,y[n-k] + \sum_{m=0}^{Q} b_m,x[n-m]
]

This single form can produce:

* **oscillators** (feedback with complex conjugate poles),
* **filters** (finite or infinite impulse response),
* **additive synthesis** (sum of many sinusoids),
* and even **comb & resonator networks** (delayed copies).

So — literally everything can live in one unified equation where **partials = weighted feedback paths** and **delays = harmonics or filter resonances**.

---

## ⚙️ 2. Example: A Single-Line “Universal Oscillator”

Here’s the essence of your idea in NumPy — a *difference equation that generates all harmonics through delayed feedback*:

```python
import numpy as np
import matplotlib.pyplot as plt

fs = 48000
N = 48000
f0 = 220
omega = 2 * np.pi * f0 / fs

# "partials" as delayed feedback taps
partials = [0, 1, 2, 4, 8, 16]      # Z^-k delays
amps = [1, 0.8, 0.5, 0.3, 0.2, 0.1] # weighting coefficients

y = np.zeros(N)
y[0] = 1

for n in range(1, N):
    y[n] = sum(amps[k] * np.cos(omega * n - 2*np.pi*f0*(partials[k]/fs))
               for k in range(len(partials)))

plt.plot(np.arange(N)/fs, y, alpha=0.8)
plt.xlim(0, 0.01)
plt.title("Single-Line Partial+Delay Oscillator")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()
```

That’s effectively doing:
[
y[n] = \sum_k a_k \cos(\omega n - \omega d_k)
]
which in the ( Z )-domain corresponds to:
[
Y(z) = X(z) \sum_k a_k z^{-d_k}
]

That’s one **single equation** combining:

* Additive synthesis (the sum of partials),
* Subtractive synthesis (the delays form a filter kernel),
* And temporal structure (delays = ( Z^{-d_k} )).

---

## 🔮 3. If You Go Fully Recursive

You can go even further — build oscillation from *feedback only*:

[
y[n] = 2\cos(\omega_0),y[n-1] - y[n-2] + \sum_k g_k,y[n-d_k]
]

That’s a **second-order resonator** + arbitrary feedback network = entire subtractive system on one line.

Example:

```python
y = np.zeros(N)
y[0], y[1] = 1, np.cos(omega)

for n in range(2, N):
    y[n] = (2*np.cos(omega))*y[n-1] - y[n-2] \
         + 0.3*y[n-20] - 0.2*y[n-50] + 0.1*y[n-100]
```

That one recursion line contains:

* a harmonic oscillator (`2cos(ω)y[n-1] - y[n-2]`),
* plus comb-style filtering (`y[n-k]`),
* plus additive partial weighting (the coefficients).

So yes — **a single difference equation line** can literally define an entire *polyphonic spectrum generator*.

---

## 🧠 4. Conceptually

* Each **delay term** ( Z^{-k} ) = phase-shifted copy (partial, harmonic, or comb notch).
* Each **coefficient** = amplitude / filter weight.
* Each **feedback loop** = resonance or pole pair.

Your insight is spot on:
👉 *Subtractive, additive, resonant, and feedback synthesis are just special cases of a single polynomial in ( Z^{-1} ).*

[
Y(z) = \frac{B(z)}{A(z)}X(z)
]
where ( B(z) = \sum b_m z^{-m} ) (partials/delays)
and ( A(z) = 1 - \sum a_k z^{-k} ) (feedback).

---

heck yes—let’s turn that idea into a tiny **general-purpose synthesizer kernel** where you can type your (B(z)) and (A(z)) (partials + delays in the numerator; feedback delays in the denominator) and it just runs the **single difference equation**:

[
y[n] ;=; \sum_{m=0}^{M} b_m,x[n-m];-;\sum_{k=1}^{K} a_k,y[n-k],\quad a_0=1
]

Below is a compact, self-contained snippet that:

* implements the kernel,
* gives helper builders for resonators and combs (all in (z^{-1}) delays),
* shows minimal examples (impulse-excited resonator, combs, and “oscillator on one line” feel).

```python
import numpy as np
import matplotlib.pyplot as plt

class SynthKernel:
    """
    General difference-equation kernel:
      y[n] = sum b[m]*x[n-m] - sum a[k]*y[n-k], with a[0] == 1

    Tips:
    - FIR (no feedback): a = [1]
    - IIR/resonators/filters: fill a[1:], b[*] as you like
    - To "excite" a filter: use an impulse x[0]=1 or noise, or a step.
    """
    def __init__(self, b, a):
        a = np.asarray(a, dtype=float)
        if len(a) == 0 or abs(a[0] - 1.0) > 1e-12:
            raise ValueError("a[0] must be 1.")
        self.b = np.asarray(b, dtype=float)
        self.a = a

    def process(self, x):
        x = np.asarray(x, dtype=float)
        y = np.zeros_like(x)
        M = len(self.b) - 1
        K = len(self.a) - 1
        # single-line recursion (conceptually): y[n] = Σ b*x - Σ a*y
        for n in range(len(x)):
            y_n = 0.0
            # feedforward (partials/delays)
            for m in range(M + 1):
                if n - m >= 0:
                    y_n += self.b[m] * x[n - m]
            # feedback (resonances / poles)
            for k in range(1, K + 1):
                if n - k >= 0:
                    y_n -= self.a[k] * y[n - k]
            y[n] = y_n
        return y

    # ---------- Helpers (all delays are powers of z^{-1}) ----------
    @staticmethod
    def resonator_f0_r(f0, fs, r=0.999, gain=1.0):
        """
        2nd-order resonator: H(z) = gain / (1 - 2 r cos ω0 z^-1 + r^2 z^-2)
        - r in (0,1): decay; r=1: pure (numerically fragile) oscillator
        Use input x as an impulse to excite; or noise for sustained texture.
        """
        ω0 = 2*np.pi*f0/fs
        a = [1.0, -2*r*np.cos(ω0), r*r]
        b = [gain]
        return b, a

    @staticmethod
    def resonator_f0_Q(f0, fs, Q=50, gain=1.0):
        """
        Convert Q -> r with BW = f0/Q and r = exp(-π*BW/fs)
        """
        BW = f0 / max(Q, 1e-9)
        r = np.exp(-np.pi * BW / fs)
        return SynthKernel.resonator_f0_r(f0, fs, r=r, gain=gain)

    @staticmethod
    def comb_feedforward(D, g=0.5):
        """
        H(z) = 1 + g z^-D  (notches periodically spaced)
        """
        b = np.zeros(D + 1); b[0] = 1.0; b[-1] = g
        a = [1.0]
        return b, a

    @staticmethod
    def comb_feedback(D, g=0.5):
        """
        H(z) = 1 / (1 - g z^-D)  (peaks periodically spaced)
        """
        b = [1.0]
        a = np.zeros(D + 1); a[0] = 1.0; a[-1] = -g  # since y = x + g*y[n-D] -> move to LHS
        return b, a

    @staticmethod
    def delay_sum(delays, gains):
        """
        Pure partial/delay sum: H(z) = Σ gains[i] z^-delays[i]
        """
        L = max(delays) if delays else 0
        b = np.zeros(L + 1)
        for d, g in zip(delays, gains):
            b[d] += g
        a = [1.0]
        return b, a

# --------------------------- Examples ---------------------------

fs = 48000
N  = 48000
t  = np.arange(N)/fs

def plot_sig(y, title, zoom_s=0.02):
    plt.figure(figsize=(10,3.2))
    plt.plot(t, y, lw=1.0)
    plt.title(title); plt.xlabel("Time (s)"); plt.ylabel("Amplitude"); plt.grid(True)
    plt.xlim(0, zoom_s)
    plt.tight_layout(); plt.show()

# 1) Impulse-excited resonator (sine-like tone from one line of recursion)
x = np.zeros(N); x[0] = 1.0  # impulse
b, a = SynthKernel.resonator_f0_r(f0=440, fs=fs, r=0.9995, gain=1.0)
kernel = SynthKernel(b, a)
y = kernel.process(x)
plot_sig(y, "Impulse-Excited Resonator @ 440 Hz (decaying)")

# 2) “Steady-ish” tone: drive resonator with white noise (subtractive flavor)
x_noise = np.random.uniform(-1, 1, N) * 0.02
y2 = kernel.process(x_noise)
plot_sig(y2, "Noise-Driven Resonator (narrowband tone)")

# 3) Feedforward comb (partials via delays in B(z))
b_ff, a_ff = SynthKernel.comb_feedforward(D=24, g=0.7)
comb_ff = SynthKernel(b_ff, a_ff)
y_comb = comb_ff.process(y2)  # filter the resonator output again
plot_sig(y_comb, "Feedforward Comb on Resonator Output")

# 4) Feedback comb (peaky resonance train)
b_fb, a_fb = SynthKernel.comb_feedback(D=24, g=0.6)
comb_fb = SynthKernel(b_fb, a_fb)
y_comb2 = comb_fb.process(x_noise)
plot_sig(y_comb2, "Feedback Comb (noise -> harmonic-ish ringing)")

# 5) Pure delay-sum “partials” (all in B(z); A(z)=1)
delays = [0, 10, 25, 55, 100]
gains  = [1.0, 0.6, 0.4, 0.25, 0.15]
b_ps, a_ps = SynthKernel.delay_sum(delays, gains)
partials = SynthKernel(b_ps, a_ps).process(np.sin(2*np.pi*220*t))  # treat sin as "excitation"
plot_sig(partials, "Delay Sum on 220 Hz Sine (partials via B(z))")
```

### How this maps to your “single line” idea

* **Additive/partials** → put gains at specific **feedforward delays** in `b` (that’s literally ( \sum b_m z^{-m} )).
* **Subtractive/filters/resonators** → put weights at **feedback delays** in `a[1:]` (that’s ( 1 - \sum a_k z^{-k} )).
* **One recursion** (inside `process`) does it all: oscillator-like behavior, combs, filter banks—just by choosing coefficients.

### Quick recipes (copy/paste coefficients)

* **Pure FIR partial sum:** `a=[1]`, `b` has nonzero taps at your desired delays.
* **2nd-order resonator at f0:** `a=[1, -2*r*cos(ω0), r**2]`, `b=[gain]`, excite with impulse.
* **Feedforward comb:** `b[0]=1, b[D]=g`, `a=[1]`.
* **Feedback comb:** `b=[1]`, `a[D] = -g`.

If you want, I can add convenience creators for:

* biquad from **(f0, Q, type)** (lowpass, bandpass, etc.),
* an **LFO** to modulate a delay (phaser/flanger) by writing `D(t)` and re-sampling,
* a compact **audio writer** (WAV) to hear the results.
