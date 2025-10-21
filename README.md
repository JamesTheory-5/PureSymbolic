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

---

ohhh that’s a gorgeous idea. let’s make a tiny “symbolic” builder that takes a **Fourier spec** (partials) and optional **delay specs**, then **unrolls** them into a single difference-equation kernel:

[
H(z)=\frac{B(z)}{A(z)}\quad\Rightarrow\quad
y[n]=\sum_{m} b_m,x[n-m]-\sum_{k\ge1} a_k,y[n-k],\ a_0=1.
]

Key trick: any periodic waveform of period (N=\frac{f_s}{f_0}) can be created by convolving an **impulse train** (period (N)) with **one period** of the waveform. In z-domain that’s:

[
H(z)=\frac{B(z)}{1-z^{-N}} \quad\text{with } B(z)=\sum_{n=0}^{N-1} h[n],z^{-n},
]

where (h[n]) is one period, which we can **synthesize from the Fourier series**. From there we can symbolically multiply in any extra feedforward/feedback **delay polynomials** you want.

Here’s a compact, self-contained implementation:

```python
import numpy as np

# ---------- Kernel runner ----------
class Kernel:
    def __init__(self, b, a):
        a = np.asarray(a, float)
        if a[0] != 1.0:
            raise ValueError("a[0] must be 1.")
        self.b = np.asarray(b, float)
        self.a = a

    def process(self, x):
        x = np.asarray(x, float)
        y = np.zeros_like(x)
        M = len(self.b) - 1
        K = len(self.a) - 1
        for n in range(len(x)):
            acc = 0.0
            # feedforward
            for m in range(M + 1):
                if n - m >= 0:
                    acc += self.b[m] * x[n - m]
            # feedback
            for k in range(1, K + 1):
                if n - k >= 0:
                    acc -= self.a[k] * y[n - k]
            y[n] = acc
        return y

# ---------- "Symbolic" builder: Fourier + delays -> (b, a) ----------
def fourier_period_from_partials(N, partials):
    """
    Build one period h[0..N-1] from Fourier partials.
    partials: iterable of (k, A, phi) with harmonic index k >= 0, amplitude A, phase phi (radians).
      k=0 term is DC if provided.
    """
    n = np.arange(N)
    h = np.zeros(N, float)
    for k, A, phi in partials:
        if k == 0:
            h += A * np.ones(N)
        else:
            h += A * np.cos(2*np.pi*k*n/N + phi)
    return h

def poly_from_delay_sum(delays, gains, Lmin=0):
    """
    Build a FIR polynomial P(z) = sum_i gains[i] z^{-delays[i]}.
    Ensures length >= Lmin by padding zeros to the right.
    """
    length = max([0] + delays + [Lmin-1]) + 1
    p = np.zeros(length, float)
    for d, g in zip(delays, gains):
        p[d] += g
    return p

def poly_convolve(p, q):  # multiply polynomials in z^{-1}
    return np.convolve(p, q)

def poly_add_pad(p, q):
    L = max(len(p), len(q))
    pp = np.pad(p, (0, L-len(p)))
    qq = np.pad(q, (0, L-len(q)))
    return pp + qq

def build_kernel_from_spec(fs, f0, partials, ff_delays=None, ff_gains=None, fb_delays=None, fb_gains=None):
    """
    fs, f0: sample rate & fundamental
    partials: list of (k, A, phi) Fourier terms at harmonics k * f0
    ff_delays/gains: extra feedforward delay sum to multiply into B(z)
    fb_delays/gains: extra feedback delay sum to multiply into A(z) (on the RHS, i.e., denominator)
                     Interpreted as A(z) *= (1 - sum g_i z^{-D_i})  (so g>0 means +g*y[n-D] on RHS)
    Returns b, a with a[0]==1.
    """
    # 1) period length
    N = int(round(fs / f0))
    if N <= 0:
        raise ValueError("Invalid N from fs/f0")

    # 2) build one period from Fourier series -> B0(z)
    h = fourier_period_from_partials(N, partials)
    B = h.copy()  # FIR taps: 0..N-1

    # 3) base denominator is 1 - z^{-N} (impulse-train feedback)
    A = np.zeros(N + 1, float)
    A[0], A[-1] = 1.0, -1.0

    # 4) multiply in extra feedforward delays (partials via delays)
    if ff_delays and ff_gains:
        FF = poly_from_delay_sum(ff_delays, ff_gains)
        B = poly_convolve(B, FF)

    # 5) multiply in extra feedback delays  A(z) *= (1 - Σ g z^{-D})
    if fb_delays and fb_gains:
        # Build (1 - Σ g z^{-D})
        FB = poly_from_delay_sum([0] + list(fb_delays), [1.0] + list(-np.asarray(fb_gains)))
        A = poly_convolve(A, FB)

    # 6) normalize (optional): keep unity peak gain for typical excitations
    scale = np.max(np.abs(B)) or 1.0
    B = B / scale
    return B, A

# ---------- convenience: classic shapes via analytic Fourier series ----------
def triangle_partials(Nharm=25, phase0=0.0, amp=1.0):
    # odd harmonics with 1/k^2 and alternating sign
    parts = [(0, 0.0, 0.0)]
    sign = 1.0
    k_used = 0
    k = 1
    while k_used < Nharm:
        if k % 2 == 1:
            A = amp * (8 / (np.pi**2)) * sign / (k**2)
            parts.append((k, A, phase0))
            sign *= -1.0
            k_used += 1
        k += 1
    return parts

def saw_partials(Nharm=50, phase0=0.0, amp=1.0):
    # all harmonics with 1/k
    parts = [(0, 0.0, 0.0)]
    for k in range(1, Nharm+1):
        A = amp * (2/np.pi) * (1.0/k)
        parts.append((k, A, phase0 - np.pi/2))  # phase shift to match standard saw
    return parts

def square_partials(Nharm=50, phase0=0.0, amp=1.0):
    # odd harmonics with 1/k
    parts = [(0, 0.0, 0.0)]
    for k in range(1, 2*Nharm, 2):
        A = amp * (4/np.pi) * (1.0/k)
        parts.append((k, A, phase0))
    return parts

# ---------- demo usage ----------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fs, f0, dur = 48000, 220, 0.03
    N = int(fs * dur)
    t = np.arange(N) / fs

    # 1) Build a triangle *symbolically* from its Fourier partials; unroll to kernel:
    parts = triangle_partials(Nharm=25, amp=1.0)
    b, a = build_kernel_from_spec(
        fs, f0, parts,
        ff_delays=None, ff_gains=None,     # extra feedforward tap polynomial (optional)
        fb_delays=None, fb_gains=None      # extra feedback tap polynomial (optional)
    )
    k_tri = Kernel(b, a)

    # excite with a single impulse to generate the periodic waveform via 1/(1 - z^-N)
    x = np.zeros(int(fs*dur)); x[0] = 1.0
    y_tri = k_tri.process(x)

    # 2) Now "symbolically" add a comb notch: multiply B(z) by (1 + g z^-D)
    b2, a2 = build_kernel_from_spec(
        fs, f0, parts,
        ff_delays=[0, 24], ff_gains=[1.0, 0.6],  # H_ff(z)=1+0.6 z^-24 (notch/comb flavor)
        fb_delays=None, fb_gains=None
    )
    y_tri_comb = Kernel(b2, a2).process(x)

    # 3) Or add feedback peaks: A(z) *= (1 - g z^-D)
    b3, a3 = build_kernel_from_spec(
        fs, f0, parts,
        fb_delays=[32], fb_gains=[0.7]           # H(z) /= (1 - 0.7 z^-32)
    )
    y_tri_peaky = Kernel(b3, a3).process(x)

    # plot
    def plot(t, y, title, xlim=0.02):
        plt.figure(figsize=(9,3))
        plt.plot(t[:int(fs*xlim)], y[:int(fs*xlim)], lw=1.0)
        plt.title(title); plt.xlabel("Time (s)"); plt.ylabel("Amplitude"); plt.grid(True)
        plt.tight_layout()

    plot(t, y_tri, "Triangle via Symbolic Fourier → Kernel (impulse excited)")
    plot(t, y_tri_comb, "Triangle with Feedforward Comb (symbolic delay multiply)")
    plot(t, y_tri_peaky, "Triangle with Feedback Peaks (symbolic delay multiply)")
    plt.show()
```

### what this gives you

* **Symbolic in:** Fourier partials `(k, A, φ)` and delay primitives.
* **Out:** a **single** kernel ((b, a)) that, when fed an **impulse**, produces your periodic waveform; and when you multiply in extra delay polynomials you get subtractive tweaks (notches/peaks) **without leaving the one-line difference equation** world.
* The *denominator* starts as (1 - z^{-N}) (periodic repetition), then you can multiply more feedback polynomials to add resonances.
* The *numerator* is literally one period’s samples (B(z)), optionally multiplied by any feedforward delay sums (partials-as-delays).

If you want, I can add:

* a tiny DSL like:
  `Wave(f0=220).fourier("triangle", Nharm=25).ff("+0 +0.6@24").fb("-0.7@32").build()`
  that compiles straight to `(b, a)`,
* or a `to_wav()` helper to listen to the results.
