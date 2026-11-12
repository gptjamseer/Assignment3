import numpy as np

# ex1 (b)
def simple_convolution(signal, kernel):
    signal = np.array(signal, dtype=float)
    kernel = np.array(kernel, dtype=float)
    result = np.zeros(len(signal) - len(kernel) + 1)

    for p in range(len(result)):
        result[p] = np.sum( signal[p:p + len(kernel)] * kernel[::-1] )
    return result

# ex1 (c) / add zero padding
def improved_simple_convolution(signal, kernel):
    signal = np.array(signal, dtype=float)
    kernel = np.array(kernel, dtype=float)

    kernel_radius = len(kernel) // 2
    padded_signal = np.pad(signal, kernel_radius) # add "0" till radius
    result = np.zeros(len(signal))

    for p in range(len(result)):
        result[p] = np.sum(
            padded_signal[p:p + len(kernel)] * kernel[::-1]
        )
    return result

def gauss(sigma):
    radius = int(3 * sigma) # diameter is 2[3*sigma] + 1
    x = np.arange(-radius, radius + 1)
    g = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-(x**2) / (2 * sigma**2))
    return g / np.sum(g) # For nomalization 

def gaussdx(sigma):
    if not np.isfinite(sigma) or sigma < 1 / 3:
        raise ValueError("sigma must be finite and give a kernel radius of at least 1")
    radius = int(3 * sigma)
    x = np.arange(-radius, radius + 1)
    derivative = -x * np.exp(-x**2 / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma**3)
    return derivative / np.sum(np.abs(derivative))

def b():
    from a3_utils import read_data
    from common import DATA, show_signals

    signal = read_data(DATA / "signal.txt")
    kernel = read_data(DATA / "kernel.txt")
    result = simple_convolution(signal, kernel)
    N = len(kernel) // 2

    print("(b) Kernel sum:", np.sum(kernel))
    # "label" : (x,y) loop
    show_signals({
        "signal": (np.arange(len(signal)), signal),
        "kernel": (np.arange(len(kernel)), kernel),
        "result": (np.arange(N, len(signal) - N), result),
    }, "Exercise 1(b)")

def c():
    from a3_utils import read_data
    from common import DATA, show_signals

    signal = read_data(DATA / "signal.txt")
    kernel = read_data(DATA / "kernel.txt")

    result = improved_simple_convolution(signal, kernel)

    show_signals({
        "signal": (np.arange(len(signal)), signal),
        "kernel": (np.arange(len(kernel)), kernel),
        "result": (np.arange(len(result)), result),
    }, "Exercise 1(c)")

def d():
    from common import show_signals

    kernels = {}
    for sigma in (0.5, 1, 2, 3, 4):
        kernel = gauss(sigma)
        radius = len(kernel) // 2
        x = np.arange(-radius, radius + 1)
        kernels[f"sigma = {sigma}"] = (x, kernel)

    show_signals(kernels, "Exercise 1(d)")

def e():
    from a3_utils import read_data
    from common import DATA, show_signals_in_subplots

    signal = read_data(DATA / "signal.txt")
    k1 = gauss(2)
    k2 = np.array([0.1, 0.6, 0.4])

    signal_k1 = improved_simple_convolution(signal, k1)
    k1_then_k2 = improved_simple_convolution(signal_k1, k2)

    signal_k2 = improved_simple_convolution(signal, k2)
    k2_then_k1 = improved_simple_convolution(signal_k2, k1)

    k3 = improved_simple_convolution(k1, k2)
    combined = improved_simple_convolution(signal, k3)

    print("(signal * k1) * k2 = (signal * k2) * k1:", np.all(np.isclose(k1_then_k2, k2_then_k1)))
    print("(signal * k2) * k1 = signal * k3       :", np.all(np.isclose(k1_then_k2, combined)))

    results = {
        "signal": signal,
        "(signal * k1) * k2": k1_then_k2,
        "(signal * k2) * k1": k2_then_k1,
        "signal * k3": combined,
    }

    show_signals_in_subplots(results, 4, 1)
