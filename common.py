from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np

DATA = Path(__file__).resolve().parent
IMAGES = DATA / "images"

def load_rgb_image(path):
    image = cv2.imread(str(path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image.astype(np.float64) / 255.0

def to_grayscale_average(image):
    float_image = image.astype(np.float64)
    red = float_image[:, :, 0]
    green = float_image[:, :, 1]
    blue = float_image[:, :, 2]
    return (red + green + blue) / 3

def load_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return image.astype(np.float64) / 255.0 # compute a normalized histogram of grayscale.

def myhist(image, n_bins):
    histogram = np.zeros(n_bins)
    pixels = image.reshape(-1)
    for value in pixels:
        bin_index = int(value * n_bins)
        if bin_index == n_bins:
            bin_index = n_bins - 1
        histogram[bin_index] += 1
    return histogram / np.sum(histogram)

def show_images(images, title):
    plt.figure()
    plt.suptitle(title)
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i], cmap="gray")

# ex1 (c)
def show_images_in_subplots(images, rows, columns):
    plt.figure(figsize=(3 * columns, 3 * rows))
    for i, (label, image) in enumerate(images.items(), start=1):
        if image is None:
            continue
        plt.subplot(rows, columns, i)
        # 영상별 최솟값과 최댓값을 흑백 기준으로 삼아 작은 미분 응답의 양수와 음수를 모두 보여 준다.
        plt.imshow(image, cmap="gray")
        plt.title(label)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.show()

def show_images_and_histograms(images, histograms, labels):
    columns = len(images)
    plt.figure(figsize=(3 * columns, 6))

    for i in range(columns):
        plt.subplot(2, columns, i+1)
        plt.imshow(images[i])
        plt.title(labels[i])

        plt.subplot(2, columns, i+1+columns)
        plt.plot(histograms[i])
    plt.tight_layout()
    plt.show()

def show_signals(signals, title):
    plt.figure()

    for label, signal in signals.items():
        positions, values = signal
        plt.plot(positions, values, label=label)

    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def show_signals_in_subplots(signals, rows, columns, ylim=None):
    plt.figure(figsize=(3 * columns, 3 * rows))
    for i, (label, signal) in enumerate(signals.items(), start=1):
        if signal is None:
            continue
        plt.subplot(rows, columns, i)
        plt.plot(signal)
        plt.title(label)
        if ylim is not None:
            plt.ylim(*ylim)
    plt.tight_layout()
    plt.show()

# ex3 (e)
def show_signals_best5(signals, rows, columns):
    plt.figure()

    for i, (label, signal) in enumerate(signals.items(), start=1):
        best5 = np.argsort(signal)[:5]

        plt.subplot(rows, columns, i)
        plt.plot(signal)
        plt.plot(best5, signal[best5], "ro", fillstyle="none")
        plt.title(label)

    plt.tight_layout()
    plt.show()
