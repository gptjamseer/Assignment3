from pathlib import Path
from a3_utils import draw_line
import cv2
import matplotlib.pyplot as plt
import numpy as np

DATA = Path(__file__).resolve().parent
IMAGES = DATA / "images"

def load_rgb_image(path):
    image = cv2.imread(str(path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image.astype(np.float64) / 255.0

def load_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return image.astype(np.float64) / 255.0 # compute a normalized histogram of grayscale.

def show_images(images, title="", figsize=None, cmap="gray", interpolation=None):
    figure = plt.figure(figsize=figsize)
    plt.suptitle(title)
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i], cmap=cmap, interpolation=interpolation)
    return figure

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

def show_figure(figure, caption="", margin=0, caption_y=0.05):
    figure.tight_layout(rect=(0, margin, 1, 1))
    if caption:
        figure.text(0.5, caption_y, caption, ha="center", fontfamily="serif")
    plt.show()


def plot_hough_lines(axis, shape, rhos, thetas, peaks, color="red", linewidth=0.5):
    plt.sca(axis)
    for row, col in peaks:
        draw_line(rhos[row], thetas[col], *shape[:2], clr=color, linewidth=linewidth)


def plot_hough_column(axes, title, image, votes, rhos, thetas, peaks, theta_step=100):
    top, bottom = axes
    top.imshow(votes, cmap="viridis")
    top.set_title(title)
    top.set_xticks(range(0, len(thetas), theta_step))
    top.set_yticks(range(0, len(rhos), 100))
    bottom.imshow(image, cmap="gray")
    plot_hough_lines(bottom, image.shape, rhos, thetas, peaks, linewidth=1)
