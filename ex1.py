from common import DATA, IMAGES, load_gray, load_rgb_image, show_images_and_histograms, show_images_in_subplots
import numpy as np
import cv2

# from Assignment2
def gauss(sigma):
    radius = int(3 * sigma) # diameter is 2[3*sigma] + 1
    x = np.arange(-radius, radius + 1)
    g = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-(x**2) / (2 * sigma**2))
    return g / np.sum(g) # To nomalization 

# (b)
def gaussdx(sigma):
    radius = int(3 * sigma) # diameter is 2[3*sigma] + 1
    x = np.arange(-radius, radius + 1)
    g_dx = (-x * np.exp(-(x**2) / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma**3))
    return g_dx / np.sum(np.abs(g_dx)) # To nomalization 

def c():
    impulse = np.zeros((50, 50))
    impulse[25, 25] = 1
    sigma = 5
    # filter2D uses correlation, so reverse each 1-D kernel for convolution.
    G = np.fliplr(gauss(sigma).reshape(1, -1)) #  g(x)
    D = np.fliplr(gaussdx(sigma).reshape(1, -1)) # g'(x)
    combinations = [
        ("(I ⋆ G) ⋆ Dᵀ", G, D.T),
        ("(I ⋆ D) ⋆ Gᵀ", D, G.T),
        ("(I ⋆ G) ⋆ Gᵀ", G, G.T),
        ("(I ⋆ Gᵀ) ⋆ D", G.T, D),
        ("(I ⋆ Dᵀ) ⋆ G", D.T, G),
    ]

    images = {"I": impulse}
    for label, first, second in combinations:
        filtered = cv2.filter2D(impulse, -1, first)
        images[label] = cv2.filter2D(filtered, -1, second)
    show_images_in_subplots(images, 2, 3)

# (d)
def partial_derivatives(image, sigma):
    G = np.fliplr(gauss(sigma).reshape(1, -1))
    D = np.fliplr(gaussdx(sigma).reshape(1, -1))

    # G = g(x) / D = g'(x) / G.T = g(y) / D.T = g'(y)
    Ix = cv2.filter2D(cv2.filter2D(image, -1, G.T), -1, D) # g'(x) * [ g(y) * I ]
    Iy = cv2.filter2D(cv2.filter2D(image, -1, G), -1, D.T) # g'(y) * [ g(x) * I ]
    return Ix, Iy

# (d)
def partial_derivatives2(image, sigma):
    Ix, Iy = partial_derivatives(image, sigma)
    Ixx, Ixy = partial_derivatives(Ix, sigma)
    Iyx, Iyy = partial_derivatives(Iy, sigma)
    return Ixx, Ixy, Iyy
# (d)
def gradient_magnitude(image, sigma):
    Ix, Iy = partial_derivatives(image, sigma)
    Imag = np.sqrt(Ix**2 + Iy**2) # Intensity of 
    Idir = np.arctan2(Iy, Ix) # direction
    return Imag, Idir

def d():
    image = load_gray(IMAGES / "museum.jpg")
    sigma = 2

    Ix, Iy = partial_derivatives(image, sigma)
    Imag, Idir = gradient_magnitude(image, sigma)

    hsv = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    hsv[:, :, 0] = (Idir + np.pi) / (2 * np.pi) * 179  # Change angle [-pi, pi] to color [0, 179]
    hsv[:, :, 1] = 255  # Set color strength to the highest value
    hsv[:, :, 2] = Imag / np.max(Imag) * 255  # Change edge strength to light [0, 255]
    Idir_hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    images = {
        "I": image,
        "Ix": Ix,
        "Iy": Iy,
        "Imag": Imag,
        "Idir": Idir,
        "Idir (HSV)": Idir_hsv,
    }
    show_images_in_subplots(images, 2, 3)

# (e)
def gradient_feature(image, sigma, n_bins=8):
    Imag, Idir = gradient_magnitude(image, sigma)
    feature = []

    for row in range(n_bins):
        y1 = row * image.shape[0] // n_bins
        y2 = (row + 1) * image.shape[0] // n_bins

        for col in range(n_bins):
            x1 = col * image.shape[1] // n_bins
            x2 = (col + 1) * image.shape[1] // n_bins

            cell_magnitude = Imag[y1:y2, x1:x2]
            cell_direction = Idir[y1:y2, x1:x2]
            direction_bins = ((cell_direction + np.pi) / (2 * np.pi) * 8).astype(int)
            direction_bins[direction_bins == 8] = 7

            histogram = np.zeros(n_bins)
            for bin_index in range(n_bins):
                histogram[bin_index] = np.sum(cell_magnitude[direction_bins == bin_index])
            feature.extend(histogram)

    feature = np.array(feature)
    return feature / np.sum(feature)

# (e)
def find_similar_images(target_index, features):
    results = []
    target_feature = features[target_index]

    for index in range(len(features)):
        if index == target_index:
            continue
        distance = np.sqrt(np.sum((target_feature - features[index]) ** 2)) # compute the distance
        results.append((distance, index))

    results.sort()
    return results[:5]

def e(n_bins=8, target_name="object_05_4.png"):
    image_directory = DATA / "dataset"
    image_paths = sorted(image_directory.glob("*.png"))
    gradient_features = []

    for image_path in image_paths:
        gradient_features.append(gradient_feature(load_gray(image_path), 2, n_bins))

    gradient_features = np.array(gradient_features)
    target_index = image_paths.index(image_directory / target_name)
    results = [(0, target_index)] + find_similar_images(target_index, gradient_features)
    images = []
    histograms = []
    labels = []

    for distance, index in results:
        images.append(load_rgb_image(image_paths[index]))
        histograms.append(gradient_features[index])
        labels.append(f"{image_paths[index].name}\nL2: {distance:.3f}")

    show_images_and_histograms(images, histograms, labels)

if __name__ == "__main__":
    e()
