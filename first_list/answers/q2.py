# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Tuple, Dict

# %%
def get_intensities(image: np.array) -> Dict[int, int]:
    """
    """
    new_intensities = [0 for i in range(256)]

    m, n = image.shape

    for i in range(m):
        for j in range(n):
            new_intensities[image[i][j]] += 1

    return new_intensities


def image_histogram(image: np.array, ax: Axes = None) -> None:
    """
    Plota um histograma de uma imagem 'uint8' informada como
    um np.array.
    """
    histogram = {i: 0 for i in range(256)}

    for row in image:
        for pixel in row:
            histogram[pixel] += 1

    if ax is not None:
        plt.bar(histogram.keys(), histogram.values())
    else:
        ax.bar(histogram.keys(), histogram.values())
    plt.show()



def plot_histogram_comparision(
    im0: np.array,
    im1: np.array,
    titles: Tuple[str, str] = ('', '')) -> None:
    """
    Plota um subplot com as duas imagens informadas. O parâmetro
    de títulos é opcional.
    """
    histogram_im0 = {i: 0 for i in range(256)}

    for row in im0:
        for pixel in row:
            histogram_im0[pixel] += 1

    histogram_im1 = {i: 0 for i in range(256)}

    for row in im1:
        for pixel in row:
            histogram_im1[pixel] += 1

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].bar(histogram_im0.keys(), histogram_im0.values())
    axs[0].set_title(titles[0])
    axs[1].bar(histogram_im1.keys(), histogram_im1.values())
    axs[1].set_title(titles[1])
    plt.show()


def plot_comparision(
    im0: np.array,
    im1: np.array,
    titles: Tuple[str, str] = ('', '')) -> None:
    """
    Plota um subplot com as duas imagens informadas. O parâmetro
    de títulos é opcional.
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(im0, cmap='gray', vmin=0, vmax=255)
    axs[0].set_title(titles[0])
    axs[1].imshow(im1, cmap='gray', vmin=0, vmax=255)
    axs[1].set_title(titles[1])
    plt.show()

# %%
image = plt.imread('../images/Fig3.15(a).jpg').astype('uint8')

#
fig, axs = plt.subplots(1, 2, figsize=(15, 5))
axs[0].imshow(image, cmap='gray', vmin=0, vmax=255)
image_histogram(image, axs[1])
plt.show()

# %%
def equalize_image(image: np.array) -> np.array:
    """
    """
    M, N = image.shape
    intensities = get_intensities(image)
    probabilities = {i: n/(M*N) for i, n in enumerate(intensities)}
    probabilities_list = list(probabilities.values())

    #
    equalized_intensities = {
    i: round(255 * sum(probabilities_list[:i]))
    for i in range(256)}

    #
    equalized_image = image.copy()

    for i in range(M):
        for j in range(N):
            equalized_image[i, j] = equalized_intensities[equalized_image[i, j]]

    return equalized_image

# %%
equalized_image = equalize_image(image)

plot_histogram_comparision(image, equalized_image, (
    'Original', 'Equalizado'))

plot_comparision(image, equalized_image, (
    'Original', 'Equalizada'))

# %%
train_image = plt.imread('../images/train.jpg').astype('uint8')
equalized_train_image = equalize_image(train_image)

plot_histogram_comparision(train_image, equalized_train_image, (
    'Original', 'Equalizado'))

plot_comparision(train_image, equalized_train_image, (
    'Original', 'Equalizada'))

# %% [markdown]
# ### Equalização de Histograma Local

# %%
def get_local_intensities(image: np.array, size: int = 49) -> Dict[int, int]:
    """
    """
    new_intensities = {}

    m, n = image.shape

    for i in range(m):
        for j in range(n):
            if image[i][j] in new_intensities:
                new_intensities[image[i][j]] += 1
            else:
                new_intensities[image[i][j]] = 1

    return new_intensities


def local_equalize_image(image: np.array, mask_size: int = 7) -> np.array:
    """
    """
    from tqdm.autonotebook import tqdm

    M, N = image.shape
    equalized_image = image.copy()
    half_mask = mask_size//2

    for i in tqdm(range(half_mask, M - half_mask)):
        for j in range(half_mask, N - half_mask):
            intensities = get_intensities(image[i:i+mask_size, j:j+mask_size])
            probabilities = {i: n/(mask_size**2) for i, n in enumerate(intensities)}
            
            #
            probabilities_keys = list(probabilities.keys())
            probabilities_values = list(probabilities.values())
            accumulated_probabilities = [sum(probabilities_values[:idx + 1])
                for idx in range(len(probabilities_values))]

            #
            index = probabilities_keys.index(image[i+half_mask, j+half_mask])
            equalized_image[i+half_mask, j+half_mask] = round(255 * accumulated_probabilities[index])

    return equalized_image

# %%
equalized_train_image = local_equalize_image(train_image, 7)

plot_histogram_comparision(train_image, equalized_train_image, (
    'Original', 'Equalizado'))

plot_comparision(train_image, equalized_train_image, (
    'Original', 'Equalizada pelo histograma'))


# %%
equalized_image = local_equalize_image(image, 3)

plot_histogram_comparision(image, equalized_image, (
    'Original', 'Equalizado'))

plot_comparision(image, equalized_image, (
    'Original', 'Equalizada pelo histograma'))



