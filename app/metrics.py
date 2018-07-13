import skimage
from skimage import io
import numpy as np
from skimage.util import invert
from skimage.measure import perimeter

from skimage.measure._regionprops import regionprops as regionprops_default
import functools

regionprops = functools.partial(regionprops_default, coordinates='rc')

from skimage import feature


def mean_color(img):
    # flatten image
    m = img.shape[0] * img.shape[1]
    rgb = img.reshape(m, 3).mean(axis=0)

    return rgb.tolist()


def getImageContrast(img):
    m = img.shape[0] * img.shape[1]
    lum = img.reshape(m, 3).dot(np.array([[0.2126], [0.7152], [0.0722]]))

    lmin = np.min(lum)
    lmax = np.max(lum)

    return (lmax - lmin) / (lmax + lmin)

def shannon_entropy(img):
    # ready-made method
    entropy = skimage.measure.shannon_entropy(img, base=2)
    return entropy

def find_contours(img):
    img = invert(img)
    contours = skimage.measure.find_contours(img, 0.8)
    return len(contours) / img.size

# This function takes an image and calculates the shape_index of the image which is an array
# It then returns the average of squares of the shape_index array
# from skimage import feature will be needed
def get_shape_index(img):
    shape_index = feature.shape_index(img)
    shape_index_1D = np.ravel(shape_index)
    avg_squares_shape_index = np.average(np.square(shape_index_1D))
    return avg_squares_shape_index

def solidity(img):
    return regionprops(img.astype(int))[0].solidity

def get_image_data(filename):
    data_ = {
        'file_name': [],
        'year': [],
        'mean_color': [],
        'mean_color_r': [],
        'mean_color_g': [],
        'mean_color_b': [],
        'shannon_entropy': [],
        'luminance': [],
        'contour': [],
        'solidity': [],
        'shape_index': [],
        'contrast': []}

    img = io.imread(filename)
    img_gray = skimage.color.rgb2gray(img)

    data_['file_name'].append(filename)
    data_['year'].append(filename.split('/')[-1].split('.')[0])

    rgb = mean_color(img)
    luminance = (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])

    data_['mean_color'].append(rgb)
    data_['mean_color_r'].append(rgb[0])
    data_['mean_color_g'].append(rgb[1])
    data_['mean_color_b'].append(rgb[2])
    data_['luminance'].append(luminance)
    data_['shannon_entropy'].append(shannon_entropy(img))

    data_['contour'].append(find_contours(img_gray))

    data_['shape_index'].append(get_shape_index(img_gray))
    try:
        data_['solidity'].append(solidity(img))
    except:
        data_['solidity'].append('Null')

    data_['contrast'].append(getImageContrast(img))

    return data_
