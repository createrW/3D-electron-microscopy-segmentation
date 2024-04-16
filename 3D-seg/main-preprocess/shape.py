import tifffile as tiff
import numpy as np
from skimage import io
img1= tiff.imread("../data/training.tif")
img2 = io.imread("../data/training.tif")
img_stack = img1.astype(np.float32) / 255.0

print(img2)
print(img2.shape)

import numpy as np
from skimage.transform import resize

def preprocess_volume(volume, target_shape=(128, 128, 128)):
    # Resize volume to target shape
    volume_resized = resize(volume, target_shape, anti_aliasing=True)

    # Normalize volume
    volume_normalized = (volume_resized - np.min(volume_resized)) / (np.max(volume_resized) - np.min(volume_resized))

    return volume_normalized

# Example usage
preprocessed_volume = preprocess_volume(img2)

print(preprocessed_volume)
print(np.array(preprocessed_volume).shape)