from typing import Dict
import glob
import cv2
import numpy as np
from tqdm import tqdm
from skimage import io
from skimage.transform import resize
def load_volume(
        path: str,
) -> Dict:
    dataset = sorted(glob.glob(path))

    volume = None
    target = None
    print(dataset)
    for z, path in enumerate(tqdm(dataset)):
        mask = (cv2.imread(path, 0) > 127.0) * 1.0
        # path = path.replace(
        #     "labels",
        #     "images",
        # ).replace(".png", ".jp2")
        # if "/kidney_3_dense/" in path:
        #     path = path.replace("kidney_3_dense", "kidney_3_sparse")
        path = path.replace("masks","images")
        image = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        image = np.array(image, dtype=np.uint16)
        if volume is None:
            volume = np.zeros((len(dataset), *image.shape[-2:]), dtype=np.uint16)
            target = np.zeros((len(dataset), *mask.shape[-2:]), dtype=np.uint8)
        volume[z] = image
        target[z] = mask

    return {"volume": volume, "target": target}


# train = load_volume("../data/train/masks/**.jpg")
# print(train["volume"].shape)


def train_pre(crop_shape = (128, 128, 128), volume =  None, target = None, mode = 'train'):
    if mode == "train":
        sample_non_empty_mask = bool(np.random.binomial(n=1, p=0.5))

        sample_new_mask = True
        # Random Crop
        while sample_new_mask:
            start_x = np.random.randint(0, volume.shape[0] - crop_shape[0])
            start_y = np.random.randint(0, volume.shape[1] - crop_shape[1])
            start_z = np.random.randint(0, volume.shape[2] - crop_shape[2])

            volume_crop = volume[
                          start_x: start_x + crop_shape[0],
                          start_y: start_y + crop_shape[1],
                          start_z: start_z + crop_shape[2],
                          ].copy()

            target_crop = target[
                          start_x: start_x + crop_shape[0],
                          start_y: start_y + crop_shape[1],
                          start_z: start_z + crop_shape[2],
                          ].copy()

            sample_new_mask = sample_non_empty_mask and target_crop.sum() == 0

            volume_crop, target_crop = random_augmentation(
                volume_crop.copy(), target_crop.copy()
            )
    else:
        start_x = np.random.randint(0, volume.shape[0] - crop_shape[0])
        start_y = np.random.randint(0, volume.shape[1] - crop_shape[1])
        start_z = np.random.randint(0, volume.shape[2] - crop_shape[2])

        volume_crop = volume[
                      start_x: start_x + crop_shape[0],
                      start_y: start_y + crop_shape[1],
                      start_z: start_z + crop_shape[2],
                      ].copy()

        target_crop = target[
                      start_x: start_x + crop_shape[0],
                      start_y: start_y + crop_shape[1],
                      start_z: start_z + crop_shape[2],
                      ].copy()

    volume_resized = resize(volume, crop_shape, anti_aliasing=True)
    xmin = np.min(volume_resized)
    xmax = np.max(volume_resized)
    volume_crop = normilize(volume_crop, xmin=xmin, xmax=xmax)

    volume_crop, target_crop = np.ascontiguousarray(
            volume_crop
    ), np.ascontiguousarray(target_crop)

    return {
            "volume": np.expand_dims(volume_crop, axis=0),
            "target": np.expand_dims(target_crop, axis=0),
            # "id": random_id,
    }

def random_augmentation(volume, mask):
        # Random rotation (90-degree increments)
    rotation_axes = [(0, 1), (0, 2), (1, 2)]
    axis = np.random.choice([0, 1, 2])
    angle = np.random.choice([0, 90, 180, 270])
    volume = np.rot90(volume, angle // 90, axes=rotation_axes[axis])
    mask = np.rot90(mask, angle // 90, axes=rotation_axes[axis])

    # Random flips
    if np.random.rand() > 0.5:
        volume = np.flip(volume, axis=0)
        mask = np.flip(mask, axis=0)
    if np.random.rand() > 0.5:
        volume = np.flip(volume, axis=1)
        mask = np.flip(mask, axis=1)
    if np.random.rand() > 0.5:
        volume = np.flip(volume, axis=2)
        mask = np.flip(mask, axis=2)

        # if np.random.rand() > 0.5:
        #     original_shape = volume.shape
        #
        #     zoom_factor = 1 + np.random.uniform(-0.2, 0.2)
        #     volume = zoom(
        #         volume, zoom_factor, order=1
        #     )  # Using bilinear interpolation for volume
        #     mask = zoom(
        #         mask, zoom_factor, order=0
        #     )  # Using nearest-neighbor interpolation for mask
        #
        #     # If zooming out, crop to original size. If zooming in, pad with zeros to original size.
        #     volume, mask = self.match_shape(volume, original_shape), self.match_shape(
        #         mask, original_shape
        #     )

        # if np.random.rand() > 0.3:
        #     brightness_factor = np.random.uniform(
        #         0.8, 1.1
        #     )  # Adjust this range as needed
        #     volume = (volume * brightness_factor).astype(np.uint16)
        #     volume = np.clip(volume, 0, 65535)

    return volume, mask



def normilize(image: np.ndarray, xmin: float, xmax: float) -> np.ndarray:
    image = (image - xmin) / (xmax - xmin)
    image = np.clip(image, 0, 1)
    return image.astype(np.float32)

# train_af = train_pre(volume = train["volume"], target = train["target"], mode='train')
# print(train_af["volume"].shape)
# print(train_af["target"].shape)


if __name__ == "__main__":
    train = load_volume("../data/train/masks/**.jpg")
    print(train["volume"].shape)
    train_af = train_pre(volume=train["volume"], target=train["target"], mode='train')
    print(train_af["volume"].shape)
    print(train_af["target"].shape)