from pathlib import Path
import argparse

from skimage import io
from skimage.exposure import equalize_adapthist
import matplotlib.pyplot as plt
from skimage.morphology import disk, erosion
from skimage.transform import resize
import numpy as np

from preprocessing import enhance_image


base_path = Path('/home/morano/SW/AV/_Data')
target_path = Path('/home/morano/SW/AV/_Processed_Data')
target_path.mkdir(exist_ok=True)




def debug_sample(img_enh_clahe, mask_enh, av3, fn, tgt_path):
    mask_enh_er = erosion(mask_enh, footprint=disk(5))
    mask_enh_border = mask_enh - mask_enh_er
    if av3 is None:
        av3_deb = mask_enh_border.copy()
    else:
        av3_deb = av3.copy()
        av3_deb[mask_enh_border > 0] = [255, 255, 255]
    plt.imshow(img_enh_clahe)
    plt.imshow(av3_deb, alpha=0.3)
    plt.title(fn.name)
    # plt.show()
    debug_fn = tgt_path / '__debug' / fn.with_suffix('.jpg').name
    debug_fn.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(debug_fn, bbox_inches='tight', dpi=400)


def save_sample(img_enh_clahe, mask_enh, av3, fn, tgt_path, cfp=None):
    fn = fn.with_suffix('.png')
    tgt_img_fn = tgt_path / 'img' / fn.name
    tgt_img_fn.parent.mkdir(parents=True, exist_ok=True)
    io.imsave(tgt_img_fn, img_enh_clahe)
    tgt_mask_fn = tgt_path / 'mask' / fn.name
    tgt_mask_fn.parent.mkdir(parents=True, exist_ok=True)
    io.imsave(tgt_mask_fn, mask_enh)
    if av3 is not None:
        tgt_av3_fn = tgt_path / 'av3' / fn.name
        tgt_av3_fn.parent.mkdir(parents=True, exist_ok=True)
        io.imsave(tgt_av3_fn, av3)
    if cfp is not None:
        tgt_cfp_fn = tgt_path / 'cfp' / fn.name
        tgt_cfp_fn.parent.mkdir(parents=True, exist_ok=True)
        io.imsave(tgt_cfp_fn, cfp)


def hrf():
    """Process HRF dataset."""
    subsets = ['train', 'test']
    tgt_path = target_path / 'HRF'
    for subset in subsets:
        path = base_path / f'HRF_AVLabel_191219/{subset}/images'
        for fn in sorted(path.iterdir()):
            mask = io.imread(str(fn).replace('images', 'masks'))
            av3 = io.imread(str(fn).replace('images', 'av3'))
            print(f'Processing {fn}')
            img_enh, mask_enh = enhance_image(fn, mask, int_format=True)
            # equalize_adapthist returns a float64 image in [0, 1]
            img_enh_clahe = equalize_adapthist(img_enh, clip_limit=0.01)
            img_enh_clahe = (img_enh_clahe * 255).astype('uint8')
            # plt.imshow(img_enh_clahe)
            # plt.title(fn.name)
            # plt.show()
            save_sample(img_enh_clahe, mask_enh, av3, fn, tgt_path)
            # DEBUG
            debug_sample(img_enh_clahe, mask_enh, av3, fn, tgt_path)


def gave():
    """Process GAVE dataset."""
    subsets = ['training', 'validation']
    for subset in subsets:
        path = base_path / f'GAVE/{subset}/images'
        tgt_path = target_path / 'GAVE' / subset
        for fn in sorted(path.iterdir()):
            print(f'Processing {fn}')
            mask = io.imread(str(fn).replace('images', 'masks'))
            img_enh, mask_enh = enhance_image(fn, mask, int_format=True)
            img_enh_clahe = equalize_adapthist(img_enh, clip_limit=0.01)
            img_enh_clahe = (img_enh_clahe * 255).astype('uint8')
            av3 = None
            if subset == 'training':
                # NOTE: Annotations are only available for training set
                av = io.imread(str(fn).replace('images', 'av'))[..., :3].astype('uint16')
                # Convert AV to AV3 format
                # In the AV image, the red channel is arteries, green
                #  channel is crossings, and blue channel is veins.
                # We need to convert it to a three-channel image
                # with red channel for arteries, green channel for
                # veins, and blue channel for vessels (arteries +
                # veins).
                # There are no unknown pixels in this dataset.
                av3 = np.zeros_like(av)
                # Arteries
                av3[..., 0] = np.clip(av[..., 0] + av[..., 1], 0, 255)
                # Veins
                av3[..., 1] = np.clip(av[..., 2] + av[..., 1], 0, 255)
                # Vessels
                av3[..., 2] = np.clip(av[..., 0] + av[..., 1] + av[..., 2], 0, 255)
                av3 = av3.astype('uint8')
            debug_sample(img_enh_clahe, mask_enh, av3, fn, tgt_path)
            save_sample(img_enh_clahe, mask_enh, av3, fn, tgt_path)


def lesav():
    """Process LES-AV dataset."""
    path = base_path / 'LES-AV_original/images'
    tgt_path = target_path / 'LES-AV'
    for fn in sorted(path.iterdir()):
        print(f'Processing {fn}')
        mask_fn = str(fn).replace('images', 'masks').replace('.png', '_mask.png')
        mask = io.imread(mask_fn)
        av_3_fn = str(fn).replace('images', 'av3').replace('.png', '_av3.png')
        av3 = io.imread(av_3_fn)
        img_enh, mask_enh = enhance_image(fn, mask, int_format=True)
        img_enh_clahe = equalize_adapthist(img_enh, clip_limit=0.01)
        img_enh_clahe = (img_enh_clahe * 255).astype('uint8')
        debug_sample(img_enh_clahe, mask_enh, av3, fn, tgt_path)
        save_sample(img_enh_clahe, mask_enh, av3, fn, tgt_path)


def rite():
    """Process RITE dataset."""
    tgt_path = target_path / 'RITE'
    subsets = ['training', 'test']
    for subset in subsets:
        for fn in sorted((base_path / f'RITE/{subset}').glob(f'??_{subset}.png')):
            print(f'Processing {fn}')
            mask_fn = str(fn).replace(f'_{subset}.png', f'_{subset}_mask.png')
            mask = io.imread(mask_fn)
            av3_fn = str(fn).replace(f'_{subset}.png', f'_av3.png')
            av3 = io.imread(av3_fn)
            img_enh, mask_enh = enhance_image(fn, mask, int_format=True)
            img_enh_clahe = equalize_adapthist(img_enh, clip_limit=0.01)
            img_enh_clahe = (img_enh_clahe * 255).astype('uint8')
            debug_sample(img_enh_clahe, mask_enh, av3, fn, tgt_path)
            save_sample(img_enh_clahe, mask_enh, av3, fn, tgt_path)


def reta_conversion(av: np.ndarray) -> np.ndarray:
    """Convert AV(Y) (RETA format) to AV3 format.
    In the AV(Y), the red channel is vessels, green channel is
    arteries, and blue channel is veins. White pixels are both
    arteries and veins at the same time. No unknown pixels.
    Vessels
    """
    av3 = np.zeros_like(av)
    # Arteries
    av3[..., 0] = av[..., 1]
    # Veins
    av3[..., 1] = av[..., 2]
    # Vessels
    av3[..., 2] = np.clip(av[..., 0] + av[..., 1] + av[..., 2], 0, 255)
    av3 = av3.astype('uint8')
    return av3


def process_reta_public_dataset(dataset_name: str):
    subsets = ['train', 'test']
    tgt_path = target_path / dataset_name
    for subset in subsets:
        path = base_path / 'RETA_public_datasets' / dataset_name / subset / 'img'
        for fn in sorted(path.iterdir()):
            print(f'Processing {fn}')
            mask_fn = str(fn).replace('img', 'mask').replace('.jpg', '_mask.png')
            mask = io.imread(mask_fn)
            av_fn = str(fn).replace('img', 'vessel').replace('.jpg', '_vessel.png')
            av = io.imread(av_fn)
            img_enh, mask_enh = enhance_image(fn, mask, int_format=True)
            img_enh_clahe = equalize_adapthist(img_enh, clip_limit=0.01)
            img_enh_clahe = (img_enh_clahe * 255).astype('uint8')
            av3 = reta_conversion(av)
            save_sample(img_enh_clahe, mask_enh, av3, fn, tgt_path)
            debug_sample(img_enh_clahe, mask_enh, av3, fn, tgt_path)


def dualmodal2019():
    process_reta_public_dataset('DualModal2019')


def afio():
    process_reta_public_dataset('AFIO')


def resize_img(img: np.ndarray, height: int = 768) -> np.ndarray:
    width = int(img.shape[1] * height / img.shape[0])
    return resize(
        img,
        (height, width),
        preserve_range=True,
        anti_aliasing=True
    ).astype('uint8')


def fundus_avseg():
    """Process Fundus-AVSeg dataset."""
    path = base_path / 'Fundus-AVSeg' / 'img'
    tgt_path = target_path / 'Fundus-AVSeg'
    for fn in sorted(path.iterdir()):
        print(f'Processing {fn}')
        av_fn = str(fn).replace('img', 'av')
        av = io.imread(av_fn) / 255.0
        img_gray = io.imread(fn).mean(axis=-1)
        mask = np.zeros_like(img_gray, dtype='uint8')
        mask[img_gray < 5] = 255  # Binarize mask
        mask = 255 - mask
        # Convert AV to AV3 format
        # In the AV image, the red channel is arteries, green
        # channel is crossings, and blue channel is veins.
        # Also, white pixels are unknown.
        av3 = np.zeros_like(av)
        unknown = av[..., 0] * av[..., 1] * av[..., 2]
        vessels = np.clip(av[..., 0] + av[..., 1] + av[..., 2], 0, 1)
        arteries = np.clip(av[..., 0] + av[..., 1], 0, 1) - unknown
        veins = np.clip(av[..., 2] + av[..., 1], 0, 1) - unknown
        av3[..., 0] = arteries
        av3[..., 1] = veins
        av3[..., 2] = vessels
        av3 = (av3 * 255).astype('uint8')
        img_enh, mask_enh = enhance_image(fn, mask, int_format=True)
        img_enh_clahe = equalize_adapthist(img_enh, clip_limit=0.01)
        img_enh_clahe = (img_enh_clahe * 255).astype('uint8')
        debug_sample(img_enh_clahe, mask_enh, av3, fn, tgt_path)
        save_sample(img_enh_clahe, mask_enh, av3, fn, tgt_path)
        # Also save the original image and mask
        original_img_fn = tgt_path / 'cfp' / fn.name
        original_img_fn.parent.mkdir(parents=True, exist_ok=True)
        cfp = io.imread(fn)
        io.imsave(original_img_fn, cfp)
        original_mask_fn = tgt_path / 'mask_or' / fn.name
        original_mask_fn.parent.mkdir(parents=True, exist_ok=True)
        io.imsave(original_mask_fn, mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('function', type=str, help='Function to execute')
    arguments = parser.parse_args()

    globals()[arguments.function]()
