import subprocess
from typing import List, Optional
from pathlib import Path
import argparse

from skimage import io
from skimage.exposure import equalize_adapthist

import process_datasets as prda
import preprocessing as pre
import infer



########################################################################
# Constants


DATA_PATH = Path('./__Data/GAVE_testing/')
CWD = '/home/morano/SW/AV/R2-V2.git/main/verification_code'
RUN = 'testing_mask_or'


########################################################################
# 1

def preprocess():
    print('>>> Preprocess')
    tgt_path = DATA_PATH.parent / 'GAVE_testing_preprocessed'
    tgt_path.mkdir(exist_ok=True)
    for image_fn in sorted((DATA_PATH / 'images').iterdir()):
        print(image_fn)
        print(f'Processing {image_fn}')
        mask = io.imread(str(image_fn).replace('images', 'masks'))
        img_enh, mask_enh = pre.enhance_image(image_fn, mask, int_format=True)
        img_enh_clahe = equalize_adapthist(img_enh, clip_limit=0.01)
        img_enh_clahe = (img_enh_clahe * 255).astype('uint8')
        av3 = None
        prda.debug_sample(img_enh_clahe, mask_enh, av3, image_fn, tgt_path)
        prda.save_sample(img_enh_clahe, mask_enh, av3, image_fn, tgt_path)


########################################################################
# 2

def _run(
    command: List,
    cwd: Optional[str]=None,
):
    subprocess.call(
        command,
        cwd=cwd,
    )


def _get_paths():
    base_path = DATA_PATH.parent / 'GAVE_testing_preprocessed'
    img_path = base_path / 'img'
    if 'mask_or' in RUN:
        mask_path = base_path / 'mask_or'
    else:
        mask_path = base_path / 'mask'
    return img_path, mask_path


# Run the following in any order

def infer_bv():
    img_path, mask_path = _get_paths()
    cmd = f'python -m infer -o predict -i {img_path} -m {mask_path} -t bv -r {RUN}'
    _run(cmd.split(), cwd=CWD)


def infer_av():
    img_path, mask_path = _get_paths()
    cmd = f'python -m infer -o predict -i {img_path} -m {mask_path} -t av -r {RUN}'
    _run(cmd.split(), cwd=CWD)


def infer_avr():
    img_path, _ = _get_paths()
    cmd = f'python -m infer -o predict -i {img_path} -t avr -r {RUN}'
    _run(cmd.split(), cwd=CWD)


########################################################################
# 3


def combine():
    """Combine the AV and BV results."""
    infer.combine_results(f'./__predictions/{RUN}/')


########################################################################
# 4


def compress():
    save_path = Path(f'./__predictions/{RUN}')
    save_path_images = save_path / 'Task1_2'
    save_path_avr = save_path / 'Task3'
    zip_file_name='R2-V2.zip'  # NOTE: use name of the team
    infer.zip_results(
        save_path,
        save_path_images,
        save_path_avr,
        zip_file_name,
    )


########################################################################
# Main


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('option', type=str)
    args = parser.parse_args()

    globals()[args.option]()

