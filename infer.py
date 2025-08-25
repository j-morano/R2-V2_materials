from pathlib import Path
import argparse
import json
import os
import zipfile
import gc

import numpy as np
import torch
from torch import Tensor
from torchvision import utils as vutils
from skimage import io
from skimage import io

from transformations import to_torch_tensors, pad_images_unet
from model import RRWNet



def zip_folders(folder_paths, output_zip):
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for folder in folder_paths:
            # Walk through each folder and add files to the ZIP
            for root, _dirs, files in os.walk(folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Add file to ZIP, preserving folder structure
                    arcname = os.path.relpath(file_path, os.path.dirname(folder))
                    zipf.write(file_path, arcname)


def compute_avr(image_fn):
    return 0.6741


def get_models_prediction(model, tensor: Tensor) -> Tensor:
    pred = model(tensor)
    if isinstance(pred, list):
        pred = pred[-1]
    else:
        pred = pred
    return pred


def get_predictions(model, tensor: Tensor, mask_tensor: Tensor, test_aug: bool = True, model_type: str = 'av') -> Tensor:
    original_tensor = tensor.clone()
    if original_tensor.shape[1] > 3:
        or_img_disp = torch.concat([original_tensor[0, :3], original_tensor[0, 3:]], dim=1)
    else:
        or_img_disp = original_tensor[0, :3]
    if test_aug:
        tensors = []
        # Rotate 90, 180, 270 degrees, and flip horizontally and vertically
        for angle in [0, 90, 180, 270]:
            rotated_tensor = torch.rot90(original_tensor, k=angle // 90, dims=(2, 3))
            for flip in [False, True]:
                if flip:
                    flipped_tensor = torch.flip(rotated_tensor, dims=(3,))
                else:
                    flipped_tensor = rotated_tensor.clone()
                tensors.append((flipped_tensor, angle, flip))
    else:
        tensors = [(tensor, 0, False)]
    # print(f'  Number of tensors for prediction: {len(tensors)}')
    all_preds = []
    for tensor, angle, flip in tensors:
        # fig, ax = plt.subplots(1, 4)
        # ax[0].set_title('Original Image')
        # ax[0].imshow(or_img_disp.cpu().numpy().transpose(1, 2, 0))
        # ax[1].set_title('Input Image (norm)')
        if tensor.shape[1] > 3:
            img_disp = torch.concat([tensor[0, :3], tensor[0, 3:]], dim=1)
        else:
            img_disp = tensor[0, :3]
        # print(f'  Input image shape: {img_disp.shape}')
        # ax[1].imshow(img_disp.cpu().numpy().transpose(1, 2, 0))
        pred = get_models_prediction(model, tensor)
        pred = torch.sigmoid(pred)
        # ax[2].set_title('Predicted')
        # ax[2].imshow(pred[0].cpu().numpy().transpose(1, 2, 0))
        if flip:
            pred = torch.flip(pred, dims=(3,))
        if angle > 0:
            pred = torch.rot90(pred, k=-angle // 90, dims=(2, 3))
        # ax[3].set_title('Predicted (retransformed)')
        # ax[3].imshow(pred[0].cpu().numpy().transpose(1, 2, 0))
        pred[mask_tensor < 0.5] = 0
        all_preds.append(pred)
        # plt.show()
        # plt.close()
    # Combine predictions from all augmentations by majority voting
    if len(all_preds) > 1:
        print(f'  Combining {len(all_preds)} predictions')
        a = torch.stack([x[:, 0] for x in all_preds], dim=1)
        v = torch.stack([x[:, 1] for x in all_preds], dim=1)
        bv = torch.stack([x[:, 2] for x in all_preds], dim=1)
        # NOTE: max returns (values, indices), so we take values
        #  with [0].
        if model_type == 'bv':
            a_uni = torch.mean(a, dim=1, keepdim=True)
            v_uni = torch.mean(v, dim=1, keepdim=True)
            av = (a + v).clamp(min=0, max=1)
            bv_av = torch.concatenate([bv, av], dim=1)
            print(f'  bv_av shape: {bv_av.shape}')
            bv_uni = torch.mean(bv_av, dim=1, keepdim=True)
            a_uni[a_uni > 0.5] = a.max(dim=1, keepdim=True).values[a_uni > 0.5]
            # a_uni[a_uni <= 0.5] = a.min(dim=1, keepdim=True).values[a_uni <= 0.5]
            v_uni[v_uni > 0.5] = v.max(dim=1, keepdim=True).values[v_uni > 0.5]
            # v_uni[v_uni <= 0.5] = v.min(dim=1, keepdim=True).values[v_uni <= 0.5]
            bv_uni[bv_uni > 0.5] = bv_av.max(dim=1, keepdim=True).values[bv_uni > 0.5]
            # bv_uni[bv_uni <= 0.5] = bv.min(dim=1, keepdim=True).values[bv_uni <= 0.5]
        # elif option == 1:
        #     a_uni = torch.median(a, dim=1, keepdim=True).values
        #     v_uni = torch.median(v, dim=1, keepdim=True).values
        #     bv_uni = torch.median(bv, dim=1, keepdim=True).values
        else:
            a_uni = torch.mean(a, dim=1, keepdim=True)
            v_uni = torch.mean(v, dim=1, keepdim=True)
            bv_uni = torch.mean(bv, dim=1, keepdim=True)
        pred_uni = torch.cat([a_uni, v_uni, bv_uni], dim=1)
        # Assign the maximum value to the artery channel, if the mean
        # value is greater than 0.5, otherwise assign 1 - value
        # a_uni = torch.where(a_uni > 0.5, a.max(dim=0, keepdim=True)[0], 1 - a.min(dim=0, keepdim=True)[0])
    else:
        # option = 1
        # if option == 0:
        #     a_uni = all_preds[0][:, 0:1]
        #     v_uni = all_preds[0][:, 1:2]
        #     bv_uni = all_preds[0][:, 2:3]
        #     # print(f' {a_uni.shape}, {v_uni.shape}, {bv_uni.shape}')
        #     bv_uni = torch.max(torch.stack([bv_uni, a_uni, v_uni], dim=1), dim=1).values
        #     pred_uni = torch.cat([a_uni, v_uni, bv_uni], dim=1)
        # else:
        pred_uni = all_preds[0]
    return pred_uni

def get_paths(args, use_cfp):
    images_path = Path(args.images_path)
    if args.masks_path is None:
        masks_path = images_path.parent / 'mask'
    else:
        if Path(args.masks_path).exists():
            masks_path = Path(args.masks_path)
        else:
             masks_path = images_path.parent / args.masks_path
             print(masks_path)
    cfps_path = None
    if use_cfp:
        if args.cfps_path is None:
            cfps_path = images_path.parent / 'cfp'
        else:
            cfps_path = Path(args.cfps_path)

    assert images_path.exists(), images_path
    assert masks_path.exists(), masks_path

    return images_path, masks_path, cfps_path


def main(parser):
    parser.add_argument('-t', '--model_type', type=str, choices=['av', 'bv', 'avr'], default='av')
    parser.add_argument('-i', '--images_path', type=str, required=True)
    parser.add_argument('-m', '--masks_path', type=str)
    parser.add_argument('-c', '--cfps_path', type=str, default=None)
    args = parser.parse_args()

    save_path = Path('./predictions')
    if args.model_type != 'avr':
        save_path = save_path / args.model_type
    save_path.mkdir(exist_ok=True)
    use_cfp = args.model_type == 'bv'
    # test_aug = args.model_type == 'bv'
    test_aug = True

    images_path, masks_path, cfps_path = get_paths(args, use_cfp)

    if args.model_type == 'avr':
        save_path_avr = save_path / 'Task3'
        save_path_avr.mkdir(exist_ok=True, parents=True)

        avr_file = save_path_avr / 'AVR.txt'
        if avr_file.exists():
            avr_file.unlink()
        for image_fn in sorted(images_path.iterdir()):
            avr = compute_avr(image_fn)
            with open(avr_file, 'a') as f:
                f.write(f'{image_fn.name} {avr:.4f}\n')
        print(f'AVR values saved in {avr_file}')
        exit(0)

    print('Loading model')
    checkpoint = torch.load(f'{args.model_type}.pth')

    print('Loading config')
    with open(f'{args.model_type}_config.json', 'r') as f:
        config = json.load(f)

    print('Config:')
    print(json.dumps(config, indent=4))

    # Namespace from config dict
    config = argparse.Namespace(**config)

    print('Creating model')
    model = RRWNet(
        config.in_channels,
        config.out_channels,
        config.base_channels,
        config.num_iterations
    )

    print('Loading weights')
    model.load_state_dict(checkpoint)

    model.eval()

    if torch.cuda.is_available():
        model.cuda()

    save_path_images = save_path
    save_path_images.mkdir(exist_ok=True, parents=True)

    for image_fn in sorted(images_path.iterdir()):
        mask_fn = None
        for mask_fn in masks_path.iterdir():
            if mask_fn.stem == image_fn.stem:
                break
        if mask_fn is None:
            print(f'ERROR: Mask not found for {image_fn.name}')
            exit(1)
        cfp_fn = None
        if use_cfp and cfps_path is not None and cfps_path.exists():
            for cfp_fn in cfps_path.iterdir():
                if cfp_fn.stem == image_fn.stem:
                    break
        if image_fn.is_file():
            print(f'> Processing {image_fn.name}')
            image = io.imread(image_fn)
            print(f'  Image shape: {image.shape}')
            if image.max() > 1:
                image = (image / 255.0)[..., :3]
            if cfp_fn is not None:
                cfp = (io.imread(cfp_fn) / 255.0)[..., :3]
                image = np.concatenate([image, cfp], axis=-1)
            mask = io.imread(mask_fn).astype(np.float32)
            if mask.max() > 1:
                mask = mask / 255.0
            images, paddings = pad_images_unet([image, mask], return_paddings=True)
            img = images[0]
            padding = paddings[0]
            mask = images[1]
            mask = np.stack([mask,] * 3, axis=2)
            # padding format: ((top, bottom), (left, right), (0, 0))
            tensors = to_torch_tensors([img, mask])
            tensor = tensors[0]
            mask_tensor = tensors[1]
            if torch.cuda.is_available():
                tensor = tensor.cuda()
            else:
                tensor = tensor.cpu()
            tensor = tensor.unsqueeze(0)
            mask_tensor = mask_tensor.unsqueeze(0)
            with torch.no_grad():
                pred = get_predictions(model, tensor, mask_tensor, test_aug=test_aug)
                pred = pred[:, :, padding[0][0]:-padding[0][1], padding[1][0]:-padding[1][1]]
                # Pred shape is torch.Size([1, 3, 1024, 1536])
                pred_gave = torch.zeros_like(pred)
                pred_gave[:, 0] = pred[:, 0]
                pred_gave[:, 1] = pred[:, 2]
                pred_gave[:, 2] = pred[:, 1]
                save_fn = save_path_images / Path(image_fn).name
                vutils.save_image(pred_gave, save_fn)
                gc.collect()

    print('Images saved in', save_path)


def zip_results(save_path, save_path_images, save_path_avr):
    # Compress the results as zip with the name 1315.zip
    # 1315 is the team id
    zip_file_name = '1315.zip'
    zip_file_path = save_path / zip_file_name
    zip_folders([save_path_images, save_path_avr], zip_file_path)
    print(f'Zipped results to {zip_file_path}')


def combine_results():
    save_path = Path('./predictions')
    av_images = save_path / 'av'
    bv_images = save_path / 'bv'
    save_path_12 = save_path / 'Task1_2'
    save_path_12.mkdir(exist_ok=True)

    for bv_fn in sorted(bv_images.iterdir()):
        b = io.imread(bv_fn)[..., 1]
        av = io.imread(av_images / bv_fn.name)
        a = av[..., 0]
        v = av[..., 2]
        avb = np.stack([a, b, v], axis=-1)
        # plt.imshow(avb); plt.show()
        save_fn = save_path_12 / bv_fn.name
        io.imsave(save_fn, avb)
        print(f'Saved combined image to {save_fn}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--option', type=str)
    args, unknown = parser.parse_known_args()
    if args.option == 'zip':
        save_path = Path('./predictions')
        save_path_images = save_path / 'Task1_2'
        save_path_avr = save_path / 'Task3'
        zip_results(save_path, save_path_images, save_path_avr)
    elif args.option == 'predict':
        main(parser)
    elif args.option == 'combine':
        combine_results()
    else:
        print('Please provide an option: --option predict or --option zip')
