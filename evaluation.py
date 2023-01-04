from segmenter import Segmenter
import torch
from main import train_val_loader, train_val_dataset
import numpy as np
import torchio as tio
import os
from tqdm import tqdm

os.environ['KMP_DUPLICATE_LIB_OK']='True'
model = Segmenter.load_from_checkpoint("./weights/epoch=97-step=25773.ckpt")
model = model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
train_dataset, val_dataset = train_val_dataset()

def make_prediction(IDX):
    mask = val_dataset[IDX]["Label"]["data"]
    imgs = val_dataset[IDX]["CT"]["data"]
    grid_sampler = tio.inference.GridSampler(val_dataset[IDX], 96, (8, 8, 8))
    aggregator = tio.inference.GridAggregator(grid_sampler)
    patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=4)

    with torch.no_grad():
        for patches_batch in patch_loader:
            input_tensor = patches_batch['CT']["data"].to(device)  # Get batch of patches
            locations = patches_batch[tio.LOCATION]  # Get locations of patches
            pred = model(input_tensor)  # Compute prediction
            aggregator.add_batch(pred, locations)

    output_tensor = aggregator.get_output_tensor()
    pred = output_tensor.argmax(0)
    return pred, mask, imgs
def dice_coef_trial(mask_,mask_label_):
    numerator = (mask_*mask_label_).ravel()
    sum_nume = numerator.sum()
    val_nume = sum_nume.item()
    denominator = (mask_*mask_+mask_label_*mask_label_).ravel()
    sum_deno = denominator.sum()
    val_deno = sum_deno.item()
    try:
        return 2*val_nume/val_deno
    except:
        return 999
dice_list = []
for i in range(0,len(val_dataset)):
    pred,mask,imgs = make_prediction(i)
    mask_ = np.where(pred >= 1, 1, 0)
    mask_label = np.where(mask[0, :, :, :] >= 1, 1, 0)
    dice = dice_coef_trial(mask_,mask_label)
    if dice < 1:
        dice_list.append(dice)
print("Dice similarity index of the model is: ", sum(dice_list)/len(dice_list))


