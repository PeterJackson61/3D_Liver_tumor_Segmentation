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
    smooth = 1
    numerator = (mask_*mask_label_).ravel()
    sum_nume = numerator.sum()
    val_nume = sum_nume.item()+smooth
    denominator = (mask_*mask_+mask_label_*mask_label_).ravel()
    sum_deno = denominator.sum()
    val_deno = sum_deno.item()+smooth
    dice_coef = 2*val_nume/val_deno
    if dice_coef == 2:
        return 1
    else:
        return dice_coef
dice_list_tumor = []
dice_list_liver = []
for i in tqdm(range(0,len(val_dataset))):
    pred,mask,imgs = make_prediction(i)
    mask_tumor = np.where(pred > 1, 1, 0)
    mask_liver = np.where(pred==1,1,0)
    mask_label_tumor = np.where(mask[0, :, :, :] > 1, 1, 0)
    mask_label_liver = np.where(mask==1,1,0)
    dice = dice_coef_trial(mask_tumor,mask_label_tumor)
    dice_liver = dice_coef_trial(mask_liver, mask_label_liver)
    dice_list_tumor.append(dice)
    dice_list_liver.append(dice_liver)
print("Dice similarity index of the model with tumor is: ", sum(dice_list_tumor)/len(dice_list_tumor))
for dice in dice_list_tumor:
    print(dice)
print("Dice similarity index of the model with liver is: ", sum(dice_list_liver)/len(dice_list_liver))
for dice in dice_list_liver:
    print(dice)


