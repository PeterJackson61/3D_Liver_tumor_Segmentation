from main import train_val_loader, train_val_dataset
import numpy as np
import matplotlib.pyplot as plt
import os
import torchio as tio
import torch
from segmenter import Segmenter
from tqdm import tqdm

train_dataset, val_dataset = train_val_dataset()
IDX = np.random.randint(0,len(val_dataset)+1)
os.environ['KMP_DUPLICATE_LIB_OK']='True'

model = Segmenter.load_from_checkpoint("./weights/epoch=97-step=25773.ckpt")
model = model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

def make_prediction_2(IDX):
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
    return output_tensor,pred, mask, imgs

output_tensor, pred, mask, imgs = make_prediction_2(IDX)
print(imgs.shape)
for i in tqdm(range(50, output_tensor.shape[3]-50, 2)):  # axial view
    plt.subplot(1,2,1)
    plt.imshow(imgs[0,:,:,i], cmap="bone")
    mask_ = np.ma.masked_where(pred[:,:,i]==0, pred[:,:,i])
    plt.imshow(mask_,cmap="autumn")
    plt.subplot(1,2,2)
    plt.imshow(imgs[0,:,:,i], cmap="bone")
    label_mask = np.ma.masked_where(mask[0,:,:,i]==0, mask[0,:,:,i])
    plt.imshow(label_mask, cmap="jet")
    plt.show()
    plt.axis("off")