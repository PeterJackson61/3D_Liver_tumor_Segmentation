### Medical background 
* Liver cancer is a rare cancer, but high death rate (20.8% 5-Year Relative Survival) [link](https://seer.cancer.gov/statfacts/html/livibd.html#:~:text=Rate%20of%20New%20Cases%20and%20Deaths%20per%20100%2C000%3A,and%20based%20on%202015%E2%80%932019%20cases%20and%202016%E2%80%932020%20deaths.). 
* Coutries with highest liver cancer rate are Asian and developing countries, including Vietnam. [link](https://www.wcrf.org/cancer-trends/liver-cancer-statistics/) 
* Automatic segmentation has some advantages:
  + Reduce the probability of missing a tumor 
  + Provide estimation of the size/volume of the tumor directly as the tumor volume is an important criterion in tumor staging. This will free the radiologist from segmenting the whole tumor. They only have to make corrections to tackle this task 
### Data visualization
* Using celluloid and HTML from IPython.display
* The tumor will be marked as yellow and the liver as purple
### 3D Data for the full information of the tumor 
* Medical segmentation 
  + Decathlon [link data](http://medicaldecathlon.com/)
* 3D-data: 131 full body CTs of varying shape 
  + The original dataset has been reduced its size (all axis were halved) for training speed-up
  + The data have a volume shape (256x256xValue) (Value of the 3rd dimension can vary between data)
  + Greatly increase the information content as the network sees multiple slices at once (3D convolution)
  + The memory problem from 3D data can be solved by patches (2 along each axis), implemented by torchio
### Support library: Torchio
* Torchio for loading, preprocessing and patching 3D medical images 
* Torchio - Training flow
  + Convert all niftis to subjects and store them in a list 
  + Create a tio.SubjectsDataset based on the subject list
  + Define a sampler to sample patches
  + Create tio.Queue for efficient data loading 
  + Create pytorch DataLoader based on Queue and proceed
*  Torchio - Inference flow (after training the model)
  + Define GridSampler to split the subject into patches
  + Define GridAggregator which merges the predicted segmentations back together
  + Compute prediction and perform aggregation
### Preprocessing
* Crop/Pad all volumns to shape (256x256x200)
* Standardize pixel into (-1,1) range
* Use 105 subjects for training, rest for validation
### Dataset (tio.SubjectSDataset)
* Load a subjects with corresponding segmentation
* Randomscale and rotation only by tio.RandomAffine(scales=(0.9, 1.1), degrees=(-10,10)) [link](https://pytorch.org/vision/stable/generated/torchvision.transforms.RandomAffine.html)
* Sample patches of size 96x96x96
  + Use LabelSampler
    + Background: 0.2
    + Liver: 0.3
    + Tumor: 0.5
* Tio.Queue: Sample 5 patches per volume per epoch
### Training 
* Optimizer: Adam(lr=1e-4)
* BinaryCrossEntropy Loss
* Train for 100 epochs
* Run the main file to train
### Evaluation 
* Dice similarity index method calculated according to the paper [link](https://www.researchgate.net/publication/341806059_3D_Liver_and_Tumor_Segmentation_with_CNNs_Based_on_Region_and_Distance_Metrics)
* Liver segmentation (tumor excluded): Unet 92.7%, liver segmentation (tumor included) 93.4%
* Tumor segmentation: Unet 55.77% 
* According to this paper, [link](https://arxiv.org/ftp/arxiv/papers/2208/2208.13271.pdf), the liver segmentation result showed competitive result
