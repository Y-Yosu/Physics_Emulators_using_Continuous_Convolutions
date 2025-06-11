---
license: mit
---

# Symmetric basis convolutions for learning lagrangian fluid mechanics (Published at ICLR 2024) - Test Case I

This dataset contains the data for the first test case (1D compressible SPH) for the paper Symmetric basis convolutions for learning lagrangian fluid mechanics (Published at ICLR 2024). 

You can find the full paper [here](https://arxiv.org/abs/2403.16680). 

The source core repository is available [here](https://github.com/tum-pbs/SFBC/) and also contains information on the data generation. You can install our BasisConvolution framework simply by running
`pip install BasisConvolution`

For the other test case datasets look here:

[Test Case I (compressible 1D)](https://huggingface.co/datasets/Wi-Re/SFBC_dataset_i)

[Test Case II (wcsph 2D)](https://huggingface.co/datasets/Wi-Re/SFBC_dataset_ii)

[Test Case III (isph 2D)](https://huggingface.co/datasets/Wi-Re/SFBC_dataset_iii)

[Test Case IV (3D)](https://huggingface.co/datasets/Wi-Re/SFBC_dataset_iv)

## File Layout

The datasets are stored as hdf5 files with a single file per experiment. Within each file there is a set of configuration parameters and each frame of the simulation stored separately as a group. Each frame contains information for all fluid particles and all potentially relevant information. For the 2D test cases there is a pre-defined test/train split on a simulation level, wheras the 1D and 3D cases do not contain such a split.


## Demonstration

This repository contains a simple Jupyter notebook (Visualizer.ipynb) that loads the dataset in its current folder and visualizes it first:

![alt text](data.png)

And then runs a simple training on it to learn the SPH summation-based density for different basis functions:

![alt text](example.png)

## Minimum Working Example

Below you can find a fully work but simple example of loading our dataset, building a network (based on our SFBC framework) and doing a single network step. This relies on our SFBC/BasisConvolution framework that you can find [here](https://github.com/tum-pbs/SFBC/) or simply install it via pip (`pip install BasisConvolution`)

```py
from BasisConvolution.util.hyperparameters import parseHyperParameters, finalizeHyperParameters
from BasisConvolution.util.network import buildModel, runInference
from BasisConvolution.util.augment import loadAugmentedBatch
from BasisConvolution.util.arguments import parser
import shlex
import torch
from torch.utils.data import DataLoader
from BasisConvolution.util.dataloader import datasetLoader, processFolder

# Example arguments 
args = parser.parse_args(shlex.split(f'--fluidFeatures constant:1 --boundaryFeatures constant:1 --groundTruth compute[rho]:constant:1/constant:rho0 --basisFunctions ffourier --basisTerms 4 --windowFunction "None" --maxUnroll 0 --frameDistance 0 --epochs 1'))
# Parse the arguments
hyperParameterDict = parseHyperParameters(args, None)
hyperParameterDict['device'] = 'cuda' # make sure to use a gpu if you can
hyperParameterDict['iterations'] = 2**10 # Works good enough for this toy problem
hyperParameterDict['batchSize'] = 4 # Automatic batched loading is supported
hyperParameterDict['boundary'] = False # Make sure the data loader does not expect boundary data (this yields a warning if not set)

# Build the dataset
datasetPath = 'dataset/train'
train_ds = datasetLoader(processFolder(hyperParameterDict, datasetPath))
# And its respective loader/iterator combo as a batch sampler (this is our preferred method)
train_loader = DataLoader(train_ds, shuffle=True, batch_size = hyperParameterDict['batchSize']).batch_sampler
train_iter = iter(train_loader)
# Align the hyperparameters with the dataset, e.g., dimensionality
finalizeHyperParameters(hyperParameterDict, train_ds)
# Build a model for the given hyperparameters
model, optimizer, scheduler = buildModel(hyperParameterDict, verbose = False)
# Get a batch of data

try:
    bdata = next(train_iter)
except StopIteration:
    train_iter = iter(train_loader)
    bdata = next(train_iter)
# Load the data, the data loader does augmentation and neighbor searching automatically
configs, attributes, currentStates, priorStates, trajectoryStates = loadAugmentedBatch(bdata, train_ds, hyperParameterDict)
# Run the forward pass
optimizer.zero_grad()
predictions = runInference(currentStates, configs, model, verbose = False)
# Compute the Loss
gts = [traj[0]['fluid']['target'] for traj in trajectoryStates]
losses = [torch.nn.functional.mse_loss(prediction, gt) for prediction, gt in zip(predictions, gts)]
# Run the backward pass
loss = torch.stack(losses).mean()
loss.backward()
optimizer.step()
# Print the loss
print(loss.item()) 
print('Done')
```