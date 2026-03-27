<img src='images/halfedgecnn_image.png' align="right" width=500>
<br><br><br>

# HalfedgeCNN With HKS

*This is a modified version of the code of Halfedge CNN. I modified the mesh viewer, added a new set of features using only the Heat Kernel Signature (HKS) as input for the model, and added some code for visualisation (we can really "see what the model sees" because the HKS is in $\mathbb{R}$).*

My contributions in the code are mostly in the following files:
- ```util/mesh_viewer_polyscope```: There was already a mesh viewer, but it was pretty horrible and hard to use. With polyscope, it is way simpler.
- ```models/layers/hks.py```: The file in which the HKS in computed. I tried to follow the logic of the other files from HalfedgeCNN.
- ```models/layers/half_edge_mesh_pool.py```: modified to store the "updated HKS values" on the halfedges BEFORE each pooling operation (this choice is quite arbitrary)
- ```models/layers/half_edge_mesh_prepare.py```: modified to add the case "features == 3", where the model uses only HKS (this was quite easy, the code was made to allow this kind of modification)
- ```scripts/test_with_settings.py```: modified the logic to be able to call every model by their name (it was not the case originally)
- ```calculate_test_hks```: Used to store the HKS of the test files
- ```scripts/calculate_test_hks_with_settings.py```: Used to call the above file

Files not used anymore:
- ```manual_calculations```: As the name indicates, it corresponds to the beginning of the project, where I was doing the HKS manually. It uses the code from the graded TP.
- ```download_from_kaggle```: Used to download modelnet40 from Kaggle. I did not use it in the end.

*Rk: I am not sure my code works well for segmentation, as all I wanted to do is use my models for classification.*

*Rk: I also added commentaries at the top of the files in the folder 'models' and on some functions, to better understand the structure of the code.*

*Rk: See the file TODO.md for ideas of things I could have done with more time.*

Concretely, to reproduce my work, you can:

- go to the [google colab](https://colab.research.google.com/drive/1g8LGdkq8E7y8MkP9wVEJNrr1XTs4-Jud?usp=sharing) to train a new halfedgeCNN model using HKS on shrec_16
- download the weights as indicated in the colab and give a nice name to the model
- download the data using ```bash scripts/get_shrec_data.sh```
- execute ```python scripts/calculate_test_hks_with_settings.py``` to store initial HKS values for each mesh.
- execute ```python scripts/test_with_settings.py --model name_of_model.pth --export_pooled_channel i``` on your machine, storing the "pooled meshes" and the "updated hks values" of the i_th channel of the model (see the images from the pdf)
- execute ```python util/mesh_viewer_polyscope --files end_of_path.obj --hks_values end_of_path.npy``` (eg. ```python util/mesh_viewer_polyscope.py --files T0_0.obj --hks_values T0_initial_hks.npy```) to see the "updated hks values" seen by the first channel of the model as we go down the layers (ie. before each pool). Everything is stored in ```checkpoints/shrec_16/export/classification```.
- enjoy

I also put the downloaded weights of my best HKS model in the folder ```stored_weights```. You can use them instead of downloading your own weights from a model you trained on the colab. See the README.md file in ```stored_weights``` for more info.

Finally, here is a modified version of the readme from the github of HalfedgeCNN:

----------
----------


### Eurographics Symposium on Geometry Processing 2023

HalfedgeCNN is a general-purpose deep neural network operating on 3D triangle meshes. Inspired by the edge-based [MeshCNN](https://ranahanocka.github.io/MeshCNN/) it includes convolution and (un)pooling layers, but operates on the basis of **halfedges**. This provides certain benefits over alternative that are formulated on the basis of edges, vertices, or faces.

HalfedgeCNN is described in more detail in the below publication.

# Citation

If you find this code useful, please consider citing our paper
```
@article{HalfedgeCNN,
author = {Ludwig, I. and Tyson, D. and Campen, M.},
title = {HalfedgeCNN for Native and Flexible Deep Learning on Triangle Meshes},
journal = {Computer Graphics Forum},
volume = {42},
number = {5},
doi = {https://doi.org/10.1111/cgf.14898},
year = {2023}
}
```

# Getting Started

### Install dependencies
1. Create a new Anaconda environment: 
```bash
conda create --name halfedgecnn
```
2. Switch to the new environment:
```bash
conda activate halfedgecnn    
```
3. Install Pytorch:
```bash
conda install pytorch torchvision torchaudio -c pytorch
```
4. Install tensorboardX for viewing the training plots (optionally):
```bash
conda install -c conda-forge tensorboardx
```

Depending on the system, the installation of additional packages might be necessary.

### 3D Shape Classification on SHREC
*(most of this is in the colab for the training of the model)*
For starting the SHREC classification training, first download and unzip the dataset using the get_shrec_data.sh script:
```bash
bash scripts/get_shrec_data.sh
```
Now start the training, with the following command:
```bash
python scripts/train_with_settings.py shrec_16  
```
To view the training loss and accuracy plots, run ```tensorboard --logdir runs``` in another terminal and click [http://localhost:6006](http://localhost:6006).

After training the latest model can be tested using the following command:
```bash
python scripts/test_with_settings.py --model latest_model.pth
```
The best found model can be tested using the following command:
```bash
python scripts/test_best_with_settings.py --model best_model.pth
```

The resulting poolings can be exported with:
```bash
python scripts/export_results_as_obj.py shrec_16  
```
Examples of the resulting poolings can be viewed (after exporting) for example with the following command:
```bash
python util/mesh_viewer.py --files checkpoints/shrec_16/result_objs/T74_0.obj checkpoints/shrec_16/result_objs/T74_1.obj checkpoints/shrec_16/result_objs/T74_2.obj checkpoints/shrec_16/result_objs/T74_3.obj checkpoints/shrec_16/result_objs/T74_4.obj
```

*The things that follow were in the original README.md. Since I do not use Segmentation in my project, it might need some adjustments in the commands. The end about neigbourhoods, pooling and feature selection is available.*

--------
--------

### 3D Shape Segmentation on Humans
For starting the human segmentation training, first download and unzip the dataset using the get_human_data.sh script:
```bash
bash scripts/get_human_data.sh
```

Make sure that in the general settings (scripts/settings/general_settings.txt) the segmentation base is set to the right value, in our example edge based (--segmentation_base edge_based)

Then start the training with the following command:
```bash
python scripts/train_with_settings.py human_seg  
```
Again, to view the training loss and accuracy plots, run ```tensorboard --logdir runs``` in another terminal and click [http://localhost:6006](http://localhost:6006).

After training the latest model can be tested using the following command:
```bash
python scripts/test_with_settings.py human_seg  
```
The best found model can be tested using the following command:
```bash
python scripts/test_best_with_settings.py human_seg
```

The resulting poolings and segmentations can be exported with:
```bash
python scripts/export_results_as_obj.py human_seg
```

The resulting segmentation for one mesh can be viewed (after exporting) for example with the following command:
```bash
python util/mesh_viewer.py --files checkpoints/human_seg/result_objs/shrec__14_0.obj
```

Similar to the shrec classification and human segmentation task, other classification and segmentation tasks can be performed. 
For every dataset a settings file needs to be created in the scripts/settings directory.
In it it needs to be defined whether it is a classification or segmentation dataset, what the name of the dataset is, what the name of the results folder should be and what the highest number or faces occurring in the dataset is. 
In the case of a classification task, also the pooling resolutions needs to be given, because the settings differ between the classification datasets.
The dataset setting files can also be used to override the default settings for specific datasets. If a setting is given both in the default settings and in the dataset settings, the dataset setting is used.
For example, the settings might be as follows:

Classification:
```bash
--dataset_mode classification
--dataroot datasets/shrec_16
--name shrec_16
--pool_res 1200 900 600 360
--niter_decay 100
```
Segmentation:
```bash
--dataset_mode segmentation
--dataroot datasets/human_seg
--name human_seg
--number_input_faces 1520
```

### Evaluate Runs
To simplify the evaluation of the runs, one can use the evaluate_runs.py, remove_unfinished_runs.py and show_chart.py script.

The evaluate_runs scripts creates a list of the end-values and the best values for all recorded runs in the runs directory and computes the mean, the 
standard deviation and other interesting statistics for all runs.

The remove_unfinished_runs.py scripts removes all runs that have crashed or that did not reach the desired number of epochs.

The show_chart.py script shows the development of the all recorded runs, marking the first occurrence of the best value of each run. 

All scripts need the desired number of epochs as a parameter, for example 200 for the shrec classification task and 300 for the human segmentation tasks.
```bash
python scripts/evaluate_runs.py 200
python scripts/remove_unfinished_runs.py 200
python scripts/show_chart.py 200
```

### Change Neighborhood Size
The used convolution neighborhood can be selected by changing the number behind the --nbh_size parameter in scripts/settings/general_settings.txt.
The nomenclature used for the nbh_size is as described in figure 2 and the table in section 7 of the paper. 
Available are neighborhoods sizes 2, 3, 4, 5, 7, and 9.

### Change Feature Selection 
By changing the number behind the --feat_selection parameter, a different feature combination can be selected.
Following the nomenclature used in the paper, "0" means symmetrized features, "1" oriented features, and "2" fundamental features.

### Change Pooling
Two different pooling methods are available, Edge-Pooling and Half-Edge-Pooling. 
For Half-Edge-Pooling the --pooling parameter has to be set to "half_edge_pooling". 
For Edge-Pooling the parameter has to be set to "edge_pooling".

### Change Segmentation Base
Three different segmentation bases are available: edge based, half-edge based, and face based.
For edge based segmentation the --segmentation_base parameter has to be set to "edge_based", for halfedge based segmentation to "half_edge_based", and for face based segmentation to "face_based".


# Acknowledgments

This code is based on a modification of the code of [MeshCNN](https://ranahanocka.github.io/MeshCNN/).
