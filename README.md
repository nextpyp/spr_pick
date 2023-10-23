# Joint Image Denoising and Particle Detection in cryoEM

This code is based on the [self-supervised denoising](https://github.com/COMP6248-Reproducability-Challenge/selfsupervised-denoising) approach.

## Python requirements
This code was tested using:
- Python 3.7
- [PyTorch](https://pytorch.org/get-started/locally/) 1.8.0 [Cuda 10.0 / CPU]
- [Anaconda 2020/02](https://www.anaconda.com/distribution/)

## Installation
1. Create an Anaconda/venv environment (Optional)
2. Install PyTorch
3. Install joint package and dependencies: ```pip install -e joint```

## Preparing datasets
First, make a ```train/``` directory for saving all training micrographs and their corresponding coordinate files. Each micrograph should have its own coordinate file: e.g. ```train_1_img.mrc``` and ```train_1_img.txt```.

For a dataset with aligned micrographs, we first downsample the full micrographs 8x (from ~0.66 Å/pix to ~5.28 Å/pix). [IMOD](https://bio3d.colorado.edu/imod/)'s [newstack](https://bio3d.colorado.edu/imod/doc/man/newstack.html) command can be used to do the downsampling. To downsample a micrograph by a factor of 4, run:
```
newstack input_micrograph output_micrograph -bin 4
```
To downsample all the micrographs in a given directory, a bash for loop can be used:
```
for i in *.mrc; do newstack $i out_$i -bin 4; done;
```
Once training micrographs are binned, move all training micrographs to the new ```train/``` directory. For example, a directory with 10 micrographs should have 10 coordinate text files.

For training coordinates, manually picking is performed on selected micrographs using IMOD. For each micrograph, find a subregion and pick around 40\% to 80\% of the particles in that subregion. For example, for an input of size 1024 * 1024, a subregion of around 300 * 300 will suffice. After manual annotation, IMOD will generate ```.mod``` files containing the annotated coordinates. Converting ```.mod``` files to ```.txt``` files can be done using the [model2point](https://bio3d.colorado.edu/imod/doc/man/model2point.html). For example:
```
model2point input.mod input.txt 
```
Once all the ```.mod``` files are converted to text files, move all ```coordinate.txt``` files to the ```train/``` directory.

To generate the training files, run ```generate_train_files.py``` file from joint folder. Two input arguments are required: ```-d/--dir``` to specify the path to the train directory, and ```-o/--out``` to specify the output training file name. For example:
```
python generate_train_files.py -e .mrc -d train_dir -o train_name
```
Once finished, two files will be generates in the train directory folder: 1. ```out_train_images.txt```; 2. ```out_train_coords.txt```. These two files will serve as input to the program.

## Running
The joint denoiser and particle picker is exposed as a CLI accessible via the ```joint``` command.

### Training:
To train a network, run:
```
joint train start [-h] --train_dataset TRAIN_DATASET --train_label TRAIN_LABEL
                        [--validation_dataset VALIDATION_DATASET] [--validation_label VALIDATION_LABEL] 
                        --iterations ITERATIONS [--eval_interval EVAL_INTERVAL]
                        [--checkpoint_interval CHECKPOINT_INTERVAL]
                        [--print_interval PRINT_INTERVAL]
                        [--train_batch_size TRAIN_BATCH_SIZE]
                        [--validation_batch_size VALIDATION_BATCH_SIZE]
                        [--patch_size PATCH_SIZE] --algorithm
                        {n2c,n2n,n2v,ssdn,ssdn_u_only} --noise_style
                        NOISE_STYLE [--noise_value {known,const,var}] [--mono]
                        [--diagonal] [--runs_dir RUNS_DIR] [--nms NMS] [--num NUM][--alpha][--tau][--bb]

The following arguments are required: --train_dataset/-t, --train_label/-l, --iterations/-i, --algorithm/-a, --noise_style/-n, --noise_value (when --algorithm=ssdn)
```
Note that the validation dataset is optional, this can be omitted but may be helpful to monitor convergence. Where a parameter is not provided the default in `cfg.py` will be used. Here ```--num``` controls the number of validation samples we want to use.

---

Training will create model checkpoints that contain the training state at specified intervals (```.training``` files). When training completes, a final output is created containing only network weights and the configuration used to create it (```.wt``` file). The latest training file for a run can be resumed using:
```
ssdn train resume [-h] [--train_dataset TRAIN_DATASET] --train_label TRAIN_LABEL
                         [--validation_dataset VALIDATION_DATASET] [--validation_label VALIDATION_LABEL]
                         [--iterations ITERATIONS]
                         [--eval_interval EVAL_INTERVAL]
                         [--checkpoint_interval CHECKPOINT_INTERVAL]
                         [--print_interval PRINT_INTERVAL]
                         [--train_batch_size TRAIN_BATCH_SIZE]
                         [--validation_batch_size VALIDATION_BATCH_SIZE]
                         [--patch_size PATCH_SIZE]
                         run_dir
The following arguments are required: run_dir (positional)
```
A sample command can be:
```
joint train start --algorithm ssdn --noise_value var --noise_style gaussian --runs_dir new_test 
--train_dataset train_imgs_updates.txt 
--train_label train_coords_updates.txt 
--validation_dataset train_imgs_updates.txt 
--validation_label train_coords_updates.txt 
--iterations 80000 --alpha 0.75 --train_batch_size 4 --nms 18 --num 1 --tau 0.01 --bb 24
```

---

Further options can be viewed using: `joint train {cmd} --help` where `{cmd}` is `start` or `resume`.

### Output For Training
All training files will be saved to the ```--runs_dir``` directory, inside the directory, there will be a subdirectory of current run, naming based on the parameters and the number of past runs. Inside the current subdirectory, there are two folders, ```training_jt``` with checkpoints snapshots and ```val_imgs_joint``` with validation image output at each evaluation iteration. For each evaluation iteration, several files are generated: 1. _scores.txt file for output coordinates after non-max suppression operation. 2. _pred_tar.png predicted heatmap output. 3. out-std and out-mu denoised image prior statistics 4. _out denoised image output. 5. _nsy input noisy image.

There will also be a log file and a tensorboard logger.

### Evaluating:

To evaluate a trained network against one of the validation sets, run:
```
joint eval [-h] --model MODEL --dataset DATASET [--runs_dir RUNS_DIR]
                 [--batch_size BATCH_SIZE][--num NUM]
The following arguments are required: --model/-m, --dataset/-d
```

A sample command can be:
```
joint eval --model new_test/00000-train-ssdn-iter300k-0.75-joint/training_jt/model_00224000.training
--dataset train_dir/test_train_imgs.txt
--runs_dir eria_test_run --num 100
```
---
Further options can be viewed using: `joint eval --help`. Note ```--num``` controls the number of samples we want to evaluate. For the whole dataset evaluation, input the total number of micrographs. 

### Output for evaluation

All output files will be saved to the ```--runs_dir``` directory, there will be a subdirectory of current run, naming based on the parameters and the number of past runs. All outputs will be saved to ```eval_imgs``` folder.

### Extra notes:

The network will attempt to use all available GPUs - `cuda0` being used as the master with the batch distributed across all remaining. To avoid this filter the GPUs available using:
```
CUDA_VISIBLE_DEVICES=#,#,# joint ...
```

---

During execution an events file is generated with all training metrics. This can be viewed using Tensorboard.

When executing remotely, it may be preferable to expose this to a local machine. The suggested method to do this is `ssh`:
```
ssh -L 16006:127.0.0.1:6006 {username}@{remote}
$ tensorboard --logdir runs
# Connect locally at: http://127.0.0.1:16006/
```
