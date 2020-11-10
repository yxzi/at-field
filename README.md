# at-field
11785 Intro to Deep Learning Project Team, A.T. Field



# Getting started

1.  Clone this repository: `git clone https://github.com/yxzi/at-field.git`

2.  Install the dependencies:  
```
conda create -n smoothing
conda activate smoothing
# below is for linux, with CUDA 10; see https://pytorch.org/ for the correct command for your system
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch 
conda install scipy pandas statsmodels matplotlib seaborn
pip install setGPU
```
3.  Download our trained models from [here](https://drive.google.com/file/d/1h_TpbXm5haY5f-l4--IKylmdz6tvPoR4/view?usp=sharing).
4. If you want to run ImageNet experiments, obtain a copy of ImageNet and preprocess the `val` directory to look
like the `train` directory by running [this script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh).
Finally, set the environment variable `IMAGENET_DIR` to the directory where ImageNet is located.

5. To get the hang of things, try running this command, which will certify the robustness of one of our pretrained CIFAR-10 models
on the CIFAR test set.
```
model="models/cifar10/resnet110/noise_0.25/checkpoint.pth.tar"
output="???"
python code/certify.py cifar10 $model 0.25 $output --skip 20 --batch 400
```
where `???` is your desired output file.


6. To run prediction + attack, install following dependency:

```
pip install adversarial-robustness-toolbox
```

execute the following in repo root:

```
model="models/cifar10/resnet110/noise_0.25/checkpoint.pth.tar"
output="./output/log/???"
python code/test_attack.py cifar10 $model 0.25 $output --N 1000 --batch 400 --max 100
```
