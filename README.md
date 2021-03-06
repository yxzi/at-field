# at-field
11785 Intro to Deep Learning Project Team, A.T. Field

(the following baseline model setup is tested on AWS g4dn.xlarge and deep learning AMI OS image)

# Getting started

1.  Clone this repository: `git clone https://github.com/yxzi/at-field.git`

2.  Install the dependencies:  
```
conda create -n adfield
conda activate atfield
# below is for linux, with CUDA 11; see https://pytorch.org/ for the correct command for your system
conda install pytorch torchvision cudatoolkit=11.0 -c pytorch 
conda install scipy pandas statsmodels matplotlib seaborn
pip install setGPU
```

3. To train the baseline model with cifar10 + resnet110, please use the following command:
```
output="[output_dir_name]"
python code/train_target.py $output resnet18 --batch 400 --noise 0.25 --eps_step 0.05
```
this will train a resnet110 model with random noised of 0.25 magnitude added with cifar10 data, it runs for 90 epoch by default, you can break out at any epoch since models will be saved within '[repo root]/exp/[output dir name]/models'; In our experiment, running around 45 epochs will get you comparable accuracy with the paper, that is above 70% validation accuracy


4. To run prediction + attack, install following dependency:

```
pip install adversarial-robustness-toolbox
# install remaining dependencies accordingly, in our case 'dataclasses' package is missing
conda install dataclasses
```

execute the following in repo root:

```
model="./exp/[output dir name]/models/[selected model].pth"
python code/test_sky.py $model 0.25 --N 1000 --batch 400 --max 500 --eps 0.3 --eps_step 0.1 --max_iter 30
```

where model is path to your trained models from last step
"--max" option specific how many test data from cifar10 you want use (the first N data points), this also specify how many adversarial example you are generating
The process of adversarial attack trainer generating adversarial data is going to take a long time (on g4dn.xlarge, every image is taking about 5-8 secs, it takes approximately 14min for generating 100 adversarial images)