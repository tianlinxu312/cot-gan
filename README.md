# COT-GAN: Generating Sequential Data via Causal Optimal Transport
Authors: Tianlin Xu, Li K. Wenliang, Michael Munn, Beatrice Acciaio

COT-GAN is an adversarial algorithm to train implicit generative models optimized for producing sequential data. The loss function of this algorithm is formulated using ideas from Causal Optimal Transport (COT), which combines classic optimal transport methods with an additional temporal causality constraint. 

This repository contains an implementation and further details of COT-GAN. 

Reference: Tianlin Xu, Li K. Wenliang, Michael Munn, Beatrice Acciaio, "COT-GAN: Generating Sequential Data via Causal Optimal Transport," Neural Information Processing Systems (NeurIPS), 2020.

Paper Link: https://arxiv.org/abs/2006.08571 (replace with Neurips link)

Contact: tianlin.xu1@gmail.com

## Setup

Begin by installing pip and setting up virtualenv.

```
$ curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
$ python get-pip.py
$ python3 -m pip install virtualenv
```

### Clone the Github repository and install requirements

```
$ git clone https://github.com/tianlinxu312/cot-gan.git
$ cd cot-gan/

# Create a virtual environment called 'venv'
$ virtualenv venv 
$ source venv/bin/activate    # Activate virtual environment
$ python3 -m pip install -r requirements.txt 
```
## Training COT-GAN
We trained COT-GAN on synthetic low-dimensional datasets as well as two high-dimensional video datasets: a [human action dataset](http://www.wisdom.weizmann.ac.il/~vision/SpaceTimeActions.html) and an [animated Sprites dataset](https://github.com/jrconway3/Universal-LPC-spritesheet)

For training on low-dimensional datasets, use a flag to specify either synthetic time series sine data (`SineImage`), auto-regressive data of order one (`AROne`), or EEG data (`eeg`). For example, to train on AR-1 data:
```
python3 -m toy_train \
  --dname="AROne"
```
See the code for how to modify the default values of other training parameters or hyperparameters.

Similarly, for training on video datasets, specify either the human action or animated Sprites dataset; either `human_action` or `animation`, resp.

```
python3 -m video_train \
  --dname="human_action"
```

### Data
.tfrecord files used in COT-GAN experiments can be downloaded here: https://drive.google.com/drive/folders/1ja9OlAyObPTDIp8bNl8rDT1RgpEt0qO-?usp=sharing

## Results

### Animated Sprites

<img src="./figs/animation.gif" width="360" height="120"/>

### Human Actions 

<img src="./figs/humanaction.gif" width="360" height="120"/>


