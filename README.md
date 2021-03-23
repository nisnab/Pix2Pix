# Image Translation using Pix2Pix GAN
In this repository,  I implemented the paper "Image-to-Image Translation with Conditional Adversarial Networks"(https://arxiv.org/pdf/1611.07004.pdf) to translate images from landscapes painting to Real-world images. 

## Table of contents
* [Task Description](#TaskDescription)
* [Installation](#Installation)
* [Usage](#Usage)
## Task Description

```bash
```
Folder structure
--------------

```
├── datasets/Train/a       - this folder contains landscape images.
│   ├── image1001.png
│   └── image1002.png
│   └── --------------------
│
│
├── datasets/Train/b      - this folder contains Real-world images.
│   ├── image1001.png
│   └── image1002.png
│   └── --------------------  
│
├── datasets/Test-set/a             - this folder contains Test images(landscapes).
│   └── image1001.png
│   └── -------------------- 
│
├── save_model    -- this folder contains saved model
│
│── Python-scripts      - this folder contains  python files(can be run driectly in Jupyter notebook/IDE)
│
├──  train-MANET.py        - this file is used for training image.
│   
├──  testing.py         - this file is used for generating test images.
│   
├──  result        - this folder contains generated test images.
│ 
└──logs/tensorlogs     

```


