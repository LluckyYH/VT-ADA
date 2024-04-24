# VT-ADA(CDNA) implemneted in PyTorch

This paper explores the utilization of Vision **Transformer (ViT) as a feature extractor** in adversarial domain adaptation, demonstrating its effectiveness as a plug-and-play component that enhances performance compared to CNN-based approaches.

## Prerequisites

- PyTorch >= 0.4.0 (with suitable CUDA and CuDNN version)
- torchvision >= 0.2.1
- Python3
- Numpy
- argparse
- PIL

## Dataset

### Office-31

Office-31 dataset can be found [here](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/).

### Office-Home

Office-Home dataset can be found [here](http://hemanthdv.org/OfficeHome-Dataset/).

### Image-clef

We release the Image-clef dataset we used [here](https://drive.google.com/file/d/0B9kJH0-rJ2uRS3JILThaQXJhQlk/view).

```bash
# You can also directly download the required datasets within our code.

git clone git@github.com:LluckyYH/.git # Please note that adjustments to the path should be noted.
```

## Training

All parameters are optimized in our experiments. Below are the commands for each task. The `test_interval` can be adjusted, representing the number of iterations between consecutive tests.

If you wish to utilize our enhanced architecture, you have the option to select from the following choices for the `--net` parameter: "vit_small_patch16_224", "vit_base_patch16_224", "vit_large_patch16_224", "vit_huge_patch14_224", and so on.

```python
# SVHN->MNIST
python train_svhnmnist.py --gpu_id id --epochs 50

# USPS->MNIST
python train_uspsmnist.py --gpu_id id --epochs 50 --task USPS2MNIST

# MNIST->USPS
python train_uspsmnist.py --gpu_id id --epochs 50 --task MNIST2USPS
```

```python
Office-31
# The order:
# amazon_list.txt-webcam_list.txt, amazon_list.txt-dslr_list.txt;
# dslr_list.txt-amazon_list.txt, dslr_list.txt-webcam_list.txt;
# webcam_list.txt-amazon_list.txt, webcam_list.txt-dslr_list.txt;

python train_image.py --gpu_id id --net vit_small_patch16_224 --dset office --test_interval 500 --s_dset_path ../data/office/webcam_list.txt --t_dset_path ../data/office/dslr_list.txt
```

```python
# Office-Home
# The order: Ar-Cl	Ar-Pr	Ar-Rw	Cl-Ar	Cl-Pr	Cl-Rw	Pr-Ar	Pr-Cl	Pr-Rw	Rw-Ar	Rw-Cl	Rw-Pr
python train_image.py --gpu_id id --net vit_small_patch16_224 --dset office-home --test_interval 2000 --s_dset_path ../data/office-home/Real_World.txt --t_dset_path ../data/office-home/Product.txt
```

```python
# Image-clef
# The order: I-P  P-I  I-C  C-I  C-P  P-C
python train_image.py --gpu_id id --net ResNet50 --dset image-clef --test_interval 500 --s_dset_path ../data/image-clef/p_list.txt --t_dset_path ../data/image-clef/c_list.txt
```

If you want to run the random version of CDAN, add `--random` as a parameter.
