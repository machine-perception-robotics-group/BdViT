# BdViT

# Dataset
Download the [ImageNet Large Scale Visual Recognition Challenge 2012](https://image-net.org/challenges/LSVRC/2012/) to `./BdViT`.
Execute `split_image.py` to extract `ILSVRC2012_img_val.tar` into a file that can be read by ImageFolder.
The code supports a datasets with the following directory structure:
```
BdViT
├─ ILSVRC2012_devkit_t12
├─ ILSVRC2012_img_val.tar
└─ ILSVRC2012_img_val_for_ImageFolder
   ├─ n01440764
   ├─ n01443537
   ├─ ...
   
```

# Vector decomposition
The weights to which the vector decomposition is applied are placed in `weights/`.
When applying vector decomposition, the convolutional and fully connected layers are defined as `nn.conv2d` and `nn.linear`.
Select the `--model` from `deit_small`, `deit_base`, or `deit_tiny`, change the quantization bits and basis with `--qb`, and change the minimum and maximum values ​​​​of the vector decomposition with `init_iter` and `max_iter` in `bdnn_module/Exhaustive_decomposer_numpy.py`.
```
BdViT
└─ weights
   ├─ deit_base/
   |  └─ deit_base_patch16_224-b5f2ef4d.pth
   ├─ deit_small/
   |  └─ deit_small_patch16_224-cd65a155.pth
   └─ deit_tiny/
      └─ deit_tiny_patch16_224-a1311bcf.pth
   
```
```
cd BdDETR
python3.10 decomposition.py --model deit_small --qb 8
```

# Decomposed weights
The decomposed weights can be downloaded [here](https://drive.google.com/file/d/1nSLbYJCZ_Dm0dO9xUBha6OXjCJlFhLVP/view?usp=sharing).

# Evaluation
For evaluation, the model definition file is `binary_model.py`.
If you change the comment out of `conv2d_binary` and `linear_binary_KA` in `bdnn_module/binary_functional.py`, you can choose DeiT and BdViT.
Quantize bits and basis change `--qb`.
```
cd BdDETR
python3.10 test.py --batch_size 2 --no_aux_loss --eval --coco_path coco --weights ./weights/detr/detr-r50-e632da11.pth --qb 8
```

# bdnn_module
```
build, binaryfunc_cython.c                         - Folders/files generated when compiling cython
binaryfunc_cython.cpython-36m-x86_64-linux-gnu.so  - Logical operation module for python3.6
binaryfunc_cython.cpython-37m-x86_64-linux-gnu.so  - Logical operation module for python3.7
binaryfunc_cython.cpython-38m-x86_64-linux-gnu.so  - Logical operation module for python3.8
binaryfunc_cython.cpython-39m-x86_64-linux-gnu.so  - Logical operation module for python3.9
binaryfunc_cython.cpython-310m-x86_64-linux-gnu.so - Logical operation module for python3.10
binaryfunc_cython.pyx                              - cython logic operation definition file
Exhaustive_decomposer_numpy.py                     - Vector decomposition module for Exhaustive algorithm
conv_binary.py                                     - BdDETR Convolution layer definition file
fully_connected_binary.py                          - BdDETR FullyConnected layer definition file
setup.py                                           - Program to compile cython code
utils 
  |---calc_comput_complex.py                       - Feature map size calculation module
  |---decomposition.py                             - Module to select method and perform decomposition
  |---extract_weight_ext.py                        - Module to extract parameters and simultaneously perform analysis such as standard deviation
```
