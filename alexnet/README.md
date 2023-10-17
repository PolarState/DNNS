Re-training Alexnet:
Alexnet was the first CNN. It’s claim to fame was to do really well on the Imagenet 2012 competition. 
Downloading imagenet:
wget --show-progress --tries 10 -O /media/Totoro/Datasets/imagenet/10k/imagenet10k_eccv2010.tar https://www.image-net.org/data/imagenet10k_eccv2010.tar

Conda environment
`conda install cuda -c nvidia/label/cuda-11.8.0`
`conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`


Environment Troubleshooting:
Mismatched cuda and pytorch versions: https://forums.developer.nvidia.com/t/failed-to-initialize-nvml-driver-library-version-mismatch/190421/4
Paper uses 5crop and that was annoying to set up. Partially due to tensorvision revision status. Used old way of solving problem. In repo.

Important links:
Pytorch alexnet website: https://pytorch.org/hub/pytorch_vision_alexnet/
Imagenet dataloader in torchvision: https://github.com/pytorch/vision/blob/main/torchvision/datasets/imagenet.py
Imagenet download: https://www.image-net.org/challenges/LSVRC/2012/2012-downloads.php
 
