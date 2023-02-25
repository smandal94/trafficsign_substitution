## Roadsign substitution

### Dataset

DFG Traffic Sign Data Set

- 200 traffic sign categories captured in Slovenian roads
- 5254 training images and 1703 testing images
- 13239 tightly annotated (polygon) traffic sign instances larger than 30 px
- 4359 loosely annotated (bounding box) traffic sign instances smaller than 30 px marked as ignore
- roughly 70% of categories with a low appearance changes and 30% with a large appearance variability
- Website: https://www.vicos.si/resources/dfg/

<p align="center">
  <img src="examples/DFG.gif"/>
</p>

### Preparation
Tested on Ubuntu 22.04, GTX 1070, CUDA 11.7, Python 3.10
1. Clone this repository, run ```conda env create -f environment.yml```
2. Download DFG Traffic Sign Data Set and annotations from [here](https://www.vicos.si/resources/dfg/)
3. Download SuperPoint and SuperGlue weights from [here](https://github.com/magicleap/SuperGluePretrainedNetwork/tree/master/models/weights) and place in ```models/weights```
4. Download official RainNet weight from [here](https://drive.google.com/file/d/1nVJFQ1iAGMeZ-ZybJm9vBQfvYs6tkctZ/view?usp=sharing) and place in ```checkpoints/experiment/```

### Substitution

- Place DFG dataset images in ```examples/DFG```
- Place foreground inpainted images in ```examples/inpainted``` (Look into ```src/create_lama_data.py``` and https://github.com/sayanxtreme/trafficsign_substitution/blob/lama/lama.ipynb)
- The sign templates are placed in ```examples/templates``` which will be used for both homography and substitution
- Run ```python main.py```

### Benchmark Homography

1. First extrack roadsign crops using ```src/save2folder.py```
2. Change paths and run ```python benchmark_homography.py```

- Creates homography dataset with random geometric and photometric augmentations and saves the image pair and H_1_2 matrix
- Uses Mean Average Corner Error from Deep Image Homography Estimation [paper](https://arxiv.org/pdf/1606.03798.pdf) for benchmarking on normalized homography gt-pred pair. Normalized image corners: ```[[-1, -1], [1, -1], [1, 1], [-1, 1]]```
- In below examples, first img1, second img2, green lines are ORB matches, third is predicted transformation with green bbox H_1_2_gt four corners, and red bbox H_1_2_pred four corners.
<p align="center">
  <img src="examples/four_corner_err.gif"/>
</p>

#### Black Background
1. SIFT+FLANN
SIFT+FLANN MACE: 40.59996903319021, Miss: 1268, Average matches: 44.20534477365152
2. ORB+BFMatcher
ORB+BFMatcher MACE: 20.900715256738984, Miss: 232, Average matches: 151.73485172207504
3. AKAZE+BFMatcher
AKAZE+BFMatcher MACE: 78.53744543406565, Miss: 1634, Average matches: 44.842636363636366
4. BRISK+BFMatcher
BRISK+BFMatcher MACE: 108.60093818119128, Miss: 693, Average matches: 52.58415064731267
5. SuperPoint+SuperGlue
SuperPoint+SuperGlue MACE: 13.231284093363495, Miss: 48, Average matches: 51.00449419146952

### Results
#### 1. Foreground Removal (Lama Inpainting)
- GT vs inpainted (```results/lama```)

<p align="center">
  <img src="examples/lama.gif"/>
</p>

#### 2. Keypoint Detection+Matching (SuperPoint+SuperGlue)
- Template to GT matching  (```results/superglue```)

<p align="center">
  <img src="examples/superglue.gif"/>
</p>

#### 3. Harmonization (RainNet)
- Copy-Paste substitution vs Harmonized  (```results/rainnet```)

<p align="center">
  <img src="examples/rainnet.gif"/>
</p>

#### 4. Final Substitution result
- Original GT roadsigns vs substituted roadsigns (Inpaint+Homography+Harmonization)  (```results/substitution```)

<p align="center">
  <img src="examples/substitution.gif"/>
</p>

### Citation
```
 @article{Tabernik2019ITS,
    author = {Tabernik, Domen and Sko{\v{c}}aj, Danijel},
    journal = {IEEE Transactions on Intelligent Transportation Systems},
    title = {{Deep Learning for Large-Scale Traffic-Sign Detection and Recognition}},
    year = {2019},
    doi={10.1109/TITS.2019.2913588}, 
    ISSN={1524-9050}
 }

@article{suvorov2021resolution,
  title={Resolution-robust Large Mask Inpainting with Fourier Convolutions},
  author={Suvorov, Roman and Logacheva, Elizaveta and Mashikhin, Anton and Remizova, Anastasia and Ashukha, Arsenii and Silvestrov, Aleksei and Kong, Naejin and Goka, Harshith and Park, Kiwoong and Lempitsky, Victor},
  journal={arXiv preprint arXiv:2109.07161},
  year={2021}
}

@inproceedings{detone18superpoint,
  author    = {Daniel DeTone and
               Tomasz Malisiewicz and
               Andrew Rabinovich},
  title     = {SuperPoint: Self-Supervised Interest Point Detection and Description},
  booktitle = {CVPR Deep Learning for Visual SLAM Workshop},
  year      = {2018},
  url       = {http://arxiv.org/abs/1712.07629}
}

@inproceedings{sarlin20superglue,
  author    = {Paul-Edouard Sarlin and
               Daniel DeTone and
               Tomasz Malisiewicz and
               Andrew Rabinovich},
  title     = {{SuperGlue}: Learning Feature Matching with Graph Neural Networks},
  booktitle = {CVPR},
  year      = {2020},
  url       = {https://arxiv.org/abs/1911.11763}
}

@inproceedings{ling2021region,
  title={Region-aware Adaptive Instance Normalization for Image Harmonization},
  author={Ling, Jun and Xue, Han and Song, Li and Xie, Rong and Gu, Xiao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9361--9370},
  year={2021}
}
```
### Acknowledgement

* Inpainting code+weights borrowed from [lama](https://github.com/advimman/lama)
* Keypoint detection+matching code+weights borrowed from [SuperGluePretrainedNetwork](https://github.com/magicleap/SuperGluePretrainedNetwork)
* Harmonization code+weights borrowed from [RainNet](https://github.com/junleen/RainNet)
* For homography data and evaluation, code borrowed from [Theseus](https://github.com/facebookresearch/theseus/blob/main/examples/homography_estimation.py)