## Roadsign Instance segmentation

### Dataset

DFG Traffic Sign Data Set

- 200 traffic sign categories captured in Slovenian roads
- 5254 training images and 1703 testing images
- 13239 tightly annotated (polygon) traffic sign instances larger than 30 px
- 4359 loosely annotated (bounding box) traffic sign instances smaller than 30 px marked as ignore
- roughly 70% of categories with a low appearance changes and 30% with a large appearance variability
- Website: https://www.vicos.si/resources/dfg/

Other datasets worth looking into (No segmentation gt, only bboxes):
- Chinese Traffic Sign Database (Website: http://www.nlpr.ia.ac.cn/pal/trafficdata/detection.html)
- German Traffic Sign Detection Benchmark GTSDB (Website: https://benchmark.ini.rub.de/gtsdb_dataset.html)


### Model

We fine-tune instance segmentation MaskRCNN model with ResNet50 backbone pre-trained on COCO.

Please download checkpoint from [here](https://drive.google.com/file/d/1H6nU-L_yoD0XZdLl7KqnzH_VQfZNebB2/view?usp=sharing) and move to ```./checkpoints/``` folder


### Train and test

- To train the model, look into ```maskrcnn/train.py```
- To test the model, look into ```maskrcnn/test.py```
- To run inference, look into ```maskrcnn/inference.py```

### Results

#### Test Images (GT::Predicted)
![Image ID: 432](results/result_id432.png)
![Image ID: 1847](results/result_id1847.png)
![Image ID: 2577](results/result_id2577.png)
![Image ID: 2825](results/result_id2825.png)

#### Test
```
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.732
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.836
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.819
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.303
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.824
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.894
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.776
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.789
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.789
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.415
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.873
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.909
IoU metric: segm
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.743
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.834
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.810
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.288
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.842
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.943
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.788
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.800
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.800
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.408
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.890
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.947
 ```
#### Test substitution
```
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.556
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.709
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.684
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.160
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.692
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.745
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.626
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.629
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.629
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.260
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.750
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.761
IoU metric: segm
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.636
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.709
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.701
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.189
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.785
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.863
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.711
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.714
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.714
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.313
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.843
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.868
```



## Citation
```
 @article{Tabernik2019ITS,
    author = {Tabernik, Domen and Sko{\v{c}}aj, Danijel},
    journal = {IEEE Transactions on Intelligent Transportation Systems},
    title = {{Deep Learning for Large-Scale Traffic-Sign Detection and Recognition}},
    year = {2019},
    doi={10.1109/TITS.2019.2913588}, 
    ISSN={1524-9050}
 }
```

```
@misc{https://doi.org/10.48550/arxiv.1703.06870,
  doi = {10.48550/ARXIV.1703.06870},
  
  url = {https://arxiv.org/abs/1703.06870},
  
  author = {He, Kaiming and Gkioxari, Georgia and Dollár, Piotr and Girshick, Ross},
  
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Mask R-CNN},
  
  publisher = {arXiv},
  
  year = {2017},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```