# Infrared and Visible Image Fusion using a Deep Learning Framework
ICPR 2018(Accepted).

### Fusion method
![](https://github.com/exceptionLi/imagefusion_deeplearning/blob/master/framework/framework_method.png)

### Fusion detail parts
![](https://github.com/exceptionLi/imagefusion_deeplearning/blob/master/framework/fusion_detail.png)

### Multi-layers fusion strategy
![](https://github.com/exceptionLi/imagefusion_deeplearning/blob/master/framework/fusion_strategy.png)

### Quality metric - Nabf
Nabf - 'B. K. Shreyamsha Kumar. Multifocus and Multispectral Image Fusion based on Pixel Significance using Discrete Cosine Harmonic Wavelet Transform. Signal, Image and Video Processing, 2012.'
![](https://github.com/exceptionLi/imagefusion_deeplearning/blob/master/framework/Nabf.png)

## Abstract
In recent years, deep learning has become a very active research tool which is used in many image processing fields. 

In this paper, we propose an effective image fusion method using a deep learning framework to generate a single image which contains all the features from infrared and visible images. First, the source images are decomposed into base parts and detail content. Then the base parts are fused by weighted-averaging. For the detail content, we use a deep learning network to extract multi-layer features. Using these features, we use l_1-norm and weighted-average strategy to generate several candidates of the fused detail content. Once we get these candidates, the max selection strategy is used to get final fused detail content. 

Finally, the fused image will be reconstructed by combining the fused base part and detail content. The experimental results demonstrate that our proposed method achieves state-of-the-art performance in both objective assessment and visual quality.


## Experimental Setting

We use MatConvNet and [VGG-19](https://pan.baidu.com/s/1eSgxtyM) to extract multi-layers strategy.

If you have any question about this code, feel free to reach me(hui_li_jnu@163.com) 


# Citation
```
@misc{li2018IVimagefusion_deeplearning,
    author = {Hui Li},
    title = {CODE: Infrared and Visible Image Fusion using a Deep Learning Framework},
    year = {2017},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/exceptionLi/imagefusion_deeplearning}}
  }
```
