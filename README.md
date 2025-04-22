# Zooming In on Fakes: A Novel Dataset for Localized AI-Generated Image Detection with Forgery Amplification Approach

The Broader Region Generated (BR-Gen) dataset was proposed in the [ArXiv preprint](https://arxiv.org/abs/2504.11922) paper "Zooming In on Fakes: A Novel Dataset for Localized AI-Generated Image Detection with Forgery Amplification Approach".

## Dataset(BR-Gen)

This dataset contains150k localized generated images, forged by traditional inpainting methods (MAT, LaMa) and text-guided inpainting methods (SDXL, BrushNet, PowerPaint). We provided the Region Masks and Localized Generated Images.

### Visual Cases

![cases](figs/cases.png)



### Dataset specifications

![cases](figs/br-gen.png)

How we created 150k localized generated images using various open-source models. We used 2 types of masks, and 5 types of inpainting methods to generated these images. Not seen in the diagram: each real image will correspond to 2 masks and 10 localized generated images.

| Generated types                             |                                           |
| ------------------------------------------- | ----------------------------------------- |
| **# masks**                                 | 2 (Stuff, Background)                     |
| **# Inpainting Methods**                    | 5 (LaMa, MAT, SDXL, BrushNet, PowerPaint) |
| **Total # generated iamges per real image** | 2 * 5 = 10                                |

| Dataset sizes                    | Training | Testing | Validation | Total   |
| -------------------------------- | -------- | ------- | ---------- | ------- |
| **# real images**                | 12,000   | 1,500   | 1,500      | 15,000  |
| **# localized generated images** | 120,000  | 15,000  | 15,000     | 150,000 |

Note, in the process of training and testing, in order  to prevent the impact o category imbalance, we sample the generated images to keep the number of real samples the same.



### Download

The **BR-Gen** dataset can be downloaded through [Google Drive](https://drive.google.com/drive/folders/1lPILaotrTplG5P83cugBnKM1EwUJFA9d?usp=sharing) and [Baidu Netdisk](https://pan.baidu.com/s/1cXgXm4EefC1sCw8vwadB_w) (Password: cclp). About stuff categories and thing categories, you can consult [COCO_stuff](https://github.com/nightrome/cocostuff) for more details. If you have any questions, please send an email to [lvpancai@stu.xmu.edu.cn](mailto:lvpancai@stu.xmu.edu.cn). 



Considering copyright issues, the BR-Gen dataset only provides Region Masks and Forged Images. The original images were collected from datasets such as COCO, ImageNet, and Places. as detailed in **Section 3.1 Real Image Collection** of the paper.

| Dataset        | Download URL                                                 |
| -------------- | ------------------------------------------------------------ |
| COCO2017_train | http://images.cocodataset.org/zips/train2017.zip             |
| ImageNet       | https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar |
| Places         | [Places2: A Large-Scale Database for Scene Understanding](http://places2.csail.mit.edu/download.html) |

However, we have provided the file name of the real image used in the dataset. You can extract the real image data used in this dataset from the original real data according to "**RealImage/xxxxx/xxxxx_image_list.txt**" in the path.



### License

The BR-Gen dataset is released only for academic research. Researchers from educational institutes are allowed to use this database freely for noncommercial purposes.

## Noise-guided Foregery Amplification Vision Transformer(NFA-ViT)

To address the BR-Gen challenge and enhance performance of local AIGC detection, we introduce NFA-ViT, a noise-guided forgery amplification transformer that leverages a dual-branch architecture to diffuse forgery cues into real regions through modulated self-attention, significantly improving the detectability of small or spatially subtle forgeries.

![nfa_vit](figs/nfa_fit.png)


## References & Acknowledgements
We sincerely thank [IMDLBenCo](https://github.com/scu-zjz/IMDLBenCo) for their exploration and support.


