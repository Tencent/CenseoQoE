## CenseoQoE: Image/Video Perception Quality Assessment Framework

![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)

### Introduction
Image/Video inevitably introduces different types/degrees of distortion in the process of encoding, decoding, transmission and display, resulting in the degradation of image quality.

Image/Video Quality Assessment (IVQA) research aims to imitate human visual perception system and evaluate the quality of image/video subjective experience in the eyes of end users through algorithms. At present, IVQA has been widely used in video codec, image quality enhancement, image quality monitoring, recommendation algorithm, competitive product analysis and other fields.

CenseoQoE provides a complete solution from image/video quality assessment algorithm model training to application implementation, mainly including [CenseoQoE-Algorithm](./CenseoQoE-Algorithm) and [CenseoQoE-SDK](./CenseoQoE-SDK).


CenseoQoE-SDK integrates image quality assessment model trained for various business scenarios, which can directly evaluate UGC video, PGC video, game video and other image quality without reference or with reference, realizing out-of-the-box.

Why is it called **CenseoQoE**？[Censeo](https://en.wiktionary.org/wiki/censeo)? Censeo is Latin for I think/I feel, which corresponds to the concept of subjective assessment in the field of image/video quality experience evaluation, while QoE stands for Quality of Experience.


### News
Our two papers based on CenseoQoE have been accepted by [ICASSP 2022](https://2022.ieeeicassp.org/)
- [Subjective and Objective Quality Assessment of Mobile Gaming Video](https://arxiv.org/abs/2103.05099)
- [Considering user agreement in learning to predict the aesthetic quality](https://arxiv.org/abs/2110.06956)

Please refer to these two papers for more detailed information. More released pretrained models can be found at [CenseoQoE-Algorithm](./CenseoQoE-Algorithm).

If our code is helpful to your research, please consider citing these two papers.


### Key Properties
- **[CenseoQoE-Algorithm](./CenseoQoE-Algorithm)**
    provides codebase for image/video quality assessment algorithm model training, integration of self-developed models and industry open source algorithm models, support no reference and reference. In addition, we expose pre-training models for different business scenarios (UGC video, PGC video, game video, live video, etc.) that users can use to fine-tune and optimize their own datasets.

- **[CenseoQoE-SDK](./CenseoQoE-SDK)**
  integrates the image quality assessment model of multiple business scenarios trained on our self-developed dataset based on CenseoQoE-Algorithm. All the models are implemented using C++. After compilation, the image quality assessment of UGC video, PGC video, game video and so on can be made directly, realizing out of the box.

### Quick Start
- If you want to train or optimize the model, see [CenseoQoE-Algorithm](./CenseoQoE-Algorithm/README.md) 
- If you want to quickly use CenseoQoE's image assessment tool /SDK, please see [CenseoQoE-SDK](./CenseoQoE-SDK/README.md).

### Demo
Below is a comparison of the quality scores of our two models for game video training (resnet-based and shufflenet-based) and vmaf's two models (VMAF-Default and VMAF-Phone) on the same video.

![our model vs vmaf](./assets/com1.gif)

### FAQ

Please refer to [FAQ](docs/faq.md) for frequently asked questions by other users.


### License
The project uses the open source protocol [MIT](./LICENSE.txt).

### Citation
```
@misc{wen2021strong,
      title={A strong baseline for image and video quality assessment}, 
      author={Shaoguo Wen and Junle Wang},
      year={2021},
      eprint={2111.07104},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```

### Contributing
Any CONTRIBUTING users to CenseoQoE are welcome at the [./CONTRIBUTING.md](./CONTRIBUTING.md) file for more details.

### Acknowledgements
We thank all contributors who provided suggestions, data, models, usage samples and new function supports for the project, as well as users who provided valuable feedback.