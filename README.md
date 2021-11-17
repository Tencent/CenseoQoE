## CenseoQoE: 视觉感知画质评价框架
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)
### 项目介绍
图像/视频在编解码、传输和显示等过程中难免引入不同类型/程度的失真导致图像质量下降。图像/视频质量评价(IVQA)的研究目标是希望模仿人类视觉感知系统，
通过算法评估图片/视频在终端用户的眼中画质主观体验的好坏，目前在视频编解码、画质增强、画质监控、推荐算法、竞品分析等领域有广泛的应用。

CenseoQoE 提供图像/视频画质评价算法模型训练到应用落地的完整方案，主要包括 [CenseoQoE-Algorithm](./CenseoQoE-Algorithm) 和 [CenseoQoE-SDK](./CenseoQoE-SDK)两部分。
CenseoQoE-Algorithm 是通用的画质评价算法模型的训练框架，CenseoQoE-SDK集成了针对多种业务场景训练好的画质评价模型，可直接对UGC视频、PGC视频、游戏视频等做无参考/有参考的画质评价，实现开箱即用。

为什么叫**CenseoQoE**？[Censeo](https://en.wiktionary.org/wiki/censeo) 在拉丁语中是我认为/我觉得的意思，与图像/视频画质体验评价领域的主观评测概念相符，QoE则是体验质量（Quality of Experience）。

### 主要特性
- **[CenseoQoE-Algorithm](./CenseoQoE-Algorithm)**
    提供图像/视频画质评价算法模型训练的codebase，集成自研的模型以及业界开源的算法模型，支持无参考和有参考。此外，我们会公开不同业务场景下
    （UGC视频、PGC视频、游戏视频、直播视频等）的预训练模型，用户可以利用这些预训练模型在自己的数据集进行微调优化。
- **[CenseoQoE-SDK](./CenseoQoE-SDK)**
    集成了基于CenseoQoE-Algorithm在我们自研的数据集上训练得到的多个业务场景下的画质评价模型，全部使用c++实现，编译之后可直接
    对UGC视频、PGC视频、游戏视频等做画质评价，实现开箱即用。
    
### 快速开始
- 如果你想做模型的训练或是优化改进，请参见[CenseoQoE-Algorithm](./CenseoQoE-Algorithm/README.md)。
- 如果你想快速使用CenseoQoE推出的画质评价工具/SDK，请参见[CenseoQoE-SDK](./CenseoQoE-SDK/README.md)。

### Demo
以下是我们针对游戏视频训练的两个模型（resnet-based和shufflenet-based）和vmaf的两个模型（vmaf-default和vmaf-phone）在同一个视频上的质量分数对比。

![our model vs vmaf](./assets/com1.gif)

### 常见问题

请参考 [FAQ](docs/faq.md) 了解其他用户的常见问题。

### 许可
该项目使用的开源协议为 [MIT](./LICENSE.txt)。

### 引用
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

### 参与贡献
我们非常欢迎用户对于 CenseoQoE 做出的任何贡献，可以参考 [CONTRIBUTION.md](./CONTRIBUTING.md) 文件了解更多细节。

### 致谢
我们感谢所有为该项目提供建议、数据、模型、使用样例和新功能支持的贡献者，以及提供宝贵反馈的用户。