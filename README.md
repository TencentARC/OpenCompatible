# Open Compatible Training Benchmark

`OpenCompatible` provides a standard compatible training benchmark, covering practical training scenarios.

Mainstreaming paradigms are easily achieved using our benchmark, which includes:
- backfilling-free backward compatibility
- hot-refresh backward compatibility
- feature-level forward compatibility (onging)

Various downstream tasks are supported:
- Content-based Image Retrieval (Google Landmark, Revisited Oxford, and Revisited Paris)
- Face Recognition (MS1M-v3 and IJB-C)
- Person Re-ID (onging)

## Published papers
- [ICLR 2022] **B. Zhang**, Y. Ge, Y. Shen, Y. Li, C. Yuan, X. Xu, Y. Wang, and Y. Shan. [Hot-Refresh Model Upgrades with Regression-Free Compatible Training in Image Retrieval](https://openreview.net/pdf?id=HTp-6yLGGX)[C]. [\[video\]]() [\[slides\]]()
- [IJCAI 2022 (oral)] 
**B. Zhang**, Y. Ge, Y. Shen, S. Su, F. Wu, C. Yuan, X. Xu, Y. Wang, and Y. Shan. [Towards Universal Backward-Compatible Representation Learning](https://arxiv.org/pdf/2203.01583.pdf)[J]. [video]() [slides]()
- [arXiv] Su S<sup>\*</sup>, **Zhang B**<sup>\*</sup>, Ge Y, et al. [Privacy-Preserving Model Upgrades with Bidirectional Compatible Training in Image Retrieval](https://arxiv.org/pdf/2204.13919.pdf)[J].

## Requirements

* Python >= 3.6
* PyTorch >= 1.6
* tensorflow >= 2.1
* termcolor
* sklearn
* faiss-gpu
* numpy
* tqdm

## Dataset Preparation
- Please refer to **[datasets.md](docs/datasets.md)** for more details.

## Train and Test
- Please refer to **[train_test.md](docs/train_test.md)** for more details.

## Model Zoo
Pre-trained models are provided (see **[model_zoo.md](docs/model_zoo.md)** )

## License

This project is licensed under the Apache v2 License.

More details are in **[LICENSE](LICENSE)**.

## Acknowledgements

This project is inspired by the
project [Pytorch-Template](https://github.com/victoresque/pytorch-template), and [OpenBCT](https://github.com/YantaoShen/openBCT).

Contributors: Binjie Zhang, Yixiao Ge, and Shupeng Su. 

## References:
```
[1] Towards backward-compatible representation learning. CVPR 2020.
[2] Learning without Forgetting. T-PAMI 2017.
[3] Positive-congruent training: Towards regression-free model updates. CVPR 2021.
```

## Citatations
``` latex
@inproceedings{zhang2021hot,
  title={Hot-Refresh Model Upgrades with Regression-Free Compatible Training in Image Retrieval},
  author={Zhang, Binjie and Ge, Yixiao and Shen, Yantao and Li, Yu and Yuan, Chun and Xu, Xuyuan and Wang, Yexin and Shan, Ying},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
```
```latex
@article{zhang2022towards,
  title={Towards Universal Backward-Compatible Representation Learning},
  author={Zhang, Binjie and Ge, Yixiao and Shen, Yantao and Su, Shupeng and Yuan, Chun and Xu, Xuyuan and Wang, Yexin and Shan, Ying},
  journal={arXiv preprint arXiv:2203.01583},
  year={2022}
}
```

## Contact
Binjie Zhang ([homepage](https://binjiezhang.github.io/)): `zbj19@tsinghua.org.cn`, Yixiao Ge ([homepage](https://geyixiao.com/)): `yixiaoge@tencent.com`.

