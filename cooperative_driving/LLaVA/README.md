# LLaVA component

This directory contains the LLaVA model components and the cooperative-driving inference and evaluation entries used by FedAttn. Follow [the cooperative-driving README](../README.md) for installation, task assets, experiments and demos. The environment YAML files and `pyproject.toml` define the dependencies for this package.

The upstream project is [LLaVA](https://github.com/haotian-liu/LLaVA). Its standalone training, serving and general-purpose evaluation workflows are not included in this experiment package. The applicable third-party terms are recorded in [LICENSE](LICENSE).

## Citation


The upstream documentation provides the following citations:
```bibtex
@misc{liu2024llavanext,
    title={LLaVA-NeXT: Improved reasoning, OCR, and world knowledge},
    url={https://llava-vl.github.io/blog/2024-01-30-llava-next/},
    author={Liu, Haotian and Li, Chunyuan and Li, Yuheng and Li, Bo and Zhang, Yuanhan and Shen, Sheng and Lee, Yong Jae},
    month={January},
    year={2024}
}

@misc{liu2023improvedllava,
      title={Improved Baselines with Visual Instruction Tuning}, 
      author={Liu, Haotian and Li, Chunyuan and Li, Yuheng and Lee, Yong Jae},
      publisher={arXiv:2310.03744},
      year={2023},
}

@misc{liu2023llava,
      title={Visual Instruction Tuning}, 
      author={Liu, Haotian and Li, Chunyuan and Wu, Qingyang and Lee, Yong Jae},
      publisher={NeurIPS},
      year={2023},
}
```

## Acknowledgement

LLaVA builds on [Vicuna and FastChat](https://github.com/lm-sys/FastChat).
