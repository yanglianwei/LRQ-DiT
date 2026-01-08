# LRQ-DiT: Log-Rotation Post-Training Quantization of Diffusion Transformers for Image and Video Generation


### Usage

We pack the quantization (including viditq methodology as a special case) related code into a standalone python package (located in `quant_utils` folder). It could be easily adapted to existing codebase, by customize a `quant_model` class inherit from the orginal model class deinition. 

# Env Setup

We recommend using conda for enviornment management. For each model in the examples folder, you could refer to the orginal codebase's readme for environment setup, we recommend using independent environment for different models since they may contain conflict package versions. 

Then, for each environment, for the support of quantization **software simulation**, you could install the `qdiff` package by locally install the package in the `./quant_utils` folder. (the `-e` is for editable installation, in case you want to modify the quantization related code.)

```
cd ./quant_utils
pip install -e .
reference resources: viditq-flux.yaml and viditq-pixart.yaml
```

# Quantization
``` bash
python fp_inference.py
python get_calib_data.py
Python quant_inference.py
bash example.sh
``` 




# Citation

If you find our work helpful, please consider citing:

```
@article{yang2025lrq,
  title={LRQ-DiT: Log-Rotation Post-Training Quantization of Diffusion Transformers for Image and Video Generation},
  author={Yang, Lianwei and Lin, Haokun and Zhao, Tianchen and Wu, Yichen and Zhu, Hongyu and Xie, Ruiqi and Sun, Zhenan and Wang, Yu and Gu, Qingyi},
  journal={arXiv preprint arXiv:2508.03485},
  year={2025}
}

@misc{zhao2024viditq,
      title={ViDiT-Q: Efficient and Accurate Quantization of Diffusion Transformers for Image and Video Generation}, 
      author={Tianchen Zhao and Tongcheng Fang and Enshu Liu and Wan Rui and Widyadewi Soedarmadji and Shiyao Li and Zinan Lin and Guohao Dai and Shengen Yan and Huazhong Yang and Xuefei Ning and Yu Wang},
      year={2024},
      eprint={2406.02540},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@article{lin2024duquant,
  title={Duquant: Distributing outliers via dual transformation makes stronger quantized llms},
  author={Lin, Haokun and Xu, Haobo and Wu, Yichen and Cui, Jingzhi and Zhang, Yingtao and Mou, Linzhan and Song, Linqi and Sun, Zhenan and Wei, Ying},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={87766--87800},
  year={2024}
}
```


# Acknowledgments
Our code was developed based on [ViDiT-Q](https://github.com/thu-nics/ViDiT-Q) and [DuQuant](https://github.com/Hsu1023/DuQuant).

# Reminder 
More implementation details can be found here: [ViDiT-Q](https://github.com/thu-nics/ViDiT-Q) and [DuQuant](https://github.com/Hsu1023/DuQuant).





