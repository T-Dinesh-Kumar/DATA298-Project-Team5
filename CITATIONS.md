# Citations & References

This document provides proper citations for all datasets, models, and research papers used in this project. Please cite these works when using this repository.

---

## 📚 Datasets

### Something-Something V2 Dataset (Model 1)

**Citation:**
```bibtex
@inproceedings{goyal2017something,
  title={The "Something Something" Video Database for Learning and Evaluating Visual Common Sense},
  author={Goyal, Raghav and Ebrahimi Kahou, Samira and Michalski, Vincent and Materzynska, Joanna and Westphal, Susanne and Kim, Heuna and Haenel, Valentin and Fruend, Ingo and Yianilos, Peter and Mueller-Freitag, Moritz and others},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  year={2017},
  pages={5842--5851}
}
```

**Dataset Link:** https://www.qualcomm.com/developer/software/something-something-v-2-dataset

**License:** Academic Research Use Only

**Usage in Project:**
- 10,000 videos used for training ModelScope (Model 1)
- Human action recognition and generation
- Training time: 9.48 hours on A100 80GB

**Important Note:** This dataset is provided by Qualcomm for academic research purposes. Commercial use requires separate licensing. Please review the terms at the official dataset page.

---

### Fashion-Text2Video Dataset (Model 2)

**Citation:**
```bibtex
@inproceedings{jiang2023text2performer,
  title={Text2Performer: Text-Driven Human Video Generation},
  author={Jiang, Yuming and Yang, Shuai and Koh, Tong Liang and Wu, Wayne and Loy, Chen Change and Liu, Ziwei},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2023}
}
```

**Dataset Link:** https://github.com/yumingj/Fashion-Text2Video

**Original Paper:** Jiang et al., "Text2Performer: Text-Driven Human Video Generation", ICCV 2023

**License:** Academic Research Use Only

**Dataset Characteristics:**
- Fashion garment videos with text descriptions
- High-quality fashion video clips
- Diverse clothing styles and garment types
- Text-video paired annotations

**Usage in Project:**
- 480 fashion videos for CogVideoX-2B fine-tuning
- Achieved 82.2% quality improvement (4.5/10 → 8.2/10)
- +100% fabric pattern accuracy, +100% dress fit, +125% sleeve details
- Training time: 62 minutes on A100 80GB

**Important Note:** This dataset is part of the Text2Performer project. Please cite the original ICCV 2023 paper when using this dataset.

---

### MSR-VTT Dataset (Model 3)

**Citation:**
```bibtex
@inproceedings{xu2016msrvtt,
  title={MSR-VTT: A Large Video Description Dataset for Bridging Video and Language},
  author={Xu, Jun and Mei, Tao and Yao, Ting and Rui, Yong},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2016},
  pages={5288--5296}
}
```

**Dataset Link:** https://huggingface.co/datasets/friedrichor/MSR-VTT

**Original Paper:** Xu et al., "MSR-VTT: A Large Video Description Dataset for Bridging Video and Language", CVPR 2016

**License:** Research Use (check original Microsoft Research terms)

**Dataset Characteristics:**
- 10,000 video clips with 200,000 captions
- Diverse content categories
- Standard benchmark for video-language tasks

**Usage in Project:**
- 200 anime-style videos selected from MSR-VTT for AnimateDiff LoRA training
- Parameter-efficient fine-tuning (16M trainable parameters)
- Training time: 8 minutes on H200 140GB
- Achievement: 30.2% temporal consistency improvement

**Important Note:** This dataset was originally released by Microsoft Research. The Hugging Face version is a redistribution. For commercial use, please verify licensing terms with Microsoft Research.

---

## 🤖 Models & Architectures

### ModelScope Text-to-Video

**Citation:**
```bibtex
@misc{modelscope,
  title={ModelScope: Text-to-Video Synthesis},
  author={Alibaba DAMO Academy},
  year={2023},
  url={https://modelscope.cn/}
}
```

**License:** Apache 2.0

**Usage:** Base architecture for Model 1 (human action generation)

---

### CogVideoX

**Citation:**
```bibtex
@article{yang2024cogvideox,
  title={CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer},
  author={Yang, Zhuoyi and Teng, Jiayan and Zheng, Wendi and Ding, Ming and Huang, Shiyu and Xu, Jiazheng and Yang, Yuanming and Hong, Wenyi and Zhang, Xiaohan and Feng, Guanyu and others},
  journal={arXiv preprint arXiv:2408.06072},
  year={2024}
}
```

**Repository:** https://github.com/THUDM/CogVideo

**License:** Apache 2.0

**Usage:** Model 2 (fashion video generation with 82.2% quality improvement)

---

### AnimateDiff

**Citation:**
```bibtex
@article{guo2023animatediff,
  title={AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning},
  author={Guo, Yuwei and Yang, Ceyuan and Rao, Anyi and Liang, Zhengyang and Wang, Yaohui and Qiao, Yu and Agrawala, Maneesh and Lin, Dahua and Dai, Bo},
  journal={International Conference on Learning Representations (ICLR)},
  year={2024}
}
```

**Repository:** https://animatediff.github.io/

**License:** Apache 2.0

**Usage:** Model 3 (anime video generation with LoRA fine-tuning)

---

### Stable Diffusion

**Citation:**
```bibtex
@inproceedings{rombach2022high,
  title={High-Resolution Image Synthesis with Latent Diffusion Models},
  author={Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj{\"o}rn},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={10684--10695},
  year={2022}
}
```

**Repository:** https://github.com/CompVis/stable-diffusion

**License:** CreativeML Open RAIL-M

**Usage:** Base model for AnimateDiff (Model 3)

---

## 📖 Foundational Research Papers

### Diffusion Models

**Denoising Diffusion Probabilistic Models (DDPM)**
```bibtex
@article{ho2020denoising,
  title={Denoising Diffusion Probabilistic Models},
  author={Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  volume={33},
  pages={6840--6851},
  year={2020}
}
```

**Denoising Diffusion Implicit Models (DDIM)**
```bibtex
@inproceedings{song2021denoising,
  title={Denoising Diffusion Implicit Models},
  author={Song, Jiaming and Meng, Chenlin and Ermon, Stefano},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2021}
}
```

---

### Parameter-Efficient Fine-Tuning

**LoRA: Low-Rank Adaptation**
```bibtex
@inproceedings{hu2022lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2022}
}
```

**Usage in Project:** Model 3 uses LoRA for parameter-efficient fine-tuning (16M trainable params, 99% reduction)

---

### Vision-Language Models

**CLIP**
```bibtex
@inproceedings{radford2021learning,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and others},
  booktitle={International Conference on Machine Learning (ICML)},
  pages={8748--8763},
  year={2021},
  organization={PMLR}
}
```

**Usage:** Text encoding for all models, evaluation metric (CLIP Score)

---

### Transformer Architecture

**Attention Is All You Need**
```bibtex
@article{vaswani2017attention,
  title={Attention Is All You Need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  volume={30},
  year={2017}
}
```

**Usage:** Foundation for CogVideoX (Model 2) transformer architecture

---

## 🛠️ Software & Libraries

### PyTorch
```bibtex
@incollection{pytorch,
  title={PyTorch: An Imperative Style, High-Performance Deep Learning Library},
  author={Paszke, Adam and Gross, Sam and Massa, Francisco and Lerer, Adam and Bradbury, James and Chanan, Gregory and Killeen, Trevor and Lin, Zeming and Gimelshein, Natalia and Antiga, Luca and others},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  pages={8026--8037},
  year={2019}
}
```

### Hugging Face Transformers
```bibtex
@inproceedings{wolf2020transformers,
  title={Transformers: State-of-the-Art Natural Language Processing},
  author={Wolf, Thomas and Debut, Lysandre and Sanh, Victor and Chaumond, Julien and Delangue, Clement and Moi, Anthony and Cistac, Pierric and Rault, Tim and Louf, R{\'e}mi and Funtowicz, Morgan and others},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations},
  pages={38--45},
  year={2020}
}
```

### Hugging Face Diffusers
```bibtex
@misc{von-platen-etal-2022-diffusers,
  title={Diffusers: State-of-the-art diffusion models},
  author={von Platen, Patrick and Patil, Suraj and Lozhkov, Anton and Cuenca, Pedro and Lambert, Nathan and Rasul, Kashif and Davaadorj, Mishig and Wolf, Thomas},
  year={2022},
  publisher={GitHub},
  howpublished={\url{https://github.com/huggingface/diffusers}}
}
```

---

## 📊 Evaluation Metrics

### Fréchet Video Distance (FVD)
```bibtex
@inproceedings{unterthiner2018towards,
  title={Towards Accurate Generative Models of Video: A New Metric \& Challenges},
  author={Unterthiner, Thomas and Van Steenkiste, Sjoerd and Kurach, Karol and Marinier, Raphael and Michalski, Marcin and Gelly, Sylvain},
  booktitle={arXiv preprint arXiv:1812.01717},
  year={2018}
}
```

### Inception Score
```bibtex
@article{salimans2016improved,
  title={Improved Techniques for Training GANs},
  author={Salimans, Tim and Goodfellow, Ian and Zaremba, Wojciech and Cheung, Vicki and Radford, Alec and Chen, Xi},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  volume={29},
  year={2016}
}
```

---

## 🎓 How to Cite This Project

If you use this code or methodology in your research, please cite:

```bibtex
@misc{sainikhil2024text2video,
  title={Multi-Model Text-to-Video Generation: A Comparative Study of Diffusion-Based Architectures},
  author={Sainikhil},
  year={2024},
  institution={San José State University},
  note={DATA 298B Master's Capstone Project},
  url={https://github.com/yourusername/text-to-video-generation}
}
```

---

## 📜 License Compliance

### Dataset Licenses
- **Something-Something V2**: Academic use only. Contact Qualcomm for commercial licensing.
- **UBC Fashion Dataset**: Academic use only. Contact UBC for other uses.

### Model Licenses
- **ModelScope**: Apache 2.0
- **CogVideoX**: Apache 2.0
- **AnimateDiff**: Apache 2.0
- **Stable Diffusion**: CreativeML Open RAIL-M (permits commercial use with restrictions)

### Library Licenses
- **PyTorch**: BSD License
- **Transformers**: Apache 2.0
- **Diffusers**: Apache 2.0

### This Project
- **License**: MIT License (see [LICENSE](LICENSE) file)
- **Commercial Use**: Permitted for the code, but respect dataset and model licenses
- **Attribution**: Please cite this project if you use or build upon it

---

## ⚠️ Important Usage Notes

### Academic Use
This project and its datasets are intended for academic research purposes. If you're using this for:
- **Academic papers**: Cite all relevant datasets and models
- **Course projects**: Follow your institution's citation guidelines
- **Thesis/Dissertation**: Include full bibliography with all citations above

### Commercial Use
If considering commercial applications:
1. **Something-Something V2**: Contact Qualcomm for commercial licensing
2. **UBC Fashion Dataset**: Contact UBC for permissions
3. **Model weights**: Most models (Apache 2.0) allow commercial use
4. **Stable Diffusion**: Review CreativeML Open RAIL-M license restrictions

### Research Ethics
- Always cite the original datasets and models
- Respect the terms of use for each component
- Acknowledge the computational resources (NVIDIA GPUs)
- Consider ethical implications of synthetic video generation

---

## 📞 Questions About Citations?

If you have questions about:
- Dataset licensing
- Citation format
- Commercial use
- Attribution requirements

**Contact:**
- Project Author: [Your Email]
- SJSU DATA 298B: [Course Instructor Email]

---

## 🔄 Updates

This citations file will be updated as new datasets, models, or papers are incorporated into the project.

**Last Updated:** November 2024

---

## 📚 Additional Resources

- **Something-Something V2 Official Page**: https://www.qualcomm.com/developer/software/something-something-v-2-dataset
- **ModelScope**: https://modelscope.cn/
- **CogVideo GitHub**: https://github.com/THUDM/CogVideo
- **AnimateDiff Project Page**: https://animatediff.github.io/
- **Hugging Face Diffusers**: https://github.com/huggingface/diffusers
- **Papers with Code**: https://paperswithcode.com/

---

**Thank you to all researchers and organizations who made their work publicly available!** 🙏
