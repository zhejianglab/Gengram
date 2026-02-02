# Gengram: Retrieval-Augmented Genomic Foundation Models

<div align="center" style="line-height: 1;">
  <a href="https://huggingface.co/ZhejiangLab/Gengram" target="_blank">
      <img alt="Hugging Face" src="https://img.shields.io/badge/🤗%20Hugging%20Face-Gengram%20-ffc107"/>
  </a>
  <a href="https://arxiv.org/abs/7211321" target="_blank">
      <img alt="arXiv" src="https://img.shields.io/badge/📜%20arXiv-Technical Report-brightgreen?logo=Linkedin&logoColor=white"/>
  </a>
  <a href="https://github.com/zhejianglab/Gengram/blob/main/LICENSE" target="_blank">
      <img alt="License" src="https://img.shields.io/badge/📑%20License- Apache 2.0-FFC0CB"/> 
  </a>
</div>

## News

🚀 **2026.2.2  |  Paper Preprint Available**: Our paper on Gengram has been released as a preprint on arXiv. Check it out at [https://www.arxiv.org/abs/2601.22203](https://www.arxiv.org/abs/2601.22203).



## 1. Introduction

Gengram is a novel conditional memory module designed for genomic foundation models (GFMs) that introduces explicit motif memory retrieval to enhance Transformer-based DNA sequence modeling. Unlike traditional GFMs that rely on dense computation to implicitly infer multi-nucleotide motifs, Gengram provides an efficient lookup mechanism for biological patterns through a genomic-specific hashing scheme.

Figure 1 illustrates the overall architecture of Gengram, together with the evaluation pipeline used to assess its effectiveness across multiple genomic benchmarks.

![Gengram](./images/gengram_model.png)

### ✨ Key Features

- **🎯 Explicit Motif Memory**: Stores and retrieves k-mers (k=1-6) via hash-based lookup tables
- **🧬 Local Window Aggregation**: 21bp window mechanism aligned with DNA helical structure
- **⚡ Computational Efficiency**: Linear time complexity with minimal overhead
- **🔧 Architecture Agnostic**: Compatible with various attention mechanisms (MHA, GQA, MLA)
- **⚖️ Stable Training**: Improves load balancing in Mixture-of-Experts models
- **🔍 Biological Interpretability**: Learns meaningful motif representations

### ✨ Biological Interpretability
Gengram exhibits clear biologically grounded behaviors, including:
- **Reverse-complement symmetry** in memory embeddings
- **Context-dependent gating** aligned with functional regions
- **Hierarchical representation** from shallow to deep layers

## 2. Model Information

### Model Configuration

The following details the model configuration, including the parameterization of Gengram, MoE routing strategies, and training hyperparameters used across all experiments.

- **Gengram Parameters**

  These parameters control how Gengram operates within the Transformer layers, including which layers to apply it to, the n-gram sizes, and embedding dimensions.
<div align="center">

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--gengram-enabled` | Enable Gengram | `true` |
| `--gengram-layer-ids` | Layers to apply Gengram | `3 6 10` |
| `--gengram-ngram-sizes` | N-gram sizes for DNA processing | `1 2 3 4 5 6` |
| `--gengram-embed-dim-per-ngram` | Embedding dimension per n-gram | `1024` |
| `--gengram-window-size` | window size | `21` |

</div>

- **Mixture of Experts (MoE)**

  These parameters define the Mixture-of-Experts architecture, including the number of experts, routing top-k, and load balancing strategies during training.

<div align="center">

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--num-experts` | Number of experts | `8` |
| `--moe-router-topk` | Top-k experts to route to | `2` |
| `--moe-router-load-balancing-type` | Load balancing strategy | `aux_loss` |
| `--moe-aux-loss-coeff` | Auxiliary loss coefficient | `1e-3` |

</div>

- **Training Parameters**

  These parameters specify the training setup, including sequence length, batch sizes, precision, and attention optimizations.

<div align="center">

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--seq-length` | Maximum sequence length | `8192` |
| `--micro-batch-size` | Micro batch size per GPU | `1` |
| `--global-batch-size` | Global batch size across all GPUs | `1024` |
| `--bf16` | Use BF16 precision | `true` |
| `--use-flash-attn` | Enable Flash Attention | `true` |

</div>

### Pre-training Data
- **Human Sequences**: HPRC Release 2, GRCh38, CHM13
- **Non-human Primates**: NCBI RefSeq database
- **Total**: 200B tokens (8k context) + 100B tokens (32k context)

## 3. Performance Evaluation

Gengram demonstrates strong performance across multiple genomic benchmarks, achieving competitive results despite being trained on significantly fewer tokens and with a smaller model size.

<div align="center">

| Metric | [Gengram-10B](https://huggingface.co/ZhejiangLab/Gengram) | [Genos-10B](https://huggingface.co/ZhejiangLab/Genos-10B) | [Evo2-40B](https://huggingface.co/arcinstitute/evo2_40b) |
|--------|:-------------:|:-----------:|:----------:|
| **Trained Tokens** | 200B | 2.2T | 9.3T |
| **Multi-species Exon Classification** | **0.9832** | 0.9755 | 0.9332 |
| **Splice Site Identification** | 0.9009 | 0.7990 | **0.9138** |
| **Human OCR Ensembl** | **0.7714** | 0.7623 | 0.7635 |

</div>

- **Key Observations**

  - Data Efficiency: Achieves comparable performance using ~10×–40× fewer tokens
  - Motif-Dominated Tasks: Up to 14% improvement
  - Long-Context Modeling: Enhanced performance with shorter sequences
  - Training Efficiency: Better parameter utilization and stable MoE training

- **Evaluation Benchmarks**
  - Genomic Benchmarks (GB)
  - Nucleotide Transformer Benchmarks (NTB)
  - Long-Range Benchmarks (LRB)
  - Genos Benchmarks (GeB)

## 4. Quickstart

### Model Download
Gengram model is available for download from Hugging Face. Please ensure that you have sufficient disk space: at least 150 GB for the `torch_dist` version or 70 GB for the `torch` version.

<div align="center">

| **Model** | **Activated Params** | **Hugging Face** | **Format** 
|:---------:|:----------------:|:----------------:|:----------------:|
| Gengram-10B | 2.87 B | [🤗 Hugging Face](https://huggingface.co/ZhejiangLab/Gengram) | torch_dist |
| Gengram-10B | 2.87 B | [🤗 Hugging Face](https://huggingface.co/ZhejiangLab/Gengram-torch) | torch |

</div>

### Pre-training

Run the pre-training script with the following command:

```bash
cd Gengram
bash Gengram_layer3-6-10_win21_pp2.sh
```

## 5. License

This repository and the Gengram model weights are licensed under the [Apache License 2.0](LICENSE). 

Please note that the primary use of Gengram model is to support genomics research, providing researchers with advanced analytical capabilities and long-context modeling tools powered by large-scale foundation models for the human genome. It is not intended for use in any manner that violates applicable laws or regulations, nor for any activities prohibited by the license agreement.

## 6. Citation and Acknowledgements

We acknowledge the high-quality sequencing data provided by [CycloneSEQ](https://www.cyclone-seq.com/), which forms an important foundation for this work. We also appreciate the inspiration from DeepSeek's [Engram](https://github.com/deepseek-ai/Engram) module and the framework support provided by [Megatron-LM](https://github.com/NVIDIA/Megatron-LM). Model training was conducted on the [021 Science Foundation Model](https://www.zero2x.org/021) and [Zero2X](https://www.zero2x.org/) open platform.

If you use this work in your research, please cite the following paper:

```bibtex
@article{gengram,
  title={Beyond Conditional Computation: Retrieval-Augmented Genomic Foundation Models with Gengram},
  author={Genos Team and Xu, Huinan and Feng, Xuyang and Chen, Junhong and Liu Junchen and Deng, Kaiwen and Ding, Kai and Long, Shengning and Shuai, Jiaxue and Li, Zhaorong and Liu, Shiping and Xue, Guirong and Xiao, Zhan},
  journal={arXiv preprint arXiv:2601.22203},
  year={2026}
}
```

## 7. Contact

For project-related questions, please raise an [issue](https://github.com/zhejianglab/Gengram/issues) or contact the project maintainer at xz@zhejianglab.org. 

For general inquiries, you are also welcome to contact us at opensource@zhejianglab.org.
