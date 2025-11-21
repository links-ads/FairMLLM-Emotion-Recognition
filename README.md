# FairMLLM Emotion Recognition

This repository contains code and resources for fair emotion recognition using multimodal large language models (MLLMs).

## EmoBox Benchmark

This project utilizes the **EmoBox** toolkit and benchmark for speech emotion recognition evaluation. EmoBox is an out-of-the-box multilingual multi-corpus speech emotion recognition toolkit that provides standardized data processing, evaluation metrics, and benchmarks for both intra-corpus and cross-corpus settings on mainstream pre-trained foundation models.

### About EmoBox

- **GitHub Repository**: [https://github.com/emo-box/EmoBox](https://github.com/emo-box/EmoBox)
- **Benchmark Website**: [https://emo-box.github.io/index.html](https://emo-box.github.io/index.html)
- **Paper**: [EmoBox: Multilingual Multi-corpus Speech Emotion Recognition Toolkit and Benchmark](https://arxiv.org/abs/2406.07162)

EmoBox includes **32 speech emotion datasets** spanning **14 distinct languages** with standardized data preparation and partitioning. The toolkit provides:

- Pre-processed metadata for all datasets
- Standardized data partitioning for fair comparison
- Evaluation utilities for both intra-corpus and cross-corpus settings
- Benchmarks on state-of-the-art pre-trained speech models

For detailed information about datasets, preparation scripts, and usage examples, please refer to the [EmoBox documentation](EmoBox/README.md).

## Sensitive Attributes by Dataset

The following table shows the availability of sensitive attributes (demographic information and transcriptions) for each dataset in the EmoBox benchmark. Values indicate: `1` = available, `0` = not available for this dataset, `-` = information not provided/unknown, `range` = age range information available instead of exact ages.

| Dataset | Age | Gender | Race | Ethnicity | Transcription |
|---------|-----|--------|------|-----------|---------------|
| aesdd | - | - | - | - | - |
| ased | 0 | 1 | 0 | 0 | 0 |
| asvp-esd | 1 | 1 | 0 | 0 | 0 |
| cafe | 1 | 1 | 0 | 0 | 0 |
| casia | - | - | - | - | - |
| crema-d | 1 | 1 | 1 | 1 | 0 |
| emns | 1 | 1 | 0 | 0 | 0 |
| emodb | - | - | - | - | 0 |
| emov-db | 0 | 1 | 0 | 0 | 1 |
| emovo | 0 | 1 | 0 | 0 | 0 |
| emozionalmente | 1 | 1 | 0 | 0 | 1 |
| enterface | - | - | - | - | 0 |
| esd | 0 | 1 | 0 | 1 | 1 |
| iemocap | 0 | 1 | 0 | 0 | 1 |
| jl-corpus | 0 | 1 | 0 | 0 | 1 |
| m3ed | range | 1 | 0 | 0 | 1 |
| mead | 0 | 1 | 0 | 0 | 0 |
| meld | 1 | 1 | 1 | 1 | 1 |
| mer2023 | - | - | - | - | - |
| mesd | 0 | 1 | 0 | 0 | - |
| msp-podcast | - | - | - | - | - |
| oreau | 1 | 1 | 0 | 0 | 1 |
| pavoque | - | - | - | - | 1 |
| polish | range | 1 | 0 | 0 | 1 |
| ravdess | 0 | 1 | 0 | 0 | 1 |
| resd | - | 1 | - | - | 1 |
| savee | range | 1 | 0 | 0 | 1 |
| shemo | 0 | 1 | 0 | 0 | 1 |
| subesco | 0 | 1 | 0 | 0 | - |
| tess | range | 1 | 0 | 0 | - |
| turev-db | 0 | 1 | 0 | 0 | - |
| urdu | 0 | 1 | 0 | 0 | 1 |

This information is crucial for fairness analysis in emotion recognition systems, as it allows researchers to evaluate model performance and bias across different demographic groups.

```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
