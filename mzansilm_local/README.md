---
language:
- af
- en
- nso
- st
- ss
- tn
- ts
- ve
- xh
- zu
- nr
tags:
- llama
- south-african-languages
- low-resource
- decoder-only
- mzansilm
license: apache-2.0
pipeline_tag: text-generation
library_name: transformers
---

# MzansiLM 125M

**MzansiLM** is a 125M-parameter decoder-only language model trained from scratch on **MzansiText**, a multilingual corpus covering all eleven official South African languages.

[![GitHub](https://img.shields.io/badge/GitHub-Anri--Lombard/sallm-blue)](https://github.com/Anri-Lombard/sallm)
[![Paper](https://img.shields.io/badge/Paper-arXiv_2603.20732-red.svg)](https://arxiv.org/abs/2603.20732)
[![Dataset](https://img.shields.io/badge/Dataset-MzansiText-green)](https://huggingface.co/datasets/uctnlp/mzansi-text)
[![Collection](https://img.shields.io/badge/Collection-MzansiLM-orange)](https://huggingface.co/collections/uctnlp/mzansilm-and-mzansitext-6a6b3de8f1bff1af24030440)

## Model Details

- Parameters: `125,008,384`
- Architecture: decoder-only `LlamaForCausalLM`
- Hidden size: `512`
- Intermediate size: `1536`
- Layers: `30`
- Attention heads: `9`
- Key/value heads: `3`
- Context length: `2048`
- RoPE theta: `10000.0`
- RMSNorm epsilon: `1e-5`
- Tied word embeddings: `true`
- Training attention implementation: `flash_attention_2`

## Tokenizer

MzansiLM uses a custom BPE tokenizer with a vocabulary size of `65536`.

- `[BOS] = 0`
- `[EOS] = 1`
- `[PAD] = 2`
- `[UNK] = 3`
- Normalizer: `NFD`
- Pre-tokenizer: `ByteLevel`
- Post-processing:
  - single sequence: `[BOS] $A [EOS]`
  - pair sequence: `[BOS] $A [EOS] [BOS] $B [EOS]`

## Training Data

The model was trained on **MzansiText** and covers all eleven official South African languages:

`af`, `en`, `nso`, `sot`, `ssw`, `tsn`, `tso`, `ven`, `xho`, `zul`, `nbl`

Related releases:

- Paper: [arXiv:2603.20732](https://arxiv.org/abs/2603.20732)
- Raw corpus: [uctnlp/mzansi-text](https://huggingface.co/datasets/uctnlp/mzansi-text)
- Tokenized corpus: [uctnlp/mzansi-text-tokenized](https://huggingface.co/datasets/uctnlp/mzansi-text-tokenized)
- GitHub code and configs: [https://github.com/Anri-Lombard/sallm](https://github.com/Anri-Lombard/sallm)

## Intended Use

MzansiLM is a research model for pretraining, fine-tuning, and evaluation on South African languages. It is intended as a reproducible baseline for language modeling and downstream task adaptation.

## Usage

MzansiLM uses an explicit attention head dimension (`head_dim=56`) with
`hidden_size=512` and `num_attention_heads=9`. Use Transformers 4.x for this
release:

```bash
pip install "transformers>=4.52.4,<5" torch accelerate safetensors
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "uctnlp/mzansilm-125m"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

inputs = tokenizer("Namuhla sifunda ngolimi ngoba abantu", return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    pad_token_id=tokenizer.pad_token_id,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

This is a base language model, not an instruction-tuned chat model, so outputs
are plain text continuations of the prompt.

### Compatibility note

Transformers 5 currently rejects this Llama configuration during validation
because `512` is not divisible by `9`, even though the model uses the explicit
`head_dim=56` stored in `config.json`. Until that validation path supports this
configuration, use Transformers 4.x as shown above.

## Citation

Please cite the paper:

```bibtex
@misc{lombard2026mzansitextmzansilmopencorpus,
      title={MzansiText and MzansiLM: An Open Corpus and Decoder-Only Language Model for South African Languages}, 
      author={Anri Lombard and Simbarashe Mawere and Temi Aina and Ethan Wolff and Sbonelo Gumede and Elan Novick and Francois Meyer and Jan Buys},
      year={2026},
      eprint={2603.20732},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.20732}, 
}
```

## License

Apache License 2.0
