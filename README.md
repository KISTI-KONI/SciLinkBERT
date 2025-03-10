
# SciLinkBERT: A Language Model for Understanding Scientific Texts with Citation Information

## Introduction

SciLinkBERT is a BERT-based pre-trained language model specifically designed to enhance the understanding of scientific texts by incorporating citation information. This model is particularly useful in scientific domains, where understanding complex language and extracting meaningful information from citations is crucial.
[[PDF]](https://arxiv.org/pdf/.pdf)
[[HuggingFace Models]](https://huggingface.co/yujuyeon/SciLinkBERT)

To use these models in 🤗 Transformers:
```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained('yujuyeon/SciLinkBERT')
model = AutoModel.from_pretrained('yujuyeon/SciLinkBERT')
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)
```
## Pretraining SciLinkBERT

Before fine-tuning on downstream tasks, you can pretrain SciLinkBERT on large-scale scientific corpora using distributed training. The following steps describe how to launch pretraining with PyTorch's distributed launch utility.

### Distributed Pretraining Setup

Ensure your cluster environment is correctly configured for distributed training. Adjust the parameters (e.g., `--nnodes`, `--nproc_per_node`, `--node_rank`, `--master_addr`, and `--master_port`) to match your system setup.

#### On Node 0

```bash
python -m torch.distributed.launch --nnodes=2 --nproc_per_node=2 --node_rank=0 --master_addr=203.250.238.31 --master_port=23456 ddp_training_ebert.py
```
#### On Node 1
```bash
python -m torch.distributed.launch --nnodes=2 --nproc_per_node=2 --node_rank=1 --master_addr=203.250.238.31 --master_port=23456 ddp_training_ebert.py
```

## Fine-Tuning on SciERC & GENIA

### Setting
```bash
git clone https://github.com/dwadden/dygiepp
conda create -n dygiepp python==3.8
conda activate dygiepp
pip install -r requirements.txt
pip install torch==1.12.0+cu102 torchvision==0.13.0+cu102 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu102
```

### Download the SciERC & GENIA data
```bash
bash ./scripts/data/get_scierc.sh
bash ./scripts/data/get_genia.sh

```

### Train the model
```bash
bash scripts/train.sh scierc
bash scripts/train genia
```

## Fine-Tuning on BLURB & MedQA

BLURB is a biomedical language understanding benchmark suite. You can fine-tune SciLinkBERT on tasks such as NER, RE, etc.

### Setting
```bash
git clone [https://github.com/dwadden/dygiepp](https://github.com/michiyasunaga/LinkBERT)
conda create -n linkbert python=3.8
source activate linkbert
pip install torch==1.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install transformers==4.9.1 datasets==1.11.0 fairscale==0.4.0 wandb sklearn seqeval
```

### Download the BLURB & MedQA data
You can download the preprocessed datasets on which we evaluated LinkBERT from [**[here]**](https://nlp.stanford.edu/projects/myasu/LinkBERT/data.zip).
- [BLURB](https://microsoft.github.io/BLURB/) biomedical NLP datasets (PubMedQA, BioASQ, HoC, Chemprot, PICO, etc.)
- [MedQA-USMLE](https://github.com/jind11/MedQA) biomedical reasoning dataset.


### Train the model
```bash
cd src/
run_examples_blurb_scilinkbert-base.sh
run_examples_medqa_scilinkbert-base.sh
```

## References and Additional Resources

For more advanced fine-tuning and implementation details, you can refer to the following repositories:

- [LinkBERT](https://github.com/michiyasunaga/LinkBERT): Provides an example of how citation links and other scientific data can be incorporated into BERT models.
- [SciDeBERTa-Fine-Tuning](https://github.com/Eunhui-Kim/SciDeBERTa-Fine-Tuning): This repository demonstrates fine-tuning approaches on scientific datasets which can be adapted for SciLinkBERT.


## How to Cite

If you use SciLinkBERT in your research, please cite the following paper:

```
@article{Yu2024SciLinkBERT,
  title={SciLinkBERT: A Language Model for Understanding Scientific Texts with Citation Information},
  author={Ju-Yeon Yu, Donghun Yang, Kyong-Ha Lee},
  journal={IEEE Access},
  year={2024},
  doi={10.1109/ACCESS.2017.DOI},
}
```

## Contributing

Contributions to SciLinkBERT are welcome! If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
