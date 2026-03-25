# MuCoS_KGC-Efficient-Drug-Target-Discovery-via-Multi-Context-Aware-Sampling-in-Knowledge-Graphs

# MuCoS: Efficient Drug-Target Discovery via Multi-Context-Aware Sampling in Knowledge Graphs

**Official PyTorch Implementation** of the MuCoS model presented in:

> **MuCoS: Efficient Drug-Target Discovery via Multi-Context-Aware Sampling in Knowledge Graphs**  
> Haji Gul<sup>a</sup>, Abdul Ghani Naim<sup>b</sup>, Ajaz Ahmad Bhat<sup>*1</sup>  
> School of Digital Science, Universiti Brunei Darussalam  
> **BioNLP 2025** (Proceedings of the 24th Workshop on Biomedical Language Processing)

**Paper Link**: [https://aclanthology.org/search/?q=MuCos-KGC](https://aclanthology.org/search/?q=MuCos-KGC)

---

## Abstract
Accurate prediction of drug–target interactions is critical for accelerating drug discovery. In this work, we frame drug–target prediction as a link prediction task on heterogeneous biomedical knowledge graphs (KG) that integrate drugs, proteins, diseases, pathways, and other relevant entities. Conventional KG embedding methods such as TransE and ComplExSE are hindered by their reliance on computationally intensive negative sampling and their limited generalization to unseen drug–target pairs. To address these challenges, we propose Multi-Context-Aware Sampling (MuCoS), a novel framework that prioritizes high-density neighbours to capture salient structural patterns and integrates these with contextual embeddings derived from BERT. By unifying structural and textual modalities and selectively sampling highly informative patterns, MuCoS circumvents the need for negative sampling, significantly reducing computational overhead while enhancing predictive accuracy for novel drug–target associations and drug targets. Extensive experiments on the KEGG50k and PharmKG-8k datasets demonstrate that MuCoS outperforms baselines, achieving up to a 13% improvement in MRR for general relation prediction on KEGG50k, a 22% improvement on PharmKG-8k, and a 6% gain in dedicated drug–target relation prediction on KEGG50k


By unifying structural and textual modalities and selectively sampling highly informative patterns, MuCoS **eliminates negative sampling**, significantly reduces computational overhead, and improves generalization to unseen drug-target pairs.

**Key Results**:
- **+13% MRR** on general relation prediction (KEGG50k)
- **+22% MRR** on PharmKG-8k
- **+6% MRR** on dedicated drug-target relation prediction (KEGG50k)

---

##  Features

- **Density-based Multi-Context Sampling** (Head, Tail, and Relation contexts)
- **BERT-based sequence classification** (supports `bert-base-uncased`, DistilBERT, RoBERTa)
- **No negative sampling** required
- **Efficient one-hop neighbour extraction** with top-k degree/density filtering
- **Link Prediction** (relation prediction) and **Tail Prediction** support
- **Modular, clean PyTorch + Hugging Face** implementation
- **Reproducible results** with automatic test-results logging

---

##  Model Pipeline

### Figure 1: MuCoS Overall Pipeline
![Figure 1: MuCoS Pipeline](model1.png)  
*Input sequence construction for BERT: `[h] [SEP] Hc [SEP] t [SEP] Tc` (link prediction) or `[h] [SEP] Hc [SEP] r [SEP] Rc` (tail prediction).*

### Figure 2: Head Context (Hc) Construction with Sampling
![Figure 2: Head Context Sampling](hc.png)  
*One-hop head neighbours → density-based top-n sampling → optimized Hc.*

### Figure 3: Relation Context (Rc) Construction with Sampling
![Figure 3: Relation Context Sampling](Rc_n.png)  
*All entity pairs connected by relation r → top-k high-density pairs → optimized Rc.*

**Place the three figures** (`figure1.png`, `figure2.png`, `figure3.png`) inside a `figures/` folder in the repository root.

---

## Project Structure

```bash
MuCoS/
├── main.py                 # Entry point + configuration
├── train.py                # Training & evaluation loop
├── dataset.py              # KGDataset with context sampling
├── utils.py                # Data loading, neighbour extraction, result saving
├── model.py                # Tokenizer & model class selector
├── requirements.txt
├── figures/                # ← Put Figure 1, 2, 3 here
│   ├── figure1.png
│   ├── figure2.png
│   └── figure3.png
└── README.md
---

## Installation
git clone <your-repo-url>
cd MuCoS

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate    # Linux/Mac
# venv\Scripts\activate     # Windows

pip install -r requirements.txt


@inproceedings{gul2025mucos,
  title     = {MuCoS: Efficient Drug-Target Discovery via Multi-Context-Aware Sampling in Knowledge Graphs},
  author    = {Haji Gul and Abdul Ghani Naim and Ajaz Ahmad Bhat},
  booktitle = {Proceedings of the 24th Workshop on Biomedical Language Processing (BioNLP 2025)},
  year      = {2025},
  url       = {https://aclanthology.org/search/?q=MuCos-KGC}
}
