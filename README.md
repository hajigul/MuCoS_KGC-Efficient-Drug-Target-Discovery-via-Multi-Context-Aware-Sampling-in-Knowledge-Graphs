# MuCoS_KGC-Efficient-Drug-Target-Discovery-via-Multi-Context-Aware-Sampling-in-Knowledge-Graphs

# MuCoS: Efficient Drug-Target Discovery via Multi-Context-Aware Sampling in Knowledge Graphs

**Official PyTorch Implementation** of the MuCoS model presented in:

> **MuCoS: Efficient Drug-Target Discovery via Multi-Context-Aware Sampling in Knowledge Graphs**  
> Haji Gul<sup>a</sup>, Abdul Ghani Naim<sup>b</sup>, Ajaz Ahmad Bhat<sup>*1</sup>  
> School of Digital Science, Universiti Brunei Darussalam  
> **BioNLP 2025** (Proceedings of the 24th Workshop on Biomedical Language Processing)

**Paper Link**: [https://aclanthology.org/search/?q=MuCos-KGC](https://aclanthology.org/search/?q=MuCos-KGC)

---

## 📋 Abstract

Accurate prediction of drug-target interactions is critical for accelerating drug discovery. MuCoS frames drug-target prediction as a link prediction task on heterogeneous biomedical knowledge graphs (KG) and introduces **Multi-Context-Aware Sampling** (MuCoS). It prioritizes high-density neighbours to capture salient structural patterns and integrates them with contextual embeddings from BERT (or DistilBERT).  

By unifying structural and textual modalities and selectively sampling highly informative patterns, MuCoS **eliminates negative sampling**, significantly reduces computational overhead, and improves generalization to unseen drug-target pairs.

**Key Results**:
- **+13% MRR** on general relation prediction (KEGG50k)
- **+22% MRR** on PharmKG-8k
- **+6% MRR** on dedicated drug-target relation prediction (KEGG50k)

---

## ✨ Features

- **Density-based Multi-Context Sampling** (Head, Tail, and Relation contexts)
- **BERT-based sequence classification** (supports `bert-base-uncased`, DistilBERT, RoBERTa)
- **No negative sampling** required
- **Efficient one-hop neighbour extraction** with top-k degree/density filtering
- **Link Prediction** (relation prediction) and **Tail Prediction** support
- **Modular, clean PyTorch + Hugging Face** implementation
- **Reproducible results** with automatic test-results logging

---

## 📊 Model Pipeline

### Figure 1: MuCoS Overall Pipeline
![Figure 1: MuCoS Pipeline](figures/model1.png)  
*Input sequence construction for BERT: `[h] [SEP] Hc [SEP] t [SEP] Tc` (link prediction) or `[h] [SEP] Hc [SEP] r [SEP] Rc` (tail prediction).*

### Figure 2: Head Context (Hc) Construction with Sampling
![Figure 2: Head Context Sampling](figures/hc.png)  
*One-hop head neighbours → density-based top-n sampling → optimized Hc.*

### Figure 3: Relation Context (Rc) Construction with Sampling
![Figure 3: Relation Context Sampling](figures/Rc_n.png)  
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
