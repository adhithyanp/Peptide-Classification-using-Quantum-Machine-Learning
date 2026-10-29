import pandas as pd
import numpy as np
from Bio import SeqIO
from collections import Counter
import os


AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")

#Kyte-Doolittle hydroprathy table

HYDRO = {
    'A':1.8,'C':2.5,'D':-3.5,'E':-3.5,'F':2.8,'G':-0.4,
    'H':-3.2,'I':4.5,'K':-3.9,'L':3.8,'M':1.9,'N':-3.5,
    'P':-1.6,'Q':-3.5,'R':-4.5,'S':-0.8,'T':-0.7,'V':4.2,
    'W':-0.9,'Y':-1.3
}

POSITIVE = set("KRH")
NEGATIVE = set("DE")


# FEATURE EXTRACTION

def extract_features(seq):
    seq = seq.upper()
    L = len(seq)
    counts = Counter(seq)

    # Amino acid composition (20)
    aac = [counts[a] / L for a in AA_LIST]

    pos = sum(seq.count(x) for x in POSITIVE)
    neg = sum(seq.count(x) for x in NEGATIVE)

    net_charge = pos - neg
    ratio = pos / (neg + 1e-6)
    hydro = np.mean([HYDRO.get(a,0) for a in seq])

    return aac + [net_charge, ratio, hydro]


def load_fasta(path, label):
    data = []
    for rec in SeqIO.parse(path, "fasta"):
        feat = extract_features(str(rec.seq))
        data.append(feat + [label])
    return data



# BUILD DATASET

def build_dataset():
    print("Generating features...")

    pos_main = load_fasta("data/raw/hemopi2_main_positive.fasta", 1)
    neg_main = load_fasta("data/raw/hemopi2_main_negative.fasta", 0)

    pos_val = load_fasta("data/raw/hemopi2_validation_positive.fasta", 1)
    neg_val = load_fasta("data/raw/hemopi2_validation_negative.fasta", 0)

    cols = AA_LIST + ["net_charge","pos_neg_ratio","hydrophobicity","label"]

    train_df = pd.DataFrame(pos_main + neg_main, columns=cols)
    test_df  = pd.DataFrame(pos_val + neg_val, columns=cols)

    os.makedirs("data/processed", exist_ok=True)

    train_df.to_csv("data/processed/train_features.csv", index=False)
    test_df.to_csv("data/processed/test_features.csv", index=False)

    print("Saved train_features.csv and test_features.csv")


if __name__ == "__main__":
    build_dataset()
