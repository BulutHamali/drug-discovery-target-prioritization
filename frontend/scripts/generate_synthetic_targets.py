#!/usr/bin/env python3
"""Generate a synthetic targets.json demo artifact with 500 realistic drug target gene entries."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

# Seed for reproducibility
SEED = 42

# Known drug targets (clinical phase ≥1) — assign top ranks 1-50
KNOWN_TARGETS = [
    "PCSK9", "VEGFA", "EGFR", "ERBB2", "TNF", "IL6R", "BTK", "JAK1", "JAK2", "BRAF",
    "KRAS", "PIK3CA", "PTEN", "TP53", "BRCA1", "BRCA2", "MTOR", "CDK4", "CDK6", "PDCD1",
    "CD274", "CTLA4", "IL17A", "IL4R", "IL13", "TSLP", "CALCA", "GLP1R", "ANGPTL3", "LDLR",
    "HMGCR", "SLC5A2", "DPP4", "GCGR", "INSR", "IGF1R", "FGFR1", "FGFR2", "FGFR3", "FGFR4",
    "MET", "ALK", "ROS1", "RET", "NTRK1", "NTRK2", "NTRK3", "ERBB3", "AKT1", "MAPK1",
]

# Realistic unlabeled genes (label=0)
REALISTIC_UNLABELED = [
    # Ion channels
    "KCNMA1", "KCNQ1", "KCNQ2", "KCNQ3", "KCNQ4", "KCNH2", "KCNJ11",
    "SCN1A", "SCN2A", "SCN5A", "SCN8A", "SCN9A", "SCN10A", "SCN11A",
    # Polycystic kidney disease
    "PKD1", "PKD2", "PKHD1",
    # HLA genes
    "HLA-A", "HLA-B", "HLA-C", "HLA-DRA", "HLA-DRB1", "HLA-DQA1", "HLA-DQB1",
    # Cystic fibrosis
    "CFTR",
    # Mucins
    "MUC1", "MUC2", "MUC4", "MUC5B", "MUC16",
    # Collagens
    "COL1A1", "COL1A2", "COL2A1", "COL3A1", "COL4A1", "COL5A1", "COL6A1", "COL7A1",
    # Fibrillins
    "FBN1", "FBN2",
    # Neurofibromatosis
    "NF1", "NF2",
    # TSC
    "TSC1", "TSC2",
    # VHL
    "VHL",
    # APC
    "APC",
    # MMR genes
    "MLH1", "MSH2", "MSH6", "PMS2",
    # Other tumor suppressors
    "RB1", "CDKN2A", "CDKN2B", "ATM", "CHEK2", "PALB2", "RAD51C", "RAD51D",
    # BRCA-related
    "BARD1", "BRIP1", "NBN",
    # Kinases
    "AURKA", "AURKB", "PLK1", "PLK4", "WEE1", "CHK1",
    # Chromatin
    "EZH2", "KDM5C", "KDM6A", "ARID1A", "SMARCA4", "SMARCB1",
    # Splicing
    "SF3B1", "U2AF1", "SRSF2",
    # RAS pathway
    "NRAS", "HRAS", "RAF1", "MAP2K1", "MAP2K2",
    # PI3K pathway
    "PIK3R1", "PIK3CB", "AKT2", "AKT3", "TSC1", "PHLPP1",
    # Apoptosis
    "BCL2", "BCL2L1", "MCL1", "BAX", "BCLAF1",
    # WNT pathway
    "CTNNB1", "AXIN1", "AXIN2", "RNF43", "ZNRF3",
    # NOTCH pathway
    "NOTCH1", "NOTCH2", "JAG1", "DLL3", "RBPJ",
    # Hedgehog
    "PTCH1", "SMO", "GLI1", "GLI2",
    # Metabolism
    "IDH1", "IDH2", "FH", "SDHA", "SDHB",
    # DNA damage response
    "POLE", "POLD1", "MSH3", "EXO1",
    # Receptor tyrosine kinases
    "KIT", "PDGFRA", "PDGFRB", "CSF1R", "FLT3", "AXL",
    # GPCRs
    "ADRB2", "ADRB1", "ADRA1A", "HTR2A", "DRD2", "CHRM1",
    # Proteases
    "ADAM10", "ADAM17", "MMP2", "MMP9", "MMP14",
    # Immune checkpoints
    "TIGIT", "LAG3", "HAVCR2", "SIGLEC7", "CD47",
    # Cytokine receptors
    "IL2RA", "IL2RB", "IL7R", "IL15RA", "IFNGR1",
    # Cytokines
    "IL1B", "IL18", "CXCL8", "CCL2", "TGFB1",
    # Nuclear receptors
    "ESR1", "ESR2", "AR", "PGR", "NR3C1",
    # Transcription factors
    "MYC", "MYCN", "RUNX1", "ETV6", "PAX5",
    # Metabolic enzymes
    "LDHA", "FASN", "ACACA", "G6PD", "TYMS",
    # Cell cycle
    "CCND1", "CCNE1", "CCNB1", "CDC20", "BUB1",
    # Epigenetics
    "DNMT1", "DNMT3A", "DNMT3B", "TET1", "TET2",
    # Ubiquitin
    "MDM2", "MDM4", "FBXW7", "VHL", "SPOP",
    # Others
    "STAT3", "STAT5A", "NFKB1", "RELA", "JUN",
    "HDAC1", "HDAC2", "HDAC3", "HDAC6", "BRD4",
    "HSP90AA1", "HSP90AB1", "HSPA1A", "TRAP1",
    "XPO1", "IPO7", "RANGAP1",
    "PARP1", "PARP2", "PARG", "TNKS", "TNKS2",
    "TOPO1", "TOP2A", "TOP2B",
    "CDK1", "CDK2", "CDK7", "CDK8", "CDK9",
    "RAC1", "RHOA", "CDC42", "PAK1", "PAK4",
    "SRC", "YES1", "FYN", "LYN", "LCK",
    "ZAP70", "ITK", "TEC", "BMX",
    "IKBKB", "IKBKG", "NEMO",
    "RIPK1", "RIPK3", "MLKL",
    "CASP1", "CASP3", "CASP8", "CASP9",
    "NLRP3", "NLRP1", "AIM2", "CARD8",
    "STING1", "CGAS", "IRF3", "IRF7",
    "TLR2", "TLR4", "TLR7", "TLR9",
    "MYD88", "TRIF", "IRAK4", "TRAF6",
    "CXCR4", "CXCR5", "CCR2", "CCR5", "CCR7",
    "PTPN11", "PTPN1", "PTPN6", "PTPRC",
    "PRMT1", "PRMT5", "EED", "SUZ12",
    "KMT2A", "KMT2C", "KMT2D", "NSD1", "NSD2",
    "SETD2", "SETDB1", "KDM1A", "KDM4A",
    "BRCA2", "FANCA", "FANCD2", "FANCL",
    "WRN", "BLM", "RECQL4", "HELQ",
    "TERC", "TERT", "DKC1", "RTEL1",
]

# Remove duplicates while preserving order
seen: set[str] = set()
unique_unlabeled = []
for g in REALISTIC_UNLABELED:
    if g not in seen and g not in set(KNOWN_TARGETS):
        seen.add(g)
        unique_unlabeled.append(g)

# Build remainder with synthetic names
N_TOTAL = 500
N_KNOWN = len(KNOWN_TARGETS)  # 50
N_REALISTIC = min(len(unique_unlabeled), N_TOTAL - N_KNOWN)
N_SYNTHETIC = N_TOTAL - N_KNOWN - N_REALISTIC

# Deterministic pseudo-random based on seed
def lcg(seed: int, n: int) -> list[int]:
    """Simple LCG for reproducible integers."""
    results = []
    state = seed
    for _ in range(n):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        results.append(state)
    return results

synthetic_nums = lcg(SEED, N_SYNTHETIC)
synthetic_genes = [f"GENE_{n % 90000 + 10000}" for n in synthetic_nums]

# All 500 entries: known (rank 1-50) + unlabeled (rank 51-500)
all_unlabeled = (unique_unlabeled[:N_REALISTIC] + synthetic_genes)[:N_TOTAL - N_KNOWN]


def score_for_rank(rank: int, total: int = 500) -> float:
    """Smooth score decay: top ranks 0.85-0.99, rest 0.01-0.84."""
    if rank <= N_KNOWN:
        # Known targets: scores 0.99 down to 0.85
        return round(0.9912 - (rank - 1) * (0.9912 - 0.8503) / (N_KNOWN - 1), 4)
    else:
        # Unlabeled: smooth exponential decay from ~0.84 to ~0.01
        t = (rank - N_KNOWN - 1) / (total - N_KNOWN - 1)
        raw = 0.84 * math.exp(-3.5 * t)
        return round(max(0.0101, raw), 4)


# Assign fold_idx deterministically
def fold_for_rank(rank: int) -> int:
    rng = lcg(SEED + rank, 1)[0]
    return (rng % 5)


targets = []

# Known targets (rank 1-50, label=1)
for i, symbol in enumerate(KNOWN_TARGETS):
    rank = i + 1
    targets.append({
        "symbol": symbol,
        "score": score_for_rank(rank),
        "label": 1,
        "rank": rank,
        "fold_idx": fold_for_rank(rank),
    })

# Unlabeled genes (rank 51-500, label=0)
for i, symbol in enumerate(all_unlabeled):
    rank = N_KNOWN + i + 1
    targets.append({
        "symbol": symbol,
        "score": score_for_rank(rank),
        "label": 0,
        "rank": rank,
        "fold_idx": fold_for_rank(rank),
    })

payload = {
    "generated_at": "2026-08-07T00:00:00+00:00",
    "source": "synthetic-demo-v1",
    "feature_set": "biology_only",
    "rows": len(targets),
    "targets": targets,
}

output_path = Path(__file__).parent.parent / "public" / "data" / "targets.json"
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(payload, indent=2) + "\n")
print(f"Wrote {output_path} ({len(targets):,} targets, {N_KNOWN} known, {len(targets) - N_KNOWN} unlabeled)")
