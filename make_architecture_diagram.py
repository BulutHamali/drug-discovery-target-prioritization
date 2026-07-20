"""
Generate the architecture diagram for the README.

Requires graphviz on the system plus the diagrams package:
    brew install graphviz
    pip install diagrams

Run:
    python3 make_architecture_diagram.py

Writes figures/architecture.png. Commit both this script and the PNG, so the
diagram is reproducible rather than a one-off export from a web tool.

Numbers below are verified against the actual code and cached data, not the
original planning figures: gene_burden_features.parquet has 50,657 rows
(confirmed via `pd.read_parquet(...).shape`), of which 16,725 end up
matching the 19,296-gene universe (see README.md Results, burden coverage
86.68%); FULL_FEATURE_COLS in ml/train_eval.py has 23 entries, not a rounder
number. Two separate ECR repos feed two different process pairs
(pipeline/nextflow.config's awsbatch profile), not one registry into one
process. The temporal holdout is drawn as a separate branch off the feature
matrix because it trains on a different, older label (Open Targets 21.06)
and a restricted feature set (no pub_count, no STRING), not because it runs
later in the same pipeline.
"""

from diagrams import Diagram, Cluster, Edge
from diagrams.aws.compute import Batch, ECR
from diagrams.aws.storage import S3
from diagrams.onprem.compute import Server
from diagrams.programming.language import Python

graph_attr = {
    "fontsize": "13",
    "bgcolor": "transparent",
    "pad": "0.4",
    "nodesep": "1.4",
    "ranksep": "0.9",
    "splines": "spline",
}

with Diagram(
    "Drug-target prioritization",
    filename="figures/architecture",
    show=False,
    direction="TB",
    graph_attr=graph_attr,
):

    public_data = S3("1000 Genomes\n(AWS Open Data, us-east-1)")

    with Cluster("AWS Batch, Spot instances\n(2 ECR images: bcftools-batch for\nPREPARE/ANNOTATE, drug-target-burden\nfor BURDEN/COLLECT)"):
        prepare = Batch("PREPARE\nsubset, normalize, index")
        annotate = Batch("ANNOTATE\nbcftools csq")
        burden = Batch("BURDEN\nper-gene rare + LoF counts")
        collect = Batch("COLLECT\nmerge 22 autosomes")

        prepare >> annotate >> burden >> collect

    burden_features = S3("gene_burden_features.parquet\n50,657 rows")

    with Cluster("Local (no AWS needed)"):
        other_sources = Python("gnomAD, GTEx, DepMap,\nSTRING, UniProt,\nHGNC, gene2pubmed")
        label = Python("Open Targets\nclinical-phase label")
        matrix = Server("Feature matrix\n19,296 genes x 23 features")
        split = Python("Gene-family GroupKFold\nleakage-safe split")
        model = Python("Gradient boosting\n4-variant ablation")
        ranked = Server("Ranked target list")
        holdout = Python("Temporal holdout\n(separate: OT 21.06 to 26.06,\nrestricted feature set)")

        other_sources >> matrix
        label >> matrix
        matrix >> split >> model >> ranked
        matrix >> Edge(style="dashed") >> holdout

    public_data >> prepare
    collect >> Edge(label="16,725 / 50,657 rows\nmatch the gene universe") >> burden_features
    burden_features >> matrix
