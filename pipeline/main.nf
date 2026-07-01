#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

// All tuneable parameters are declared in nextflow.config.
// Override on the command line, e.g.:
//   --chrom 21 --af_max 0.005 --samples path/to/ids.txt

// ── processes ────────────────────────────────────────────────────────────────

process PREPARE {
    // Subset to the target chromosome, split multi-allelic sites into
    // biallelic records (one ALT per line), and re-index. Splitting ensures
    // each variant has a single INFO/AF value, which simplifies AF filtering
    // in BURDEN without any multi-allelic parsing logic.
    tag "chr${params.chrom}"
    cpus   1
    memory '2 GB'

    input:
    path vcf
    path tbi

    output:
    tuple path("prepared.vcf.gz"), path("prepared.vcf.gz.tbi")

    script:
    def sample_arg = params.samples ? "--samples-file ${params.samples}" : ""
    """
    bcftools view \
        --regions ${params.chrom} \
        ${sample_arg} \
        ${vcf} \
    | bcftools norm \
        --multiallelics -any \
        --output-type z \
        --output prepared.vcf.gz
    bcftools index --tbi prepared.vcf.gz
    """
}

process ANNOTATE {
    // Tool: bcftools csq.
    //
    // Why bcftools csq and not SnpEff or VEP:
    //   SnpEff  -- Java, requires downloading a ~1 GB per-genome effect database
    //              at first run; not suitable for a lightweight validation run.
    //   VEP     -- requires a ~60 GB reference cache; operationally heavy.
    //   bcftools csq -- needs only a GFF3 (~15 MB for chr22, cached by
    //              data/fetch_ref.sh) and a chr22 FASTA (~51 MB). Runs in the
    //              same container as PREPARE. Produces gene symbol and
    //              consequence type (stop_gained, frameshift, splice_donor,
    //              splice_acceptor, start_lost, missense, synonymous, ...) in
    //              the BCSQ INFO tag, which is all BURDEN needs.
    //
    // --ncsq 30: raise the per-variant consequence cap from the default 16.
    //            Some chr22 loci have many overlapping transcripts.
    tag "chr${params.chrom}"
    cpus   2
    memory '4 GB'

    input:
    tuple path(vcf), path(tbi)
    path gff
    path fasta
    path fai

    output:
    path "annotated.vcf.gz"

    script:
    """
    bcftools csq \
        --fasta-ref ${fasta} \
        --gff-annot ${gff} \
        --ncsq 30 \
        --threads ${task.cpus} \
        --output-type z \
        --output annotated.vcf.gz \
        ${vcf}
    """
}

process BURDEN {
    // Count per-gene rare and LoF variant burden using bin/burden.py.
    // The output is a TSV; one row per gene with n_rare and n_lof columns.
    // Rare is defined as AF < params.af_max (default 0.01).
    tag "chr${params.chrom}"
    cpus   1
    memory '2 GB'

    input:
    path vcf

    output:
    path "burden_${params.chrom}.tsv"

    script:
    """
    burden.py \
        --vcf    ${vcf} \
        --af-max ${params.af_max} \
        --chrom  ${params.chrom} \
        --out    burden_${params.chrom}.tsv
    """
}

process COLLECT {
    // Merge per-chromosome burden TSVs into a single Parquet feature table
    // using bin/collect.py. This is the pipeline output consumed by the ML layer.
    publishDir params.outdir, mode: 'copy'
    cpus   1
    memory '2 GB'

    input:
    path tsv_files

    output:
    path "gene_burden_features.parquet"

    script:
    """
    collect.py --out gene_burden_features.parquet ${tsv_files}
    """
}

// ── workflow ──────────────────────────────────────────────────────────────────

workflow {
    vcf_ch   = Channel.fromPath(params.vcf,            checkIfExists: true)
    tbi_ch   = Channel.fromPath("${params.vcf}.tbi",   checkIfExists: true)
    gff_ch   = Channel.fromPath(params.gff,            checkIfExists: true)
    fasta_ch = Channel.fromPath(params.fasta,          checkIfExists: true)
    fai_ch   = Channel.fromPath("${params.fasta}.fai", checkIfExists: true)

    prepared  = PREPARE(vcf_ch, tbi_ch)
    annotated = ANNOTATE(prepared, gff_ch, fasta_ch, fai_ch)
    burden    = BURDEN(annotated)
    COLLECT(burden.collect())
}
