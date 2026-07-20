#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

// All tuneable parameters are declared in nextflow.config.
//
// Single-chromosome (smoke test), unchanged from before:
//   --chrom 21 --vcf path/to/chr21.vcf.gz --gff path/to/chr21.gff3.gz --fasta path/to/chr21.fa
//
// Multi-chromosome (batch run): set --chroms to a comma-separated list and
// the DAG fans out per chromosome, all BURDEN outputs feeding one COLLECT.
// vcf_dir/ref_dir must hold chr<N>.vcf.gz(.tbi) / chr<N>.gff3.gz / chr<N>.fa(.fai)
// per chromosome, exactly what data/fetch_1000genomes.sh and data/fetch_ref.sh
// produce:
//   --chroms 3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20,21

// ── processes ────────────────────────────────────────────────────────────────

process PREPARE {
    // Subset to the target chromosome, split multi-allelic sites into
    // biallelic records (one ALT per line), and re-index. Splitting ensures
    // each variant has a single INFO/AF value, which simplifies AF filtering
    // in BURDEN without any multi-allelic parsing logic.
    tag "chr${chrom}"
    cpus   1
    memory '2 GB'

    input:
    tuple val(chrom), path(vcf), path(tbi)

    output:
    tuple val(chrom), path("prepared.vcf.gz"), path("prepared.vcf.gz.tbi")

    script:
    def sample_arg = params.samples ? "--samples-file ${params.samples}" : ""
    """
    bcftools view \
        --regions ${chrom} \
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
    //   bcftools csq -- needs only a per-chromosome GFF3 and FASTA (cached by
    //              data/fetch_ref.sh). Runs in the same container as PREPARE.
    //              Produces gene symbol and consequence type (stop_gained,
    //              frameshift, splice_donor, splice_acceptor, start_lost,
    //              missense, synonymous, ...) in the BCSQ INFO tag, which is
    //              all BURDEN needs.
    //
    // --ncsq 30: raise the per-variant consequence cap from the default 16.
    //            Some loci have many overlapping transcripts.
    tag "chr${chrom}"
    cpus   2
    memory '4 GB'

    input:
    tuple val(chrom), path(vcf), path(tbi), path(gff), path(fasta), path(fai)

    output:
    tuple val(chrom), path("annotated.vcf.gz")

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
    tag "chr${chrom}"
    cpus   1
    memory '2 GB'

    input:
    tuple val(chrom), path(vcf)

    output:
    path "burden_${chrom}.tsv"

    script:
    """
    burden.py \
        --vcf    ${vcf} \
        --af-max ${params.af_max} \
        --chrom  ${chrom} \
        --out    burden_${chrom}.tsv
    """
}

process COLLECT {
    // Merge per-chromosome burden TSVs into a single Parquet feature table
    // using bin/collect.py. This is the pipeline output consumed by the ML layer.
    // In multi-chromosome mode this merges ALL chromosomes processed in this
    // run; running it again with a different --chroms list (or a single
    // --chrom) will overwrite this file, so downstream merges across
    // separate runs (e.g. combining with earlier chr22/chr1/chr2/chr17/chr19
    // results) still need the same manual concat step used before.
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
    if( params.chroms ) {
        chrom_list = params.chroms.toString().tokenize(',').collect { it.trim() }

        input_ch = Channel.fromList(chrom_list).map { chrom ->
            tuple(
                chrom,
                file("${params.vcf_dir}/chr${chrom}.vcf.gz",     checkIfExists: true),
                file("${params.vcf_dir}/chr${chrom}.vcf.gz.tbi", checkIfExists: true),
                file("${params.ref_dir}/chr${chrom}.gff3.gz",    checkIfExists: true),
                file("${params.ref_dir}/chr${chrom}.fa",         checkIfExists: true),
                file("${params.ref_dir}/chr${chrom}.fa.fai",     checkIfExists: true),
            )
        }
    } else {
        input_ch = Channel.of(
            tuple(
                params.chrom,
                file(params.vcf,            checkIfExists: true),
                file("${params.vcf}.tbi",   checkIfExists: true),
                file(params.gff,            checkIfExists: true),
                file(params.fasta,          checkIfExists: true),
                file("${params.fasta}.fai", checkIfExists: true),
            )
        )
    }

    prepare_input = input_ch.map { chrom, vcf, tbi, gff, fasta, fai -> tuple(chrom, vcf, tbi) }
    prepared      = PREPARE(prepare_input)

    // Join PREPARE's per-chromosome output back to that SAME chromosome's
    // gff/fasta/fai (channel .join() matches on the first tuple element,
    // chrom, so this pairs correctly even though chromosomes complete
    // PREPARE in whatever order they finish, not the order they were
    // submitted in).
    ref_by_chrom   = input_ch.map { chrom, vcf, tbi, gff, fasta, fai -> tuple(chrom, gff, fasta, fai) }
    annotate_input = prepared.join(ref_by_chrom)
    annotated      = ANNOTATE(annotate_input)

    burden = BURDEN(annotated)
    COLLECT(burden.collect())
}
