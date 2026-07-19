FROM python:3.11-slim
# awscli added for the awsbatch profile: Nextflow's awsbatch executor runs
# `aws s3 cp` INSIDE the container to stage .command.run and push outputs,
# so BURDEN/COLLECT need the CLI too, not just numpy/pyarrow/pandas. Reuses
# this image's existing Python/pip, so it stays a normal pip install rather
# than bundling a separate CLI runtime (see pipeline/docker/bcftools-batch/
# Dockerfile for the base image that had no Python at all to reuse).
RUN pip install --no-cache-dir "numpy<2" pyarrow pandas awscli
