# PREPARE/ANNOTATE, awsbatch profile only.
#
# Same bcftools biocontainer the `local` profile uses unchanged, plus the
# AWS CLI. Nextflow's awsbatch executor runs `aws s3 cp` INSIDE the
# container itself to stage .command.run and push logs/outputs, so every
# image used under -profile awsbatch needs the CLI present, not just our
# own tools.
#
# This base biocontainer (a stripped bioconda/mulled build) has no package
# manager and no Python at all: no apt, no conda, no pip. There is no lean
# apt-get/pip install path available. Rather than layering in a separate
# Python plus pip plus botocore by hand, we copy the already-built,
# self-contained AWS CLI v2 bundle out of AWS's own official image. This is
# the standard way to add the CLI to a minimal/scratch-like image without a
# package manager; it does add a real ~240 MB (the CLI's bundled Python
# runtime), there is no way to make that meaningfully smaller given this
# base image has nothing to build on top of.
FROM public.ecr.aws/aws-cli/aws-cli:2.17.62 AS awscli

FROM quay.io/biocontainers/bcftools:1.18--h8b25389_0
COPY --from=awscli /usr/local/aws-cli/ /usr/local/aws-cli/

# /usr/local/bin/aws in the source image is itself a symlink into the
# aws-cli tree. Copying a symlink BY NAME with COPY dereferences it into a
# flattened real-file copy, which breaks the CLI's $ORIGIN-relative library
# lookup (it ends up looking for its bundled libpython .so next to wherever
# the flattened copy landed, not next to the real binary). Recreate it as
# an actual symlink instead so the relative lookup still resolves correctly.
RUN ln -sf /usr/local/aws-cli/v2/current/bin/aws /usr/local/bin/aws

# The aws binary links libz.so.1, which this base image already has under
# /usr/local/lib (already on the ld.so.conf.d search path), but the
# dynamic linker cache was built before that library existed. Refresh it.
RUN ldconfig
