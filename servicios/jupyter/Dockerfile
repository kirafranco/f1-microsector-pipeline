# JupyterLab for exploration against this project's own pins (F014).
#
# Not built from jupyter/* : every tag of jupyter/minimal-notebook,
# scipy-notebook and base-notebook was last pushed on 2023-10-20. The Jupyter
# Docker Stacks moved to quay.io/jupyter in late 2023 and the Docker Hub
# namespace was left behind, so global CLAUDE.md 2.1's example list cannot be
# followed literally -- three years without a security rebuild fails the
# *trusted* half of the source policy on its own.
#
# Not built from quay.io/jupyter/* either, and for the reason F013 hit with
# apache/spark: those images ship a conda-based Python with their own pandas,
# numpy and pyarrow. A notebook there would run different versions from src/
# and from the tests, which is precisely what a shared requirements.txt exists
# to prevent. Building from the official python image keeps one set of pins
# across the conda env, every service image and every kernel.
#
# Jupyter itself is the upstream release from PyPI, pinned.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/project \
    HOME=/home/jupyter \
    JUPYTER_RUNTIME_DIR=/tmp/jupyter-runtime

# Dependencies first, code second: editing a module must not reinstall pandas.
# `pip check` runs in the build so a pin conflict fails the image rather than
# surfacing halfway through someone's notebook.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && pip install --no-cache-dir jupyterlab==4.6.3 \
    && pip check

# A named account for uid 1000, not just the number, and a real HOME. F013
# learned this the hard way: a bare `USER 1000:1000` on this base image has no
# /etc/passwd entry, and tooling that resolves the effective user fails on
# start-up. The uid is still 1000 so files written on a bind mount are not
# root-owned on the host (global 2.1).
#
# The runtime directory is container-internal on purpose. Kernel connection
# files and ZeroMQ sockets are exactly the small-file churn that crawls on a
# Windows bind mount, which is why D2 keeps Spark's shuffle spill off it too.
RUN useradd --uid 1000 --create-home --home-dir /home/jupyter --shell /usr/sbin/nologin jupyter \
    && mkdir -p /tmp/jupyter-runtime /opt/project/notebooks \
    && chown -R 1000:1000 /tmp/jupyter-runtime /home/jupyter /opt/project/notebooks

# The image runs standalone (global 2.1); the compose mounts the live tree over
# it. The copy also makes an import error surface at build time.
COPY src/ /opt/project/src/

USER 1000:1000

EXPOSE 8888

# --notebook-dir is the bind-mounted notebooks/ folder, so the file tree in the
# browser is the versioned one and nothing else.
CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--notebook-dir=/opt/project/notebooks"]
