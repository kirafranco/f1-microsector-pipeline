# The pit-wall service: a second read-only consumer of the star schema (F017).
#
# Built from python:3.12-slim rather than apache/superset, and the reason is in
# D1. Superset was weighed first because the brief names it: 885 MB compressed
# against 259 MB, its own metadata database and SECRET_KEY to run, the global
# 2.2 RAM exception to declare -- and still no synced-hover equivalent, which is
# the one interaction this service exists to add. D1's 2026-09-04 amendment
# named the route instead: Plotly subplots sharing an x-axis, in a FastAPI
# service. Superset stays possible as a third consumer if the brief's wording
# must one day be literal; nothing here precludes it.
#
# Same base and the same shared pins as every other image here, so the figure
# is drawn from the versions the pipeline wrote the data with.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/project \
    HOME=/home/pitwall

# Dependencies first, code second: editing a route must not reinstall pandas.
# `pip check` runs in the build so a pin conflict fails the image rather than
# the first request.
COPY requirements.txt /tmp/requirements.txt
COPY requirements-pitwall.txt /tmp/requirements-pitwall.txt
RUN pip install --no-cache-dir -r /tmp/requirements-pitwall.txt \
    && pip check

# A named account for uid 1000, not just the number: F013 found that a bare
# `USER 1000:1000` on this base has no /etc/passwd entry and tooling that
# resolves the effective user fails at start-up. The uid is still 1000 so
# anything written on a bind mount is not root-owned on the host (global 2.1).
RUN useradd --uid 1000 --create-home --home-dir /home/pitwall --shell /usr/sbin/nologin pitwall

# The image runs standalone (global 2.1); the compose mounts the live tree over
# it. The copy also makes an import error surface at build time.
COPY src/ /opt/project/src/

USER 1000:1000

EXPOSE 8000

# One worker: a pit wall has one viewer, and the connection pool inside the
# process is what keeps a warm connection. More workers would multiply pools
# without shortening a request.
CMD ["uvicorn", "src.pitwall.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
