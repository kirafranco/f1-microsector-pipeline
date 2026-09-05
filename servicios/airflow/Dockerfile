# Airflow with this project's pipeline installed (F006).
#
# The first Dockerfile in the repository, so it also brings the .dockerignore
# obligation (global CLAUDE.md 2.1). Its build context is the project root,
# because the image has to carry src/ and requirements.txt.
#
# Airflow core declares no pandas, numpy, psycopg or pyarrow requirement --
# checked against apache-airflow-core 3.3.1's metadata, which names only
# sqlalchemy from that family -- so installing this project's pins on top
# overrides nothing Airflow needs. Its constraints file suggests other
# versions; a constraint is not a requirement.
FROM apache/airflow:3.3.1-python3.12

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/airflow/project

# Dependencies first, code second: editing a module must not reinstall pandas.
COPY --chown=airflow:root requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# The compose bind-mounts the project over this at runtime. The copy is still
# required so the image runs standalone (global 2.1), and so an import error
# surfaces at build time rather than on first schedule.
COPY --chown=airflow:root src/ /opt/airflow/project/src/
COPY --chown=airflow:root servicios/postgres/migrations/ /opt/airflow/project/servicios/postgres/migrations/

COPY --chown=airflow:root servicios/airflow/entrypoint.sh /opt/airflow/entrypoint-f1.sh

# The official image is built around its own airflow user (uid 50000) and its
# entrypoint expects it, so `USER 1000:1000` from global 2.1 does not apply
# here. The reason for that rule -- bind-mounted files ending up root-owned on
# the host -- does not arise on Docker Desktop for Windows, which does not
# surface container ownership onto the host filesystem. Declared in CLAUDE.md.
USER airflow

ENTRYPOINT ["/opt/airflow/entrypoint-f1.sh"]
CMD ["standalone"]
