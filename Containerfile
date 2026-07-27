# agentharness — UBI10 base with the latest available Python.
#
# UBI10 has no python-3.x S2I image, so we start from the plain UBI10 base and
# install a parallel-installable interpreter from AppStream. Verified present in
# ubi10/ubi:10.0: python3.14 (3.14.5) and python3.14-pip. The system default is
# python3.12; we target 3.14 explicitly.
FROM registry.access.redhat.com/ubi10/ubi:10.0

RUN dnf -y install python3.14 python3.14-pip \
    && dnf clean all \
    && rm -rf /var/cache/dnf

WORKDIR /app

# Install dependencies first (better layer caching), then the package.
COPY pyproject.toml README.md ./
COPY src ./src
RUN python3.14 -m pip install --no-cache-dir .

# Rootless / OpenShift-friendly: run as a non-root user in group 0 with
# group-writable app/skills/workspace/config directories.
RUN mkdir -p /skills /workspace /config \
    && chgrp -R 0 /app /skills /workspace /config \
    && chmod -R g=u /app /skills /workspace /config

USER 1001

# Skills (read-only) and the workspace (read-write) are bind-mounted here at
# runtime — see README for the podman flags, including the userns mapping the
# writable mount needs under rootless podman.
ENV AGENTHARNESS_SKILLS_DIR=/skills \
    AGENTHARNESS_WORKSPACE_DIR=/workspace \
    AGENTHARNESS_CONFIG=/config/config.yaml

ENTRYPOINT ["python3.14", "-m", "agentharness"]
