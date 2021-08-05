FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-runtime
ARG USER=user
ARG GROUP=user
ARG UID=1000
ARG GID=1000

ADD requirements.txt /tmp/requirements.txt

RUN groupadd -g ${GID} ${GROUP} && \
  useradd -m ${USER} --uid=${UID} --gid=${GID} && \
  chown -R ${USER}:${GROUP} /workspace && \
  pip install --no-cache-dir --requirement /tmp/requirements.txt && \
  rm -rf /tmp/requirements.txt

USER ${USER}
WORKDIR /workspace
