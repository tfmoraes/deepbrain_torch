FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-runtime

ADD requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir --requirement /tmp/requirements.txt && \
      rm -rf /tmp/requirements.txt

USER ${USER}
WORKDIR /workspace
