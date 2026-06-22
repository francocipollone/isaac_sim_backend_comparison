FROM nvcr.io/nvidia/isaac-sim:6.0.0

ENV DEBIAN_FRONTEND=noninteractive

# Why? Just because. Isaac Sim fails with missing dependency torch, and it is not included in the base image.
RUN /isaac-sim/python.sh -m pip install torch

USER root

RUN mkdir -p /ws

WORKDIR /ws

ENTRYPOINT [ "" ]
CMD [ "bash" ]
