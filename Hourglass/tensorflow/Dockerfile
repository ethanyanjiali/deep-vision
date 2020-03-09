FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

LABEL maintainer="Ethan Yanjia Li"

COPY requirements.in .
RUN apt-get update && apt-get -y install python3-pip curl
RUN python3 -m pip install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.in && \
    rm requirements.in

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64

COPY *.py ./

ENTRYPOINT [ "python3", "main.py" ]