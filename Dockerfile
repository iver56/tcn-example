FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

# CUDA-related environment variables
ENV CUDA_PATH /usr/local/cuda
ENV PATH ${CUDA_PATH}/bin:$PATH
ENV LD_LIBRARY_PATH ${CUDA_PATH}/bin64:${CUDA_PATH}/lib64:${CUDA_PATH}/lib64/stubs:$LD_LIBRARY_PATH

# Miniconda-related environment variables
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

# Install packages
RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git nano && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda 3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Tini: A tiny but valid `init` for containers
ENV TINI_VERSION v0.16.1
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]

WORKDIR /usr/src/app

# First copy only environment.yml, to cache dependencies
COPY environment.yml ./
RUN conda env create
RUN echo "source activate tcn-example" >> ~/.bashrc

# Then copy the rest of the files
COPY . .
