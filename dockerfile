FROM nvidia/cuda:10.2-base-ubuntu18.04

RUN apt-get update && apt-get install -y curl ca-certificates sudo wget unrar
RUN rm -rf /var/lib/apt/lists/*   # clear package cache

RUN apt update -yq && apt install -yq cmake

RUN mkdir /app
WORKDIR /app

RUN adduser --disabled-password --gecos '' --shell /bin/bash user
RUN chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

ENV HOME=/home/user
RUN chmod 777 /home/user

ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/user/miniconda/bin:$PATH

RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py38_4.8.2-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda install -y python==3.7.7 \
 && conda clean -ya

RUN conda install -c conda-forge opencv scikit-learn
RUN conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch && conda clean -ya

RUN pip install -U ray ray[default] gym gym[atari] lz4 atari_py psutil matplotlib

RUN wget http://www.atarimania.com/roms/Roms.rar

RUN unrar e Roms.rar

RUN python3 -m atari_py.import_roms .

ADD . .

CMD ["python3"]