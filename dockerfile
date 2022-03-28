FROM nvidia/cuda:10.2-base-ubuntu18.04

RUN apt-get update && apt-get install -y curl ca-certificates sudo wget unrar unzip
RUN rm -rf /var/lib/apt/lists/*   # clear package cache

RUN apt update -yq && apt install -yq cmake

RUN apt-get install ffmpeg libsm6 libxext6  -yq

RUN mkdir /app
WORKDIR /app

ADD . .

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
 && conda install -y python==3.8.12 \
 && conda clean -ya

RUN pip install --upgrade pip
RUN pip3 install -r requirements.txt -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html

RUN wget http://www.atarimania.com/roms/Roms.rar

RUN unrar e Roms.rar

RUN unzip ROMS.zip

RUN	ale-import-roms ROMS

CMD ["python3"]