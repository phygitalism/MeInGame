FROM nvidia/cudagl:10.0-devel-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt-get remove --purge python3.6 -y && \
    apt-get update -y && apt-get install wget python3.8 python3-pip git cmake protobuf-compiler -y && python3 -m pip install --upgrade pip

WORKDIR /home
RUN pip3 install --no-cache-dir -f https://download.pytorch.org/whl/torch_stable.html \
                 torch==1.4.0+cu100 \
                 torchvision==0.5.0+cu100 \
                 scikit-build \
                 opencv-python \
                 scikit-image \
                 fvcore \
                 h5py \
                 scipy \
                 dlib \
                 face-alignment==1.2.0 \
                 scikit-learn \
                 tensorflow-gpu==1.14.0 \
                 gast==0.2.2 && \
    pip3 install --no-cache-dir "git+https://github.com/Agent-INF/pytorch3d.git@3dface"

RUN git clone https://github.com/phygitalism/MeInGame.git && \
    cd MeInGame/data && \
    mkdir models && \
    mkdir checkpoints && \
    cd checkpoints && mkdir celeba_hq_demo

WORKDIR /home/MeInGame
RUN cd ./data/models && \
    wget https://raw.githubusercontent.com/microsoft/Deep3DFaceReconstruction/master/BFM/similarity_Lm3D_all.mat
COPY ./BFM/bfm2009_face.mat ./data/models
COPY ./checkpoints/512_1024_gen_0030.pth ./checkpoints/celeba_hq_demo/
COPY ./checkpoints/torch_FaceSegment_300.pkl ./data/models
COPY ./checkpoints/FaceReconModel.pb ./data/models
