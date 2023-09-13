FROM nvidia/cuda:11.0-base
RUN apt-get update && apt install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt install -y python3.8
RUN apt install -y python3-pip
RUN apt install -y git
COPY requirements_Team_lucy.txt /usr/local/bin
RUN pip3 install -r /usr/local/bin/requirements_Team_lucy.txt
RUN pip3 install torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
COPY nnUNet/ /usr/local/bin/nnUNet/
COPY trained_models /usr/local/bin/trained_models/
RUN pip3 install -U setuptools
RUN pip3 install -e /usr/local/bin/nnUNet
ENV RESULTS_FOLDER=/usr/local/bin/trained_models/
COPY inference_Team_lucy.py /usr/local/bin

ENTRYPOINT ["python3", "/usr/local/bin/inference_Team_lucy.py"]