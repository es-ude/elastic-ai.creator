FROM  nvidia/cuda:11.5.2-devel-ubuntu20.04
ENV LANG C.UTF-8
RUN apt -y update
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata software-properties-common git
ENV PATH="/root/.local/bin/:${PATH}"
# use python 3.10 and setup pip and venv module
RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update && apt -y install python3.10 python3.10-distutils python3.10-venv
ENV VIRTUAL_ENV=/opt/env
RUN python3.10 -m venv $VIRTUAL_ENV 
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python -m ensurepip --upgrade && python -m pip install wheel
# install our creator
RUN python -m pip install "elasticai.creator"



