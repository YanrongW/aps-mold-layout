# FROM reg-qd-huangdao.haier.net/library/python:3.9.16
FROM python:3.9.16

ARG source=/
ARG workRoot=/ai-work/
RUN echo "RootPathOnDocker: $workRoot"


RUN mkdir $workRoot
WORKDIR $workRoot

COPY $source .

RUN pip install -r /ai-work/requirements.txt -i https://x-repo.haier.net/repository/pypi-public/simple

EXPOSE 5000