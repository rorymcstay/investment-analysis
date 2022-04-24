#
# To run the container:
#
#    docker run -v /path/to/your/notebooks:/projects -v ~/.zipline:/root/.zipline -p 8888:8888/tcp --name zipline -it quantopian/zipline
#
# To access Jupyter when running docker locally (you may need to add NAT rules):
#
#    https://127.0.0.1
#
# default password is jupyter.  to provide another, see:
#    http://jupyter-notebook.readthedocs.org/en/latest/public_server.html#preparing-a-hashed-password
#
# once generated, you can pass the new value via `docker run --env` the first time
# you start the container.
#
# You can also run an algo using the docker exec command.  For example:
#
#    docker exec -it zipline zipline run -f /projects/my_algo.py --start 2015-1-1 --end 2016-1-1 -o /projects/result.pickle
#
FROM python:3.8

#
# set up environment
#


ENV PROJECT_DIR=/projects \
    NOTEBOOK_PORT=8888 \
    SSL_CERT_PEM=/root/.jupyter/jupyter.pem \
    SSL_CERT_KEY=/root/.jupyter/jupyter.key \
    PW_HASH="u'sha1:31cb67870a35:1a2321318481f00b0efdf3d1f71af523d3ffc505'" \
    CONFIG_PATH=/root/.jupyter/jupyter_notebook_config.py

#
# install TA-Lib and other requirements
#
RUN mkdir ${PROJECT_DIR} \
    && apt-get -y update \
    && apt-get -y install \
        libatlas-base-dev \
        python-dev gfortran \
        pkg-config libfreetype6-dev \
        hdf5-tools \
        tini \
    && curl -L https://downloads.sourceforge.net/project/ta-lib/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz \
    && wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib/ \
    && ./configure \
    && make install \
    && cd ../ && rm -r ta-lib/ \
    && pip install matplotlib \
            jupyter

#
EXPOSE ${NOTEBOOK_PORT}

#
#
WORKDIR /investment-analysis
ADD . ./
RUN pip install -e /investment-analysis

#
# start the jupyter server
#
WORKDIR ${PROJECT_DIR}
RUN chmod +x /investment-analysis/docker_cmd.sh

ENTRYPOINT ["tini", "--"]
CMD /investment-analysis/docker_cmd.sh
