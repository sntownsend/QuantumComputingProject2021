FROM chapmanbe/qsharp_base
MAINTAINER chapmanbe <brian.chapman@unimelb.edu.au>
USER root
# dependencies for spell nbextensions (including spell check) and other requirements

WORKDIR /home/jovyan/work

ADD notebooks notebooks

RUN chown -R jovyan demo

USER jovyan
CMD ["start-notebook.sh"]
