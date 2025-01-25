FROM josafatburmeister/pointtorch:latest

RUN apt-get update && apt-get install -y g++

RUN python -m pip install pointtree[dev,docs]
