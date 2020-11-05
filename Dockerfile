FROM ubuntu

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y python3.8 python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests \
    -y libglib2.0-0 libgl1-mesa-dev libsm6 libxrender1 libxtst6 libxi6 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY data/exp_3_4_weights /app
COPY code/efficientSiameseNet.py /app/efficientSiameseNet.py
COPY code/kafkaJobQueue.py /app/kafkaJobQueue.py
COPY code/service.py /app/service.py
COPY service_requirements.txt /app/requirements.txt

RUN pip3 install -r requirements.txt

ENV KAFKA_URL=kafka-cluster.kashtanka:9092
ENV INPUT_QUEUE=DetectedPets
ENV OUTPUT_QUEUE=ProcessedCards
ENV PET_TYPE=unknown

CMD python3.8 -u service.py