FROM lostpetinitiative/tensorflow-2-no-avx-cpu:2.3.0

# RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y python3.8 python3-pip && \
#     rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests \
    -y libglib2.0-0 libgl1-mesa-dev libsm6 libxrender1 libxtst6 libxi6 && \
    rm -rf /var/lib/apt/lists/*

COPY service_requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip3 install -r requirements.txt

COPY data/exp_3_4_weights /app
COPY code/efficientSiameseNet.py /app/efficientSiameseNet.py
COPY code/kafkaJobQueue.py /app/kafkaJobQueue.py
COPY code/imageSerialization.py /app/imageSerialization.py
COPY code/npSerialization.py /app/npSerialization.py
COPY code/service.py /app/service.py

ENV KAFKA_URL=kafka-cluster.kashtanka:9092
ENV INPUT_QUEUE=kashtanka_detected_pets
ENV OUTPUT_QUEUE=kashtanka_processed_cards
ENV PET_TYPE=unknown

CMD python3.6 -u service.py