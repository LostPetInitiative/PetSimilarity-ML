FROM lostpetinitiative/tensorflow-2-no-avx-cpu:2.3.0

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests \
    -y wget libglib2.0-0 libgl1-mesa-dev libsm6 libxrender1 libxtst6 libxi6 && \
    rm -rf /var/lib/apt/lists/*

RUN wget -v https://grechka.family/dmitry/sandbox/dist/kafka_job_scheduler-0.1.1-py3-none-any.whl && \
    pip install kafka_job_scheduler-0.1.1-py3-none-any.whl && \
    rm kafka_job_scheduler-0.1.1-py3-none-any.whl

COPY service_requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip3 install -r requirements.txt

COPY data/exp_3_4_weights /app
COPY code/efficientSiameseNet.py /app/efficientSiameseNet.py
COPY code/service.py /app/service.py

ENV KAFKA_URL=kafka-cluster.kashtanka:9092
ENV INPUT_QUEUE=kashtanka_detected_pets
ENV OUTPUT_QUEUE=kashtanka_cards_with_features
ENV ANIMAL_TYPE=unknown

CMD python3.6 -u service.py