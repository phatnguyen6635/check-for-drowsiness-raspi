FROM python:3.10-bullseye
ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-tk \
        libatlas-base-dev \
        libprotobuf-dev \
        protobuf-compiler \
        libgl1 \
        libglib2.0-0 \
        libjpeg-dev \
        libpng-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libqt5gui5 \
        libqt5core5a \
        libqt5widgets5 \
        libx11-6 \
        libxext6 \
        libxrender1 \
        libxft2 \
        ffmpeg \
        x11-apps \
        cron \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app


COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt


COPY . /app

COPY cronjob.txt /etc/cron.d/log_handler_cron
RUN chmod 0644 /etc/cron.d/log_handler_cron && crontab /etc/cron.d/log_handler_cron

ENV DISPLAY=:0
ENV QT_X11_NO_MITSHM=1

CMD ["bash", "-c", "cron -f & python -m tools.mainui"]

