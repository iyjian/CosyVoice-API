#!/bin/bash

# docker build -t iyjian/cosyvoice .
# docker stop cosyvoice && docker rm cosyvoice
# docker logs -f --tail 100 cosyvoice

docker run -d \
--restart always \
--name cosyvoice \
--env-file .env \
-p 5000:5000 \
-v ${PWD}/pretrained_models:/app/CosyVoice/pretrained_models \
--gpus all \
iyjian/cosyvoice
