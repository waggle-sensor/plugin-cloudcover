FROM waggle/plugin-base:1.0.0-ml-cuda11.0-amd64

COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r /app/requirements.txt

COPY network /app/network
COPY unet /app/unet
COPY app.py unet_module.py deeplab_module.py /app/

ARG SAGE_STORE_URL="HOST"
ARG SAGE_USER_TOKEN="-10"
ARG BUCKET_ID_MODEL="BUCKET_ID_MODEL"

ENV SAGE_STORE_URL=${SAGE_STORE_URL} \
    SAGE_USER_TOKEN=${SAGE_USER_TOKEN} \
    BUCKET_ID_MODEL=${BUCKET_ID_MODEL}

RUN sage-cli.py storage files download ${BUCKET_ID_MODEL} wagglecloud_unet_300.pth --target /app/wagglecloud_unet_300.pth \
 && sage-cli.py storage files download ${BUCKET_ID_MODEL} wagglecloud_deeplab_300.pth --target /app/wagglecloud_deeplab_300.pth

WORKDIR /app
ENTRYPOINT ["python3", "/app/app.py"]
