FROM nvcr.io/nvidia/pytorch:22.08-py3

RUN rm -r /opt/conda/lib/python3.8/site-packages/cv2
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN pip install pbaa