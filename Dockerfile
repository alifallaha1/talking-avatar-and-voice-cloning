FROM ubuntu:22.04
FROM python:3.10.13
COPY . /usr/app/
WORKDIR /usr/app/
RUN pip install --upgrade pip
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y ffmpeg
# for cv2 problem
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*


RUN pip install -r requirements.txt
CMD streamlit run app.py
