FROM python:3.9.5

COPY . .

WORKDIR ./

RUN apt update && apt-get install libsndfile1 libportaudio2 libasound-dev -y

RUN python3 -m pip install -r ./requirements.txt

EXPOSE 8501

CMD streamlit run demo_voice.py