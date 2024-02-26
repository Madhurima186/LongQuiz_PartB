FROM python:3.11.8-bookworm
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD python app.py