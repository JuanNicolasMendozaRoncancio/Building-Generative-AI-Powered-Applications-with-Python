FROM python:3.10

WORKDIR /MyChatGPT

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "myapp.py"]
