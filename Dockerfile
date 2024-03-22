# from docker hub use the version of the python to use
FROM python:3.8-slim-buster

# update 
RUN apt update -y 

# MAKING new directory 
WORKDIR /app

# copy everything over
COPY . /app
RUN pip install -r requirements.txt

EXPOSE 80

CMD ["python3", "app.py"]




