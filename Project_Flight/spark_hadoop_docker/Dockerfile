FROM datamechanics/spark:3.0.0-hadoop-3.2.0-java-11-scala-2.12-python-3.8-latest

COPY requirements.txt .
RUN python3 -m pip install --upgrade pip --user
RUN python3 -m pip install --user -r requirements.txt
