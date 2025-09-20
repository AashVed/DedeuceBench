FROM python:3.11-slim
WORKDIR /app

# Copy and install the dedeuce environment (local source)
COPY dedeuce/ dedeuce/
RUN pip install -e ./dedeuce

# Copy and install the benchmark wrapper
COPY DedeuceBench/ DedeuceBench/
RUN pip install -e ./DedeuceBench[all]

# Provide seeds in a known path inside the image
COPY DedeuceBench/seeds/ seeds/

ENTRYPOINT ["dedeucebench-eval"]
