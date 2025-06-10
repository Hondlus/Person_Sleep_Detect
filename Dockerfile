FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
RUN apt-get update
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0
WORKDIR /app
ADD . /app
RUN pip3 install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8086"]
EXPOSE 8086
