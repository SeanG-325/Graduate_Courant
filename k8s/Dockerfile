# Start with a Linux micro-container to keep the image tiny
FROM mapler/pytorch-cpu:latest

# Document who is responsible for this image
MAINTAINER Shuo Guan "sg6063@nyu.edu"

# Install just the Python runtime (no dev)
RUN pip install flask
RUN mkdir /HW4

# Expose any ports the app is expecting in the environment
ENV PORT 8001
EXPOSE $PORT

# Set up a working folder and install the pre-reqs
WORKDIR /HW4
#ADD requirements.txt /HW4
#RUN pip3 --no-cache-dir install -r requirements.txt
#RUN pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp35-cp35m-linux_x86_64.whl
#RUN pip3 install torchvision

# Add the code as the last Docker layer because it changes the most
ADD ./mnist/main.py  /HW4/main.py

# Run the service
CMD [ "python", "main.py"]

