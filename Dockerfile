# Use the official Python image as the base image
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy the application code into the container
COPY . /app

# Install any dependencies
RUN pip install -r requirements.txt

# Specify the command to run your Python application
CMD ["python", "app.py"]
