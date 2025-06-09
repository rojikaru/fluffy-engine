FROM python:3.13-slim AS builder

# install build tools & libraries needed to compile any C-extensions
RUN apt update && \
    apt install -y --no-install-recommends \
      build-essential \
      git \
      libssl-dev \
      libffi-dev \
      curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy only your lock file (or requirements.txt)
COPY requirements.txt .

# install *all* dependencies here
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt


FROM python:3.13-slim AS runtime

WORKDIR /app

# Copy only the installed packages from builder
COPY --from=builder /usr/local/lib/python3.13/site-packages \
                  /usr/local/lib/python3.13/site-packages

# Copy your app code
COPY . .

# Expose and run
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
