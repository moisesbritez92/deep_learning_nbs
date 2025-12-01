FROM tensorflow/tensorflow:2.15.0-gpu-jupyter

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Instalar paquetes Python
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    seaborn \
    jupyterlab \
    opencv-python-headless \
    pillow \
    tensorflow-addons \
    vit-keras==0.1.2
