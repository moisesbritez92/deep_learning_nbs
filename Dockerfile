FROM tensorflow/tensorflow:2.15.0-gpu-jupyter

# Instalar paquetes Python
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    seaborn \
    jupyterlab \
    opencv-python \
    pillow
