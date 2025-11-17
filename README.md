# Deep Learning Notebooks

Este proyecto contiene una colección de notebooks para experimentar y aprender sobre Deep Learning. Los notebooks están organizados en la carpeta `Notebooks/` y cubren temas variados relacionados con redes neuronales, aprendizaje profundo y aplicaciones prácticas.

## Entorno de ejecución

Para facilitar la configuración y asegurar la reproducibilidad, el proyecto utiliza Docker y Docker Compose. El contenedor principal incluye TensorFlow y otras dependencias necesarias para ejecutar los notebooks.

### Requisitos previos
- Docker
- Docker Compose

### Cómo levantar el entorno

1. Clona este repositorio en tu máquina local.
2. Abre una terminal en la raíz del proyecto.
3. Ejecuta el siguiente comando para levantar el entorno:

```powershell
docker compose up -d
```

Esto iniciará el contenedor con TensorFlow y todas las dependencias necesarias.

### Acceso a los notebooks

Una vez levantado el entorno, accede a los notebooks a través de Jupyter (o el método especificado en el contenedor, revisa los logs para la URL de acceso). Los notebooks se encuentran en la carpeta `Notebooks/`.

## Personalización

Si necesitas instalar paquetes adicionales, puedes modificar el archivo `requirements.txt` y reconstruir el contenedor:

```powershell
docker compose build
```

## Estructura del proyecto
- `Notebooks/`: Notebooks de Deep Learning
- `Dockerfile`: Imagen personalizada con TensorFlow
- `docker-compose.yml`: Orquestación de servicios

---

¡Explora, experimenta y aprende Deep Learning de forma sencilla y reproducible!
