::: {.cell .markdown id="KK_5ZUKYQQ8f"}
## Libraries
:::

::: {.cell .code id="ZsusBqpCQQ8i"}
``` python
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt
import shutil
import random

random.seed(7)
```
:::

:::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" executionInfo="{\"elapsed\":21307,\"status\":\"ok\",\"timestamp\":1701776275272,\"user\":{\"displayName\":\"Diego Borro\",\"userId\":\"13584858323109103783\"},\"user_tz\":-60}" id="gjd8aBZBvljo" outputId="16fc01cd-3cdb-4a01-e335-8c9ffd174162"}
``` python
from google.colab import drive
drive.mount('/content/drive')
```

::: {.output .stream .stdout}
    Mounted at /content/drive
:::
::::

::: {.cell .code id="9koP1G1fvpmV"}
``` python
PATH = '/content/sample_data/Data/'
BASE_PATH = '/content/sample_data/'
DRIVE_PATH = '/content/drive/MyDrive/Colab Notebooks/Computer Vision course/DL/03b - MultipleDetection-PCB/'
```
:::

::: {.cell .code id="rBrLYXSzvrfY"}
``` python
!unzip -q '/content/drive/MyDrive/Colab Notebooks/Computer Vision course/DL/03b - MultipleDetection-PCB/Data/Original Images.zip' -d '/content/sample_data/Data'
```
:::

::: {.cell .markdown id="6StQdi9qQQ8j"}
YOLO ANNOTATIONS FORMAT: class, x_center, y_center, width, height

DATA FORMAT IN THE FILES PROVIDED: x_min, y_min, x_max, y_max, class
:::

:::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" executionInfo="{\"elapsed\":486,\"status\":\"ok\",\"timestamp\":1701708489332,\"user\":{\"displayName\":\"Diego Borro\",\"userId\":\"13584858323109103783\"},\"user_tz\":-60}" id="fcIHOVn7QQ8j" outputId="7e06a702-ccd7-417b-98ad-07569f0ea5a9"}
``` python
# OBTAIN PATH TO ALL OBJECTS

path_origin = PATH+'Original Images'

items_paths = []

groups_folders = os.listdir(path_origin)

for folder in groups_folders:
    subfolders = os.listdir(os.path.join(path_origin, folder))

    for subfolder in subfolders:
        items = os.listdir(os.path.join(os.path.join(path_origin, folder), subfolder))

        for item in items:
            items_paths.append(os.path.join(os.path.join(os.path.join(path_origin, folder), subfolder), item))

print(len(items_paths))
```

::: {.output .stream .stdout}
    4501
:::
::::

::: {.cell .code id="fHOP1kb8QQ8k"}
``` python
# SPLIT BETWEEN ANNOTATIONS AND IMAGES

annotations_path = []
images_path = []

for item_path in items_paths:
    if '_not' in item_path:
        annotations_path.append(item_path)

    if '_test' in item_path:
        images_path.append(item_path)
```
:::

:::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" executionInfo="{\"elapsed\":3,\"status\":\"ok\",\"timestamp\":1701708495225,\"user\":{\"displayName\":\"Diego Borro\",\"userId\":\"13584858323109103783\"},\"user_tz\":-60}" id="QdcqRM8oQQ8k" outputId="909f6116-3f3d-47f2-d188-6110545fdcf5"}
``` python
# CHECK LENGTH

print(len(images_path))
print(len(annotations_path))
```

::: {.output .stream .stdout}
    1500
    1500
:::
::::

:::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":35}" executionInfo="{\"elapsed\":5,\"status\":\"ok\",\"timestamp\":1701708498104,\"user\":{\"displayName\":\"Diego Borro\",\"userId\":\"13584858323109103783\"},\"user_tz\":-60}" id="Y3l2yCZlQQ8l" outputId="1f95b525-c99f-423f-d655-699365b7ed17"}
``` python
images_path[50]
```

::: {.output .execute_result execution_count="8"}
``` json
{"type":"string"}
```
:::
::::

:::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":35}" executionInfo="{\"elapsed\":3,\"status\":\"ok\",\"timestamp\":1701708499274,\"user\":{\"displayName\":\"Diego Borro\",\"userId\":\"13584858323109103783\"},\"user_tz\":-60}" id="OO4OPwi6QQ8l" outputId="7c3c662d-6612-465c-9456-e7b618d05c17"}
``` python
# SPLIT THE PATH TO OBTAIN ONLY THE IMAGE NAME

words = images_path[50].split('/')
words[-1]
```

::: {.output .execute_result execution_count="9"}
``` json
{"type":"string"}
```
:::
::::

:::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":35}" executionInfo="{\"elapsed\":2,\"status\":\"ok\",\"timestamp\":1701708500816,\"user\":{\"displayName\":\"Diego Borro\",\"userId\":\"13584858323109103783\"},\"user_tz\":-60}" id="Mo18zxSAQQ8m" outputId="0566d501-4d45-4414-e27f-7e3f5a607680"}
``` python
# OBTAIN ONLY THE CODE FOR THE IMAGE

image_name = words[-1].replace('_test.jpg', '')
image_name
```

::: {.output .execute_result execution_count="10"}
``` json
{"type":"string"}
```
:::
::::

:::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":35}" executionInfo="{\"elapsed\":3,\"status\":\"ok\",\"timestamp\":1701708501392,\"user\":{\"displayName\":\"Diego Borro\",\"userId\":\"13584858323109103783\"},\"user_tz\":-60}" id="OVnCxAnTQQ8m" outputId="0189b648-647f-4ea5-b2e5-005dc235f85c"}
``` python
# SEARCH FOR THE ANNOTATION OF THE IMAGE

for annotation in annotations_path:

    if image_name in annotation:
        break
annotation
```

::: {.output .execute_result execution_count="11"}
``` json
{"type":"string"}
```
:::
::::

:::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" executionInfo="{\"elapsed\":545,\"status\":\"ok\",\"timestamp\":1701708506826,\"user\":{\"displayName\":\"Diego Borro\",\"userId\":\"13584858323109103783\"},\"user_tz\":-60}" id="8lGzz1yVQQ8n" outputId="4ac18997-8f93-4ac1-ce6a-597ad8621e08"}
``` python
# READ TH FILE

with open(annotation, 'r') as content:
    details = content.readlines()

details
```

::: {.output .execute_result execution_count="12"}
    ['166 324 214 354 1\n',
     '449 594 503 629 2\n',
     '426 225 464 252 5\n',
     '479 132 511 162 6\n',
     '300 180 330 215 3\n',
     '150 191 197 229 2\n']
:::
::::

::: {.cell .code id="XMjdUbAlQQ8n"}
``` python
# CLASS DICTIONARY FROM GITHUB (https://github.com/tangsanli5201/DeepPCB)

dict_classes = {1: 'open', 2: 'short', 3: 'mousebite', 4: 'spur', 5: 'copper', 6: 'pin-hole'}
```
:::

::::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":452}" executionInfo="{\"elapsed\":676,\"status\":\"ok\",\"timestamp\":1701708523397,\"user\":{\"displayName\":\"Diego Borro\",\"userId\":\"13584858323109103783\"},\"user_tz\":-60}" id="c7ediRjHQQ8n" outputId="68d0362c-4934-4de3-df8b-51f036946f8a"}
``` python
# DRAW ANNOTATIONS IN THE IMAGE

image = images_path[50]
label = details

image = cv2.imread(image)

for detail in details:

    splitted = detail.split()
    x1 = int(splitted[0])
    x2 = int(splitted[2])
    y1 = int(splitted[1])
    y2 = int(splitted[3])

    rectangle_image = cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)

plt.imshow(rectangle_image)
```

::: {.output .execute_result execution_count="14"}
    <matplotlib.image.AxesImage at 0x7dd8659b21a0>
:::

::: {.output .display_data}
![](fee37c4d13705242fece97684412891461d889f1.png)
:::
:::::

::: {.cell .code id="cil6HCbEQQ8n"}
``` python
# AS WE CHECK THE FORMAT OF THE ANNOTATION, LET'S TRANSLATE IT TO YOLO FORMAT

def to_yolo(image_path, annotations_path, folder_path):

    yolo_out = []

    image = cv2.imread(image_path)
    h, w, c = image.shape

    cadena = ''.join(image_path)

    words = cadena.split('/')
    image_name = words[-1].replace('_test.jpg', '')

    for annotation in annotations_path:

        if image_name in annotation:
            break

    with open(annotation, 'r') as content:
        details = content.readlines()

    for detail in details:

        splitted = detail.split()
        x1 = int(splitted[0])
        x2 = int(splitted[2])
        y1 = int(splitted[1])
        y2 = int(splitted[3])
        label = int(splitted[4])-1 # CLASSES STARTS WITH 0 IN YOLO

        x_center = (x2 + x1) / (2*w)
        y_center = (y2 + y1) / (2*h)
        width = (x2-x1) / w
        height = (y2-y1) / h

        yolo_out.append([label, x_center, y_center, width, height])

    shutil.copy(image_path, os.path.join(folder_path, 'images/' + image_name + '.jpg'))

    with open(folder_path + '/labels/' + image_name + '.txt', 'w') as archivo:
        for line in yolo_out:
            for w in line:
                archivo.write(str(w))
                archivo.write(' ')
            archivo.write('\n')

    return yolo_out
```
:::

::: {.cell .code id="uQ7RGDA5QQ8n"}
``` python
total_images = len(images_path)

train_number = int(total_images * 0.7)

train_images = random.sample(images_path, train_number)
val_images = [element for element in images_path if element not in train_images]
```
:::

::: {.cell .code id="CXCkHSwjQQ8o"}
``` python
os.makedirs(BASE_PATH+'train/images', exist_ok=True)
os.makedirs(BASE_PATH+'train/labels', exist_ok=True)
os.makedirs(BASE_PATH+'val/images', exist_ok=True)
os.makedirs(BASE_PATH+'val/labels', exist_ok=True)

for image in train_images:
    to_yolo(image, annotations_path, BASE_PATH+'train')

for image in val_images:
    to_yolo(image, annotations_path, BASE_PATH+'val')
```
:::

::: {.cell .markdown id="xAFFiD2WQQ8o"}
# YOLO TRAIN
:::

:::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" executionInfo="{\"elapsed\":4021,\"status\":\"ok\",\"timestamp\":1701708584469,\"user\":{\"displayName\":\"Diego Borro\",\"userId\":\"13584858323109103783\"},\"user_tz\":-60}" id="GmzXeXxiQQ8o" outputId="573e3cd6-2984-4479-f4c0-cf7f6010aba9"}
``` python
import torch
torch.cuda.is_available()
```

::: {.output .execute_result execution_count="19"}
    True
:::
::::

:::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":35}" executionInfo="{\"elapsed\":2079,\"status\":\"ok\",\"timestamp\":1701709291788,\"user\":{\"displayName\":\"Diego Borro\",\"userId\":\"13584858323109103783\"},\"user_tz\":-60}" id="EJrkNtRpvz8K" outputId="d4bcee74-2fa1-49ea-809f-b9d6ef6a8f11"}
``` python
# copy configuration file
shutil.copy(DRIVE_PATH+"custom.yaml", PATH)
```

::: {.output .execute_result execution_count="36"}
``` json
{"type":"string"}
```
:::
::::

:::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" executionInfo="{\"elapsed\":8308,\"status\":\"ok\",\"timestamp\":1701708632096,\"user\":{\"displayName\":\"Diego Borro\",\"userId\":\"13584858323109103783\"},\"user_tz\":-60}" id="I9AXnZZCQQ8o" outputId="d8e3cf65-8bbf-43ae-ff0d-e8b710623d42"}
``` python
!pip install ultralytics

from ultralytics import YOLO
```

::: {.output .stream .stdout}
    Collecting ultralytics
      Downloading ultralytics-8.0.222-py3-none-any.whl (653 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0.0/654.0 kB ? eta -:--:--━━━━━━╸━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 112.6/654.0 kB 3.2 MB/s eta 0:00:01━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 654.0/654.0 kB 12.1 MB/s eta 0:00:00
    ent already satisfied: matplotlib>=3.3.0 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (3.7.1)
    Requirement already satisfied: numpy>=1.22.2 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (1.23.5)
    Requirement already satisfied: opencv-python>=4.6.0 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (4.8.0.76)
    Requirement already satisfied: pillow>=7.1.2 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (9.4.0)
    Requirement already satisfied: pyyaml>=5.3.1 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (6.0.1)
    Requirement already satisfied: requests>=2.23.0 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (2.31.0)
    Requirement already satisfied: scipy>=1.4.1 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (1.11.4)
    Requirement already satisfied: torch>=1.8.0 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (2.1.0+cu118)
    Requirement already satisfied: torchvision>=0.9.0 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (0.16.0+cu118)
    Requirement already satisfied: tqdm>=4.64.0 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (4.66.1)
    Requirement already satisfied: pandas>=1.1.4 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (1.5.3)
    Requirement already satisfied: seaborn>=0.11.0 in /usr/local/lib/python3.10/dist-packages (from ultralytics) (0.12.2)
    Requirement already satisfied: psutil in /usr/local/lib/python3.10/dist-packages (from ultralytics) (5.9.5)
    Requirement already satisfied: py-cpuinfo in /usr/local/lib/python3.10/dist-packages (from ultralytics) (9.0.0)
    Collecting thop>=0.1.1 (from ultralytics)
      Downloading thop-0.1.1.post2209072238-py3-none-any.whl (15 kB)
    Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.3.0->ultralytics) (1.2.0)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.3.0->ultralytics) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.3.0->ultralytics) (4.45.1)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.3.0->ultralytics) (1.4.5)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.3.0->ultralytics) (23.2)
    Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.3.0->ultralytics) (3.1.1)
    Requirement already satisfied: python-dateutil>=2.7 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.3.0->ultralytics) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas>=1.1.4->ultralytics) (2023.3.post1)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests>=2.23.0->ultralytics) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests>=2.23.0->ultralytics) (3.6)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests>=2.23.0->ultralytics) (2.0.7)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests>=2.23.0->ultralytics) (2023.11.17)
    Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from torch>=1.8.0->ultralytics) (3.13.1)
    Requirement already satisfied: typing-extensions in /usr/local/lib/python3.10/dist-packages (from torch>=1.8.0->ultralytics) (4.5.0)
    Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from torch>=1.8.0->ultralytics) (1.12)
    Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch>=1.8.0->ultralytics) (3.2.1)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch>=1.8.0->ultralytics) (3.1.2)
    Requirement already satisfied: fsspec in /usr/local/lib/python3.10/dist-packages (from torch>=1.8.0->ultralytics) (2023.6.0)
    Requirement already satisfied: triton==2.1.0 in /usr/local/lib/python3.10/dist-packages (from torch>=1.8.0->ultralytics) (2.1.0)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.7->matplotlib>=3.3.0->ultralytics) (1.16.0)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch>=1.8.0->ultralytics) (2.1.3)
    Requirement already satisfied: mpmath>=0.19 in /usr/local/lib/python3.10/dist-packages (from sympy->torch>=1.8.0->ultralytics) (1.3.0)
    Installing collected packages: thop, ultralytics
    Successfully installed thop-0.1.1.post2209072238 ultralytics-8.0.222
:::
::::

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" executionInfo="{\"elapsed\":4220817,\"status\":\"ok\",\"timestamp\":1701713515742,\"user\":{\"displayName\":\"Diego Borro\",\"userId\":\"13584858323109103783\"},\"user_tz\":-60}" id="4UrkmUCsQQ8o" outputId="1b23640b-e407-4892-939c-05ec020cc136"}
``` python
model_p = YOLO("yolov8n.pt")
model_p.train(data=PATH+"custom.yaml", epochs=100, imgsz=640, batch=4, workers=1)
```

::: {.output .stream .stdout}
    Ultralytics YOLOv8.0.222 🚀 Python-3.10.12 torch-2.1.0+cu118 CUDA:0 (Tesla T4, 15102MiB)
    engine/trainer: task=detect, mode=train, model=yolov8n.pt, data=/content/sample_data/Data/custom.yaml, epochs=100, patience=50, batch=4, imgsz=640, save=True, save_period=-1, cache=False, device=None, workers=1, project=None, name=train8, exist_ok=False, pretrained=True, optimizer=auto, verbose=True, seed=0, deterministic=True, single_cls=False, rect=False, cos_lr=False, close_mosaic=10, resume=False, amp=True, fraction=1.0, profile=False, freeze=None, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, vid_stride=1, stream_buffer=False, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, show=False, save_frames=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, show_boxes=True, line_width=None, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=False, opset=None, workspace=4, nms=False, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, label_smoothing=0.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0, cfg=None, tracker=botsort.yaml, save_dir=runs/detect/train8
    Downloading https://ultralytics.com/assets/Arial.ttf to '/root/.config/Ultralytics/Arial.ttf'...
:::

::: {.output .stream .stderr}
    100%|██████████| 755k/755k [00:00<00:00, 91.4MB/s]
:::

::: {.output .stream .stdout}
    Overriding model.yaml nc=80 with nc=6

                       from  n    params  module                                       arguments                     
      0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]                 
      1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]                
      2                  -1  1      7360  ultralytics.nn.modules.block.C2f             [32, 32, 1, True]             
      3                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]                
      4                  -1  2     49664  ultralytics.nn.modules.block.C2f             [64, 64, 2, True]             
      5                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
      6                  -1  2    197632  ultralytics.nn.modules.block.C2f             [128, 128, 2, True]           
      7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]              
      8                  -1  1    460288  ultralytics.nn.modules.block.C2f             [256, 256, 1, True]           
      9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]                 
     10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
     11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
     12                  -1  1    148224  ultralytics.nn.modules.block.C2f             [384, 128, 1]                 
     13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
     14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
     15                  -1  1     37248  ultralytics.nn.modules.block.C2f             [192, 64, 1]                  
     16                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                
     17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
     18                  -1  1    123648  ultralytics.nn.modules.block.C2f             [192, 128, 1]                 
     19                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
     20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
     21                  -1  1    493056  ultralytics.nn.modules.block.C2f             [384, 256, 1]                 
     22        [15, 18, 21]  1    752482  ultralytics.nn.modules.head.Detect           [6, [64, 128, 256]]           
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}
    Model summary: 225 layers, 3012018 parameters, 3012002 gradients, 8.2 GFLOPs

    Transferred 319/355 items from pretrained weights
    TensorBoard: Start with 'tensorboard --logdir runs/detect/train8', view at http://localhost:6006/
    Freezing layer 'model.22.dfl.conv.weight'
    AMP: running Automatic Mixed Precision (AMP) checks with YOLOv8n...
    WARNING ⚠️ NMS time limit 0.550s exceeded
    AMP: checks passed ✅
:::

::: {.output .stream .stderr}
    train: Scanning /content/sample_data/train/labels... 1050 images, 0 backgrounds, 0 corrupt: 100%|██████████| 1050/1050 [00:00<00:00, 1855.84it/s]
:::

::: {.output .stream .stdout}
    train: New cache created: /content/sample_data/train/labels.cache
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}
    albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01), CLAHE(p=0.01, clip_limit=(1, 4.0), tile_grid_size=(8, 8))
:::

::: {.output .stream .stderr}
    val: Scanning /content/sample_data/val/labels... 450 images, 0 backgrounds, 0 corrupt: 100%|██████████| 450/450 [00:00<00:00, 803.22it/s]
:::

::: {.output .stream .stdout}
    val: New cache created: /content/sample_data/val/labels.cache
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}
    Plotting labels to runs/detect/train8/labels.jpg... 
    optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
    optimizer: AdamW(lr=0.001, momentum=0.9) with parameter groups 57 weight(decay=0.0), 64 weight(decay=0.0005), 63 bias(decay=0.0)
    Image sizes 640 train, 640 val
    Using 1 dataloader workers
    Logging results to runs/detect/train8
    Starting training for 100 epochs...

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
          1/100     0.726G      2.159      3.258      1.362         23        640: 100%|██████████| 263/263 [00:40<00:00,  6.53it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:08<00:00,  6.54it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985       0.45      0.563       0.48       0.24
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
          2/100     0.757G      1.536       1.96      1.093          9        640: 100%|██████████| 263/263 [00:39<00:00,  6.73it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:05<00:00, 10.74it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.688      0.677      0.714      0.411

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
          3/100     0.757G      1.489       1.67      1.064         26        640: 100%|██████████| 263/263 [00:35<00:00,  7.32it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:07<00:00,  7.89it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.745      0.689      0.749      0.362

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
          4/100     0.747G      1.418      1.472      1.049         19        640: 100%|██████████| 263/263 [00:37<00:00,  7.04it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:04<00:00, 11.47it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.751      0.736      0.809      0.414

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
          5/100     0.759G       1.38      1.354      1.032         12        640: 100%|██████████| 263/263 [00:36<00:00,  7.30it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:06<00:00,  9.27it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.699      0.655      0.649      0.205

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
          6/100     0.742G      1.357       1.25      1.014         21        640: 100%|██████████| 263/263 [00:35<00:00,  7.38it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:05<00:00, 10.08it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.887      0.822      0.906      0.588
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
          7/100     0.744G      1.331      1.166      1.013         16        640: 100%|██████████| 263/263 [00:35<00:00,  7.39it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:04<00:00, 11.72it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.914      0.852      0.926      0.604
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
          8/100     0.755G      1.272      1.109      0.998         10        640: 100%|██████████| 263/263 [00:35<00:00,  7.39it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:07<00:00,  7.52it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.914      0.838       0.91      0.471
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
          9/100     0.757G      1.264      1.073     0.9957         24        640: 100%|██████████| 263/263 [00:35<00:00,  7.31it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:04<00:00, 11.49it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.919       0.85      0.937      0.637
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         10/100     0.755G      1.233      1.012     0.9816         11        640: 100%|██████████| 263/263 [00:36<00:00,  7.19it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:06<00:00,  8.57it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.943      0.865      0.948      0.639
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         11/100      0.74G      1.231     0.9725      0.985          7        640: 100%|██████████| 263/263 [00:37<00:00,  6.95it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:04<00:00, 11.69it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.932      0.876      0.942      0.597
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         12/100      0.74G      1.185     0.9348     0.9696         12        640: 100%|██████████| 263/263 [00:35<00:00,  7.36it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:05<00:00, 10.79it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.916      0.873      0.948      0.641
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         13/100     0.742G      1.192     0.9163     0.9755         16        640: 100%|██████████| 263/263 [00:36<00:00,  7.18it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:06<00:00,  8.83it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.935      0.887      0.952      0.633
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         14/100      0.74G      1.212     0.9125      0.981         11        640: 100%|██████████| 263/263 [00:36<00:00,  7.24it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:04<00:00, 11.53it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.931      0.894      0.953      0.643
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         15/100      0.74G       1.19     0.8701       0.97         25        640: 100%|██████████| 263/263 [00:36<00:00,  7.25it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:07<00:00,  7.58it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.935      0.897      0.952      0.618
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         16/100     0.753G      1.171      0.853     0.9697         31        640: 100%|██████████| 263/263 [00:35<00:00,  7.32it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:04<00:00, 11.72it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.932        0.9      0.964      0.706
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         17/100     0.742G      1.189     0.8447     0.9729         11        640: 100%|██████████| 263/263 [00:35<00:00,  7.35it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:07<00:00,  7.81it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.923      0.895       0.94      0.435
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         18/100     0.755G      1.177     0.8446      0.971         10        640: 100%|██████████| 263/263 [00:36<00:00,  7.25it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:04<00:00, 11.72it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.933        0.9      0.947      0.552
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         19/100     0.742G      1.123     0.8088     0.9613         13        640: 100%|██████████| 263/263 [00:35<00:00,  7.46it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:05<00:00, 10.68it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.948      0.905      0.963      0.694
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         20/100     0.755G      1.164     0.8095     0.9722         15        640: 100%|██████████| 263/263 [00:35<00:00,  7.34it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:06<00:00,  8.45it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.952      0.906      0.965      0.658
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         21/100     0.761G      1.135     0.7886     0.9588         14        640: 100%|██████████| 263/263 [00:35<00:00,  7.41it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:04<00:00, 11.83it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.946      0.917      0.966      0.664
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         22/100     0.742G        1.1      0.758     0.9478          8        640: 100%|██████████| 263/263 [00:35<00:00,  7.48it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:06<00:00,  8.47it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.954      0.911      0.969      0.689
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         23/100      0.74G      1.125      0.768     0.9592          8        640: 100%|██████████| 263/263 [00:35<00:00,  7.45it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:05<00:00, 10.39it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.937       0.91      0.956      0.544
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         24/100     0.742G       1.09     0.7537     0.9453         12        640: 100%|██████████| 263/263 [00:35<00:00,  7.35it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:04<00:00, 11.96it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.955      0.925      0.971      0.657
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         25/100      0.74G      1.127      0.746     0.9561         14        640: 100%|██████████| 263/263 [00:36<00:00,  7.30it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:07<00:00,  7.69it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985       0.95      0.919      0.963      0.559
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         26/100     0.753G      1.116     0.7267     0.9523         10        640: 100%|██████████| 263/263 [00:36<00:00,  7.23it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:04<00:00, 11.72it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.956      0.915      0.971      0.701
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         27/100     0.753G      1.093     0.7073     0.9489         15        640: 100%|██████████| 263/263 [00:36<00:00,  7.27it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:06<00:00,  8.51it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.946      0.916      0.964      0.558
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         28/100      0.74G      1.079     0.6998     0.9456         14        640: 100%|██████████| 263/263 [00:36<00:00,  7.21it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:05<00:00, 10.65it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.955      0.912      0.963      0.519
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         29/100      0.74G      1.158     0.7252     0.9614         14        640: 100%|██████████| 263/263 [00:36<00:00,  7.20it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:05<00:00, 11.09it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.955      0.923      0.975      0.706
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         30/100      0.74G      1.067     0.7112     0.9463         12        640: 100%|██████████| 263/263 [00:36<00:00,  7.23it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:06<00:00,  8.63it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.952      0.907      0.965       0.59
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         31/100     0.757G      1.049     0.6913     0.9346          6        640: 100%|██████████| 263/263 [00:35<00:00,  7.34it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:04<00:00, 12.01it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.959      0.924      0.974      0.708
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         32/100     0.742G      1.092     0.6971     0.9431         18        640: 100%|██████████| 263/263 [00:35<00:00,  7.32it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:07<00:00,  8.11it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.948      0.918      0.964      0.534
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         33/100      0.74G      1.046     0.6593     0.9348          6        640: 100%|██████████| 263/263 [00:35<00:00,  7.42it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:05<00:00, 11.19it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.954      0.937      0.974      0.699
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         34/100     0.738G      1.088     0.6777     0.9423         14        640: 100%|██████████| 263/263 [00:35<00:00,  7.40it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:04<00:00, 11.96it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.971      0.935      0.979      0.728
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         35/100     0.742G      1.025     0.6625     0.9295         14        640: 100%|██████████| 263/263 [00:35<00:00,  7.39it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:07<00:00,  7.86it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.966      0.937      0.978      0.692
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         36/100     0.753G      1.033     0.6436     0.9359          9        640: 100%|██████████| 263/263 [00:35<00:00,  7.38it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:04<00:00, 11.85it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.962      0.932      0.973      0.647
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         37/100      0.74G      1.046     0.6571     0.9393          8        640: 100%|██████████| 263/263 [00:35<00:00,  7.48it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:05<00:00, 10.43it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.963      0.925      0.974      0.689
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         38/100     0.755G      1.025       0.65     0.9315         17        640: 100%|██████████| 263/263 [00:34<00:00,  7.59it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:06<00:00,  8.33it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.953      0.939      0.976      0.681
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         39/100     0.757G      1.042     0.6516     0.9356         11        640: 100%|██████████| 263/263 [00:35<00:00,  7.41it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:04<00:00, 11.82it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.955      0.918      0.968       0.59
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         40/100     0.757G      1.049      0.641     0.9417         17        640: 100%|██████████| 263/263 [00:35<00:00,  7.39it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:06<00:00,  8.53it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.959      0.925      0.974      0.668
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         41/100     0.742G      1.001     0.6127     0.9211         20        640: 100%|██████████| 263/263 [00:36<00:00,  7.25it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:05<00:00, 10.89it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985       0.96      0.938      0.976      0.659
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         42/100     0.751G      1.002     0.6161     0.9278         24        640: 100%|██████████| 263/263 [00:36<00:00,  7.26it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:04<00:00, 11.43it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.965      0.934      0.977      0.731
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         43/100     0.742G      1.044     0.6329     0.9298         17        640: 100%|██████████| 263/263 [00:36<00:00,  7.20it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:06<00:00,  8.32it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.959      0.932      0.974      0.651
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         44/100      0.74G      1.064     0.6302     0.9388         13        640: 100%|██████████| 263/263 [00:36<00:00,  7.26it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:04<00:00, 11.76it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.968      0.941       0.98      0.733
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         45/100     0.753G      0.979     0.5914     0.9223          9        640: 100%|██████████| 263/263 [00:36<00:00,  7.27it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:07<00:00,  7.89it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.963      0.925      0.972      0.633
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         46/100      0.74G     0.9783     0.5936     0.9216         30        640: 100%|██████████| 263/263 [00:36<00:00,  7.25it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:04<00:00, 11.50it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.964      0.933      0.975      0.678
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         47/100     0.755G      1.022     0.6088      0.933          9        640: 100%|██████████| 263/263 [00:36<00:00,  7.30it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:05<00:00, 11.25it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.961      0.925      0.969      0.544
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         48/100     0.742G     0.9822     0.5994     0.9291         16        640: 100%|██████████| 263/263 [00:36<00:00,  7.20it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:06<00:00,  8.47it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.954      0.932      0.968      0.589
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         49/100     0.755G     0.9927     0.5918     0.9244         19        640: 100%|██████████| 263/263 [00:36<00:00,  7.19it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:04<00:00, 11.82it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.968      0.936      0.979      0.723
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         50/100     0.753G     0.9908      0.596     0.9256         26        640: 100%|██████████| 263/263 [00:35<00:00,  7.35it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:07<00:00,  7.85it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.966      0.949      0.981      0.735
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         51/100     0.757G     0.9771     0.5821     0.9224         20        640: 100%|██████████| 263/263 [00:35<00:00,  7.37it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:04<00:00, 11.75it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.967      0.944       0.98      0.691
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         52/100     0.738G     0.9682     0.5873     0.9184         33        640: 100%|██████████| 263/263 [00:36<00:00,  7.14it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:06<00:00,  9.37it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.961      0.935      0.978      0.717
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         53/100     0.742G     0.9898     0.5804     0.9242         20        640: 100%|██████████| 263/263 [00:35<00:00,  7.31it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:05<00:00,  9.93it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985       0.96      0.944      0.978      0.675
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         54/100      0.74G     0.9763     0.5844     0.9218         10        640: 100%|██████████| 263/263 [00:35<00:00,  7.47it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:04<00:00, 11.76it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.966      0.941      0.981      0.752
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         55/100      0.74G     0.9735     0.5813     0.9197         15        640: 100%|██████████| 263/263 [00:35<00:00,  7.39it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:07<00:00,  7.72it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.966      0.945       0.98      0.722
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         56/100     0.738G     0.9507      0.563     0.9169         30        640: 100%|██████████| 263/263 [00:36<00:00,  7.23it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:04<00:00, 11.83it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.967      0.945      0.981      0.742
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         57/100      0.74G     0.9579     0.5721     0.9193          8        640: 100%|██████████| 263/263 [00:36<00:00,  7.24it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:05<00:00,  9.82it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.969      0.944      0.979        0.7
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         58/100     0.755G     0.9567     0.5718      0.917         15        640: 100%|██████████| 263/263 [00:37<00:00,  7.11it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:06<00:00,  9.35it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985       0.97      0.951      0.981      0.771
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         59/100     0.742G     0.9402     0.5619     0.9179         16        640: 100%|██████████| 263/263 [00:36<00:00,  7.16it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:04<00:00, 11.60it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.963      0.948      0.978      0.655
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         60/100     0.738G     0.9722     0.5725     0.9215         18        640: 100%|██████████| 263/263 [00:37<00:00,  6.99it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:06<00:00,  8.22it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985       0.97      0.955      0.982      0.749
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         61/100     0.742G     0.9457     0.5607      0.914         20        640: 100%|██████████| 263/263 [00:35<00:00,  7.41it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:04<00:00, 11.60it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.964       0.95      0.981      0.736
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         62/100      0.74G     0.9394     0.5516     0.9129          9        640: 100%|██████████| 263/263 [00:36<00:00,  7.25it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:06<00:00,  8.24it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985       0.97      0.942      0.979       0.69
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         63/100      0.74G     0.9333     0.5459     0.9086         11        640: 100%|██████████| 263/263 [00:38<00:00,  6.86it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:04<00:00, 11.57it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.969      0.953      0.982      0.709
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         64/100      0.74G     0.9303     0.5432     0.9112         28        640: 100%|██████████| 263/263 [00:35<00:00,  7.50it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:04<00:00, 11.70it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.972      0.942      0.977      0.663
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         65/100     0.738G     0.9305     0.5432     0.9104         40        640: 100%|██████████| 263/263 [00:34<00:00,  7.59it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:07<00:00,  8.02it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.967      0.947      0.979       0.72
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         66/100     0.755G     0.9425     0.5402     0.9124         28        640: 100%|██████████| 263/263 [00:34<00:00,  7.56it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:04<00:00, 12.48it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.971       0.95      0.981      0.671
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         67/100     0.751G     0.9266     0.5315     0.9059         23        640: 100%|██████████| 263/263 [00:34<00:00,  7.68it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:04<00:00, 12.36it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.976      0.952      0.983      0.766
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         68/100     0.755G     0.9267     0.5406     0.9075         15        640: 100%|██████████| 263/263 [00:34<00:00,  7.71it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:06<00:00,  8.49it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985       0.97      0.956      0.982      0.726
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         69/100      0.74G      0.903     0.5326     0.9045         11        640: 100%|██████████| 263/263 [00:34<00:00,  7.64it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:05<00:00, 10.85it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.966      0.954      0.982      0.681
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         70/100     0.742G     0.9097     0.5293      0.908         21        640: 100%|██████████| 263/263 [00:35<00:00,  7.39it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:04<00:00, 12.28it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.969      0.955      0.982      0.711
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         71/100     0.742G     0.9189     0.5345     0.9114         23        640: 100%|██████████| 263/263 [00:34<00:00,  7.59it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:07<00:00,  7.71it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.968      0.949       0.98      0.666
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         72/100     0.738G      0.906     0.5194     0.9039         30        640: 100%|██████████| 263/263 [00:34<00:00,  7.59it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:04<00:00, 12.19it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985       0.97      0.949       0.98       0.69
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         73/100     0.742G     0.9186      0.531     0.9137         19        640: 100%|██████████| 263/263 [00:34<00:00,  7.60it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:04<00:00, 12.06it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.968      0.946      0.979      0.675
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         74/100      0.74G     0.9189     0.5165     0.9064         12        640: 100%|██████████| 263/263 [00:34<00:00,  7.70it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:06<00:00,  8.61it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.975       0.95      0.984      0.761
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         75/100     0.742G     0.9068      0.519     0.9026         18        640: 100%|██████████| 263/263 [00:34<00:00,  7.71it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:05<00:00, 10.68it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.979      0.944      0.984      0.738
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         76/100      0.74G     0.8835     0.5069     0.9038         10        640: 100%|██████████| 263/263 [00:34<00:00,  7.59it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:04<00:00, 12.39it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.967      0.952      0.982      0.777
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         77/100     0.738G     0.8802     0.5054     0.9011         20        640: 100%|██████████| 263/263 [00:34<00:00,  7.56it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:05<00:00,  9.86it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.974      0.949      0.981       0.71
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         78/100      0.74G     0.8833     0.5044     0.8981          6        640: 100%|██████████| 263/263 [00:34<00:00,  7.62it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:06<00:00,  9.12it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.976      0.945      0.982      0.754
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         79/100     0.753G     0.9086     0.5106     0.9015         26        640: 100%|██████████| 263/263 [00:34<00:00,  7.53it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:04<00:00, 12.27it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985       0.97       0.95      0.982      0.701
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         80/100     0.742G     0.8636     0.5038     0.8992         19        640: 100%|██████████| 263/263 [00:34<00:00,  7.55it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:05<00:00, 10.33it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.975      0.945      0.981      0.716
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         81/100     0.753G      0.879     0.4955     0.8958         12        640: 100%|██████████| 263/263 [00:34<00:00,  7.54it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:06<00:00,  8.81it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.977      0.951      0.984      0.751
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         82/100     0.738G     0.8727     0.5048     0.8963         11        640: 100%|██████████| 263/263 [00:35<00:00,  7.45it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:04<00:00, 12.33it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.977      0.947      0.983      0.758
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         83/100     0.751G     0.8773     0.4897     0.9027         23        640: 100%|██████████| 263/263 [00:35<00:00,  7.48it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:05<00:00, 10.63it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.976      0.946      0.982      0.727
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         84/100     0.753G     0.8922     0.5035     0.9038         36        640: 100%|██████████| 263/263 [00:34<00:00,  7.60it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:06<00:00,  8.61it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.975      0.951      0.984      0.729
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         85/100     0.738G      0.872     0.4898      0.899         26        640: 100%|██████████| 263/263 [00:34<00:00,  7.57it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:04<00:00, 12.20it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.975      0.947      0.982      0.733
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         86/100     0.742G     0.8853     0.4934     0.9022         22        640: 100%|██████████| 263/263 [00:37<00:00,  6.95it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:05<00:00, 10.27it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.967      0.955      0.983      0.738
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         87/100     0.753G       0.86     0.4855     0.8963          8        640: 100%|██████████| 263/263 [00:33<00:00,  7.81it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:06<00:00,  8.41it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.975      0.953      0.984      0.739
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         88/100     0.742G     0.8539     0.4841     0.8964         15        640: 100%|██████████| 263/263 [00:34<00:00,  7.60it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:04<00:00, 12.19it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.974      0.955      0.984      0.702
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         89/100     0.738G     0.8541     0.4859     0.8958         31        640: 100%|██████████| 263/263 [00:34<00:00,  7.63it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:05<00:00, 11.19it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.973      0.951      0.983      0.734

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         90/100     0.738G     0.8388     0.4754     0.8922         24        640: 100%|██████████| 263/263 [00:33<00:00,  7.87it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:07<00:00,  8.04it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.969      0.954      0.984      0.742
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}
    Closing dataloader mosaic
    albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01), CLAHE(p=0.01, clip_limit=(1, 4.0), tile_grid_size=(8, 8))

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         91/100      0.74G     0.8354     0.4309     0.8955         15        640: 100%|██████████| 263/263 [00:34<00:00,  7.67it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:04<00:00, 11.98it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985       0.97      0.953      0.983      0.712
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         92/100     0.742G     0.8151     0.4234     0.8966         18        640: 100%|██████████| 263/263 [00:33<00:00,  7.87it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:04<00:00, 12.27it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.975       0.95      0.983       0.75
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         93/100      0.74G     0.7997     0.4152     0.8946         10        640: 100%|██████████| 263/263 [00:33<00:00,  7.95it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:05<00:00, 10.26it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.974      0.954      0.984      0.761
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         94/100      0.74G     0.8054     0.4159     0.8962         11        640: 100%|██████████| 263/263 [00:36<00:00,  7.16it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:06<00:00,  9.18it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.975      0.948      0.984      0.724
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         95/100      0.74G     0.8055     0.4131     0.8913         12        640: 100%|██████████| 263/263 [00:33<00:00,  7.83it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:04<00:00, 12.24it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.975      0.954      0.985      0.771
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         96/100      0.74G     0.7932     0.4104     0.8925         11        640: 100%|██████████| 263/263 [00:33<00:00,  7.93it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:04<00:00, 12.05it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.978      0.953      0.984      0.731
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         97/100     0.738G     0.7781     0.4066     0.8844         11        640: 100%|██████████| 263/263 [00:32<00:00,  7.98it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:05<00:00,  9.99it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.976      0.954      0.984      0.771
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         98/100     0.738G     0.7804      0.406     0.8875         14        640: 100%|██████████| 263/263 [00:32<00:00,  8.13it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:06<00:00,  8.56it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.978      0.952      0.985      0.765
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
         99/100      0.74G     0.7875     0.4053     0.8863         14        640: 100%|██████████| 263/263 [00:33<00:00,  7.91it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:04<00:00, 12.32it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.978      0.949      0.984      0.722
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
:::

::: {.output .stream .stderr}
        100/100      0.74G       0.79     0.4091     0.8882         16        640: 100%|██████████| 263/263 [00:33<00:00,  7.81it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:04<00:00, 12.28it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.978      0.948      0.984      0.732
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}

    100 epochs completed in 1.160 hours.
    Optimizer stripped from runs/detect/train8/weights/last.pt, 6.3MB
    Optimizer stripped from runs/detect/train8/weights/best.pt, 6.3MB

    Validating runs/detect/train8/weights/best.pt...
    Ultralytics YOLOv8.0.222 🚀 Python-3.10.12 torch-2.1.0+cu118 CUDA:0 (Tesla T4, 15102MiB)
    Model summary (fused): 168 layers, 3006818 parameters, 0 gradients, 8.1 GFLOPs
:::

::: {.output .stream .stderr}
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 57/57 [00:09<00:00,  5.92it/s]
:::

::: {.output .stream .stdout}
                       all        450       2985      0.967      0.952      0.982      0.777
                    1-open        450        588      0.965      0.976      0.983      0.708
                   2-short        450        451      0.945      0.916      0.974      0.691
               3-mousebite        450        589      0.974      0.959      0.985      0.756
                    4-spur        450        475      0.976      0.944      0.977      0.739
                  5-copper        450        439      0.995      0.951      0.987      0.889
                6-pin-hole        450        443      0.948      0.966      0.988      0.879
    Speed: 0.6ms preprocess, 4.8ms inference, 0.0ms loss, 2.4ms postprocess per image
    Results saved to runs/detect/train8
:::

::: {.output .execute_result execution_count="37"}
    ultralytics.utils.metrics.DetMetrics object with attributes:

    ap_class_index: array([0, 1, 2, 3, 4, 5])
    box: ultralytics.utils.metrics.Metric object
    confusion_matrix: <ultralytics.utils.metrics.ConfusionMatrix object at 0x7dd6a18a8dc0>
    curves: ['Precision-Recall(B)', 'F1-Confidence(B)', 'Precision-Confidence(B)', 'Recall-Confidence(B)']
    curves_results: [[array([          0,    0.001001,    0.002002,    0.003003,    0.004004,    0.005005,    0.006006,    0.007007,    0.008008,    0.009009,     0.01001,    0.011011,    0.012012,    0.013013,    0.014014,    0.015015,    0.016016,    0.017017,    0.018018,    0.019019,     0.02002,    0.021021,    0.022022,    0.023023,
              0.024024,    0.025025,    0.026026,    0.027027,    0.028028,    0.029029,     0.03003,    0.031031,    0.032032,    0.033033,    0.034034,    0.035035,    0.036036,    0.037037,    0.038038,    0.039039,     0.04004,    0.041041,    0.042042,    0.043043,    0.044044,    0.045045,    0.046046,    0.047047,
              0.048048,    0.049049,     0.05005,    0.051051,    0.052052,    0.053053,    0.054054,    0.055055,    0.056056,    0.057057,    0.058058,    0.059059,     0.06006,    0.061061,    0.062062,    0.063063,    0.064064,    0.065065,    0.066066,    0.067067,    0.068068,    0.069069,     0.07007,    0.071071,
              0.072072,    0.073073,    0.074074,    0.075075,    0.076076,    0.077077,    0.078078,    0.079079,     0.08008,    0.081081,    0.082082,    0.083083,    0.084084,    0.085085,    0.086086,    0.087087,    0.088088,    0.089089,     0.09009,    0.091091,    0.092092,    0.093093,    0.094094,    0.095095,
              0.096096,    0.097097,    0.098098,    0.099099,      0.1001,      0.1011,      0.1021,      0.1031,      0.1041,     0.10511,     0.10611,     0.10711,     0.10811,     0.10911,     0.11011,     0.11111,     0.11211,     0.11311,     0.11411,     0.11512,     0.11612,     0.11712,     0.11812,     0.11912,
               0.12012,     0.12112,     0.12212,     0.12312,     0.12412,     0.12513,     0.12613,     0.12713,     0.12813,     0.12913,     0.13013,     0.13113,     0.13213,     0.13313,     0.13413,     0.13514,     0.13614,     0.13714,     0.13814,     0.13914,     0.14014,     0.14114,     0.14214,     0.14314,
               0.14414,     0.14515,     0.14615,     0.14715,     0.14815,     0.14915,     0.15015,     0.15115,     0.15215,     0.15315,     0.15415,     0.15516,     0.15616,     0.15716,     0.15816,     0.15916,     0.16016,     0.16116,     0.16216,     0.16316,     0.16416,     0.16517,     0.16617,     0.16717,
               0.16817,     0.16917,     0.17017,     0.17117,     0.17217,     0.17317,     0.17417,     0.17518,     0.17618,     0.17718,     0.17818,     0.17918,     0.18018,     0.18118,     0.18218,     0.18318,     0.18418,     0.18519,     0.18619,     0.18719,     0.18819,     0.18919,     0.19019,     0.19119,
               0.19219,     0.19319,     0.19419,      0.1952,      0.1962,      0.1972,      0.1982,      0.1992,      0.2002,      0.2012,      0.2022,      0.2032,      0.2042,     0.20521,     0.20621,     0.20721,     0.20821,     0.20921,     0.21021,     0.21121,     0.21221,     0.21321,     0.21421,     0.21522,
               0.21622,     0.21722,     0.21822,     0.21922,     0.22022,     0.22122,     0.22222,     0.22322,     0.22422,     0.22523,     0.22623,     0.22723,     0.22823,     0.22923,     0.23023,     0.23123,     0.23223,     0.23323,     0.23423,     0.23524,     0.23624,     0.23724,     0.23824,     0.23924,
               0.24024,     0.24124,     0.24224,     0.24324,     0.24424,     0.24525,     0.24625,     0.24725,     0.24825,     0.24925,     0.25025,     0.25125,     0.25225,     0.25325,     0.25425,     0.25526,     0.25626,     0.25726,     0.25826,     0.25926,     0.26026,     0.26126,     0.26226,     0.26326,
               0.26426,     0.26527,     0.26627,     0.26727,     0.26827,     0.26927,     0.27027,     0.27127,     0.27227,     0.27327,     0.27427,     0.27528,     0.27628,     0.27728,     0.27828,     0.27928,     0.28028,     0.28128,     0.28228,     0.28328,     0.28428,     0.28529,     0.28629,     0.28729,
               0.28829,     0.28929,     0.29029,     0.29129,     0.29229,     0.29329,     0.29429,      0.2953,      0.2963,      0.2973,      0.2983,      0.2993,      0.3003,      0.3013,      0.3023,      0.3033,      0.3043,     0.30531,     0.30631,     0.30731,     0.30831,     0.30931,     0.31031,     0.31131,
               0.31231,     0.31331,     0.31431,     0.31532,     0.31632,     0.31732,     0.31832,     0.31932,     0.32032,     0.32132,     0.32232,     0.32332,     0.32432,     0.32533,     0.32633,     0.32733,     0.32833,     0.32933,     0.33033,     0.33133,     0.33233,     0.33333,     0.33433,     0.33534,
               0.33634,     0.33734,     0.33834,     0.33934,     0.34034,     0.34134,     0.34234,     0.34334,     0.34434,     0.34535,     0.34635,     0.34735,     0.34835,     0.34935,     0.35035,     0.35135,     0.35235,     0.35335,     0.35435,     0.35536,     0.35636,     0.35736,     0.35836,     0.35936,
               0.36036,     0.36136,     0.36236,     0.36336,     0.36436,     0.36537,     0.36637,     0.36737,     0.36837,     0.36937,     0.37037,     0.37137,     0.37237,     0.37337,     0.37437,     0.37538,     0.37638,     0.37738,     0.37838,     0.37938,     0.38038,     0.38138,     0.38238,     0.38338,
               0.38438,     0.38539,     0.38639,     0.38739,     0.38839,     0.38939,     0.39039,     0.39139,     0.39239,     0.39339,     0.39439,      0.3954,      0.3964,      0.3974,      0.3984,      0.3994,      0.4004,      0.4014,      0.4024,      0.4034,      0.4044,     0.40541,     0.40641,     0.40741,
               0.40841,     0.40941,     0.41041,     0.41141,     0.41241,     0.41341,     0.41441,     0.41542,     0.41642,     0.41742,     0.41842,     0.41942,     0.42042,     0.42142,     0.42242,     0.42342,     0.42442,     0.42543,     0.42643,     0.42743,     0.42843,     0.42943,     0.43043,     0.43143,
               0.43243,     0.43343,     0.43443,     0.43544,     0.43644,     0.43744,     0.43844,     0.43944,     0.44044,     0.44144,     0.44244,     0.44344,     0.44444,     0.44545,     0.44645,     0.44745,     0.44845,     0.44945,     0.45045,     0.45145,     0.45245,     0.45345,     0.45445,     0.45546,
               0.45646,     0.45746,     0.45846,     0.45946,     0.46046,     0.46146,     0.46246,     0.46346,     0.46446,     0.46547,     0.46647,     0.46747,     0.46847,     0.46947,     0.47047,     0.47147,     0.47247,     0.47347,     0.47447,     0.47548,     0.47648,     0.47748,     0.47848,     0.47948,
               0.48048,     0.48148,     0.48248,     0.48348,     0.48448,     0.48549,     0.48649,     0.48749,     0.48849,     0.48949,     0.49049,     0.49149,     0.49249,     0.49349,     0.49449,      0.4955,      0.4965,      0.4975,      0.4985,      0.4995,      0.5005,      0.5015,      0.5025,      0.5035,
                0.5045,     0.50551,     0.50651,     0.50751,     0.50851,     0.50951,     0.51051,     0.51151,     0.51251,     0.51351,     0.51451,     0.51552,     0.51652,     0.51752,     0.51852,     0.51952,     0.52052,     0.52152,     0.52252,     0.52352,     0.52452,     0.52553,     0.52653,     0.52753,
               0.52853,     0.52953,     0.53053,     0.53153,     0.53253,     0.53353,     0.53453,     0.53554,     0.53654,     0.53754,     0.53854,     0.53954,     0.54054,     0.54154,     0.54254,     0.54354,     0.54454,     0.54555,     0.54655,     0.54755,     0.54855,     0.54955,     0.55055,     0.55155,
               0.55255,     0.55355,     0.55455,     0.55556,     0.55656,     0.55756,     0.55856,     0.55956,     0.56056,     0.56156,     0.56256,     0.56356,     0.56456,     0.56557,     0.56657,     0.56757,     0.56857,     0.56957,     0.57057,     0.57157,     0.57257,     0.57357,     0.57457,     0.57558,
               0.57658,     0.57758,     0.57858,     0.57958,     0.58058,     0.58158,     0.58258,     0.58358,     0.58458,     0.58559,     0.58659,     0.58759,     0.58859,     0.58959,     0.59059,     0.59159,     0.59259,     0.59359,     0.59459,      0.5956,      0.5966,      0.5976,      0.5986,      0.5996,
                0.6006,      0.6016,      0.6026,      0.6036,      0.6046,     0.60561,     0.60661,     0.60761,     0.60861,     0.60961,     0.61061,     0.61161,     0.61261,     0.61361,     0.61461,     0.61562,     0.61662,     0.61762,     0.61862,     0.61962,     0.62062,     0.62162,     0.62262,     0.62362,
               0.62462,     0.62563,     0.62663,     0.62763,     0.62863,     0.62963,     0.63063,     0.63163,     0.63263,     0.63363,     0.63463,     0.63564,     0.63664,     0.63764,     0.63864,     0.63964,     0.64064,     0.64164,     0.64264,     0.64364,     0.64464,     0.64565,     0.64665,     0.64765,
               0.64865,     0.64965,     0.65065,     0.65165,     0.65265,     0.65365,     0.65465,     0.65566,     0.65666,     0.65766,     0.65866,     0.65966,     0.66066,     0.66166,     0.66266,     0.66366,     0.66466,     0.66567,     0.66667,     0.66767,     0.66867,     0.66967,     0.67067,     0.67167,
               0.67267,     0.67367,     0.67467,     0.67568,     0.67668,     0.67768,     0.67868,     0.67968,     0.68068,     0.68168,     0.68268,     0.68368,     0.68468,     0.68569,     0.68669,     0.68769,     0.68869,     0.68969,     0.69069,     0.69169,     0.69269,     0.69369,     0.69469,      0.6957,
                0.6967,      0.6977,      0.6987,      0.6997,      0.7007,      0.7017,      0.7027,      0.7037,      0.7047,     0.70571,     0.70671,     0.70771,     0.70871,     0.70971,     0.71071,     0.71171,     0.71271,     0.71371,     0.71471,     0.71572,     0.71672,     0.71772,     0.71872,     0.71972,
               0.72072,     0.72172,     0.72272,     0.72372,     0.72472,     0.72573,     0.72673,     0.72773,     0.72873,     0.72973,     0.73073,     0.73173,     0.73273,     0.73373,     0.73473,     0.73574,     0.73674,     0.73774,     0.73874,     0.73974,     0.74074,     0.74174,     0.74274,     0.74374,
               0.74474,     0.74575,     0.74675,     0.74775,     0.74875,     0.74975,     0.75075,     0.75175,     0.75275,     0.75375,     0.75475,     0.75576,     0.75676,     0.75776,     0.75876,     0.75976,     0.76076,     0.76176,     0.76276,     0.76376,     0.76476,     0.76577,     0.76677,     0.76777,
               0.76877,     0.76977,     0.77077,     0.77177,     0.77277,     0.77377,     0.77477,     0.77578,     0.77678,     0.77778,     0.77878,     0.77978,     0.78078,     0.78178,     0.78278,     0.78378,     0.78478,     0.78579,     0.78679,     0.78779,     0.78879,     0.78979,     0.79079,     0.79179,
               0.79279,     0.79379,     0.79479,      0.7958,      0.7968,      0.7978,      0.7988,      0.7998,      0.8008,      0.8018,      0.8028,      0.8038,      0.8048,     0.80581,     0.80681,     0.80781,     0.80881,     0.80981,     0.81081,     0.81181,     0.81281,     0.81381,     0.81481,     0.81582,
               0.81682,     0.81782,     0.81882,     0.81982,     0.82082,     0.82182,     0.82282,     0.82382,     0.82482,     0.82583,     0.82683,     0.82783,     0.82883,     0.82983,     0.83083,     0.83183,     0.83283,     0.83383,     0.83483,     0.83584,     0.83684,     0.83784,     0.83884,     0.83984,
               0.84084,     0.84184,     0.84284,     0.84384,     0.84484,     0.84585,     0.84685,     0.84785,     0.84885,     0.84985,     0.85085,     0.85185,     0.85285,     0.85385,     0.85485,     0.85586,     0.85686,     0.85786,     0.85886,     0.85986,     0.86086,     0.86186,     0.86286,     0.86386,
               0.86486,     0.86587,     0.86687,     0.86787,     0.86887,     0.86987,     0.87087,     0.87187,     0.87287,     0.87387,     0.87487,     0.87588,     0.87688,     0.87788,     0.87888,     0.87988,     0.88088,     0.88188,     0.88288,     0.88388,     0.88488,     0.88589,     0.88689,     0.88789,
               0.88889,     0.88989,     0.89089,     0.89189,     0.89289,     0.89389,     0.89489,      0.8959,      0.8969,      0.8979,      0.8989,      0.8999,      0.9009,      0.9019,      0.9029,      0.9039,      0.9049,     0.90591,     0.90691,     0.90791,     0.90891,     0.90991,     0.91091,     0.91191,
               0.91291,     0.91391,     0.91491,     0.91592,     0.91692,     0.91792,     0.91892,     0.91992,     0.92092,     0.92192,     0.92292,     0.92392,     0.92492,     0.92593,     0.92693,     0.92793,     0.92893,     0.92993,     0.93093,     0.93193,     0.93293,     0.93393,     0.93493,     0.93594,
               0.93694,     0.93794,     0.93894,     0.93994,     0.94094,     0.94194,     0.94294,     0.94394,     0.94494,     0.94595,     0.94695,     0.94795,     0.94895,     0.94995,     0.95095,     0.95195,     0.95295,     0.95395,     0.95495,     0.95596,     0.95696,     0.95796,     0.95896,     0.95996,
               0.96096,     0.96196,     0.96296,     0.96396,     0.96496,     0.96597,     0.96697,     0.96797,     0.96897,     0.96997,     0.97097,     0.97197,     0.97297,     0.97397,     0.97497,     0.97598,     0.97698,     0.97798,     0.97898,     0.97998,     0.98098,     0.98198,     0.98298,     0.98398,
               0.98498,     0.98599,     0.98699,     0.98799,     0.98899,     0.98999,     0.99099,     0.99199,     0.99299,     0.99399,     0.99499,       0.996,       0.997,       0.998,       0.999,           1]), array([[          1,           1,           1, ...,    0.066057,    0.033028,           0],
           [          1,           1,           1, ...,    0.079574,    0.039787,           0],
           [          1,           1,           1, ...,    0.070921,     0.03546,           0],
           [          1,           1,           1, ...,    0.042264,    0.021132,           0],
           [          1,           1,           1, ...,     0.13047,    0.065235,           0],
           [          1,           1,           1, ...,     0.37263,     0.18631,           0]]), 'Recall', 'Precision'], [array([          0,    0.001001,    0.002002,    0.003003,    0.004004,    0.005005,    0.006006,    0.007007,    0.008008,    0.009009,     0.01001,    0.011011,    0.012012,    0.013013,    0.014014,    0.015015,    0.016016,    0.017017,    0.018018,    0.019019,     0.02002,    0.021021,    0.022022,    0.023023,
              0.024024,    0.025025,    0.026026,    0.027027,    0.028028,    0.029029,     0.03003,    0.031031,    0.032032,    0.033033,    0.034034,    0.035035,    0.036036,    0.037037,    0.038038,    0.039039,     0.04004,    0.041041,    0.042042,    0.043043,    0.044044,    0.045045,    0.046046,    0.047047,
              0.048048,    0.049049,     0.05005,    0.051051,    0.052052,    0.053053,    0.054054,    0.055055,    0.056056,    0.057057,    0.058058,    0.059059,     0.06006,    0.061061,    0.062062,    0.063063,    0.064064,    0.065065,    0.066066,    0.067067,    0.068068,    0.069069,     0.07007,    0.071071,
              0.072072,    0.073073,    0.074074,    0.075075,    0.076076,    0.077077,    0.078078,    0.079079,     0.08008,    0.081081,    0.082082,    0.083083,    0.084084,    0.085085,    0.086086,    0.087087,    0.088088,    0.089089,     0.09009,    0.091091,    0.092092,    0.093093,    0.094094,    0.095095,
              0.096096,    0.097097,    0.098098,    0.099099,      0.1001,      0.1011,      0.1021,      0.1031,      0.1041,     0.10511,     0.10611,     0.10711,     0.10811,     0.10911,     0.11011,     0.11111,     0.11211,     0.11311,     0.11411,     0.11512,     0.11612,     0.11712,     0.11812,     0.11912,
               0.12012,     0.12112,     0.12212,     0.12312,     0.12412,     0.12513,     0.12613,     0.12713,     0.12813,     0.12913,     0.13013,     0.13113,     0.13213,     0.13313,     0.13413,     0.13514,     0.13614,     0.13714,     0.13814,     0.13914,     0.14014,     0.14114,     0.14214,     0.14314,
               0.14414,     0.14515,     0.14615,     0.14715,     0.14815,     0.14915,     0.15015,     0.15115,     0.15215,     0.15315,     0.15415,     0.15516,     0.15616,     0.15716,     0.15816,     0.15916,     0.16016,     0.16116,     0.16216,     0.16316,     0.16416,     0.16517,     0.16617,     0.16717,
               0.16817,     0.16917,     0.17017,     0.17117,     0.17217,     0.17317,     0.17417,     0.17518,     0.17618,     0.17718,     0.17818,     0.17918,     0.18018,     0.18118,     0.18218,     0.18318,     0.18418,     0.18519,     0.18619,     0.18719,     0.18819,     0.18919,     0.19019,     0.19119,
               0.19219,     0.19319,     0.19419,      0.1952,      0.1962,      0.1972,      0.1982,      0.1992,      0.2002,      0.2012,      0.2022,      0.2032,      0.2042,     0.20521,     0.20621,     0.20721,     0.20821,     0.20921,     0.21021,     0.21121,     0.21221,     0.21321,     0.21421,     0.21522,
               0.21622,     0.21722,     0.21822,     0.21922,     0.22022,     0.22122,     0.22222,     0.22322,     0.22422,     0.22523,     0.22623,     0.22723,     0.22823,     0.22923,     0.23023,     0.23123,     0.23223,     0.23323,     0.23423,     0.23524,     0.23624,     0.23724,     0.23824,     0.23924,
               0.24024,     0.24124,     0.24224,     0.24324,     0.24424,     0.24525,     0.24625,     0.24725,     0.24825,     0.24925,     0.25025,     0.25125,     0.25225,     0.25325,     0.25425,     0.25526,     0.25626,     0.25726,     0.25826,     0.25926,     0.26026,     0.26126,     0.26226,     0.26326,
               0.26426,     0.26527,     0.26627,     0.26727,     0.26827,     0.26927,     0.27027,     0.27127,     0.27227,     0.27327,     0.27427,     0.27528,     0.27628,     0.27728,     0.27828,     0.27928,     0.28028,     0.28128,     0.28228,     0.28328,     0.28428,     0.28529,     0.28629,     0.28729,
               0.28829,     0.28929,     0.29029,     0.29129,     0.29229,     0.29329,     0.29429,      0.2953,      0.2963,      0.2973,      0.2983,      0.2993,      0.3003,      0.3013,      0.3023,      0.3033,      0.3043,     0.30531,     0.30631,     0.30731,     0.30831,     0.30931,     0.31031,     0.31131,
               0.31231,     0.31331,     0.31431,     0.31532,     0.31632,     0.31732,     0.31832,     0.31932,     0.32032,     0.32132,     0.32232,     0.32332,     0.32432,     0.32533,     0.32633,     0.32733,     0.32833,     0.32933,     0.33033,     0.33133,     0.33233,     0.33333,     0.33433,     0.33534,
               0.33634,     0.33734,     0.33834,     0.33934,     0.34034,     0.34134,     0.34234,     0.34334,     0.34434,     0.34535,     0.34635,     0.34735,     0.34835,     0.34935,     0.35035,     0.35135,     0.35235,     0.35335,     0.35435,     0.35536,     0.35636,     0.35736,     0.35836,     0.35936,
               0.36036,     0.36136,     0.36236,     0.36336,     0.36436,     0.36537,     0.36637,     0.36737,     0.36837,     0.36937,     0.37037,     0.37137,     0.37237,     0.37337,     0.37437,     0.37538,     0.37638,     0.37738,     0.37838,     0.37938,     0.38038,     0.38138,     0.38238,     0.38338,
               0.38438,     0.38539,     0.38639,     0.38739,     0.38839,     0.38939,     0.39039,     0.39139,     0.39239,     0.39339,     0.39439,      0.3954,      0.3964,      0.3974,      0.3984,      0.3994,      0.4004,      0.4014,      0.4024,      0.4034,      0.4044,     0.40541,     0.40641,     0.40741,
               0.40841,     0.40941,     0.41041,     0.41141,     0.41241,     0.41341,     0.41441,     0.41542,     0.41642,     0.41742,     0.41842,     0.41942,     0.42042,     0.42142,     0.42242,     0.42342,     0.42442,     0.42543,     0.42643,     0.42743,     0.42843,     0.42943,     0.43043,     0.43143,
               0.43243,     0.43343,     0.43443,     0.43544,     0.43644,     0.43744,     0.43844,     0.43944,     0.44044,     0.44144,     0.44244,     0.44344,     0.44444,     0.44545,     0.44645,     0.44745,     0.44845,     0.44945,     0.45045,     0.45145,     0.45245,     0.45345,     0.45445,     0.45546,
               0.45646,     0.45746,     0.45846,     0.45946,     0.46046,     0.46146,     0.46246,     0.46346,     0.46446,     0.46547,     0.46647,     0.46747,     0.46847,     0.46947,     0.47047,     0.47147,     0.47247,     0.47347,     0.47447,     0.47548,     0.47648,     0.47748,     0.47848,     0.47948,
               0.48048,     0.48148,     0.48248,     0.48348,     0.48448,     0.48549,     0.48649,     0.48749,     0.48849,     0.48949,     0.49049,     0.49149,     0.49249,     0.49349,     0.49449,      0.4955,      0.4965,      0.4975,      0.4985,      0.4995,      0.5005,      0.5015,      0.5025,      0.5035,
                0.5045,     0.50551,     0.50651,     0.50751,     0.50851,     0.50951,     0.51051,     0.51151,     0.51251,     0.51351,     0.51451,     0.51552,     0.51652,     0.51752,     0.51852,     0.51952,     0.52052,     0.52152,     0.52252,     0.52352,     0.52452,     0.52553,     0.52653,     0.52753,
               0.52853,     0.52953,     0.53053,     0.53153,     0.53253,     0.53353,     0.53453,     0.53554,     0.53654,     0.53754,     0.53854,     0.53954,     0.54054,     0.54154,     0.54254,     0.54354,     0.54454,     0.54555,     0.54655,     0.54755,     0.54855,     0.54955,     0.55055,     0.55155,
               0.55255,     0.55355,     0.55455,     0.55556,     0.55656,     0.55756,     0.55856,     0.55956,     0.56056,     0.56156,     0.56256,     0.56356,     0.56456,     0.56557,     0.56657,     0.56757,     0.56857,     0.56957,     0.57057,     0.57157,     0.57257,     0.57357,     0.57457,     0.57558,
               0.57658,     0.57758,     0.57858,     0.57958,     0.58058,     0.58158,     0.58258,     0.58358,     0.58458,     0.58559,     0.58659,     0.58759,     0.58859,     0.58959,     0.59059,     0.59159,     0.59259,     0.59359,     0.59459,      0.5956,      0.5966,      0.5976,      0.5986,      0.5996,
                0.6006,      0.6016,      0.6026,      0.6036,      0.6046,     0.60561,     0.60661,     0.60761,     0.60861,     0.60961,     0.61061,     0.61161,     0.61261,     0.61361,     0.61461,     0.61562,     0.61662,     0.61762,     0.61862,     0.61962,     0.62062,     0.62162,     0.62262,     0.62362,
               0.62462,     0.62563,     0.62663,     0.62763,     0.62863,     0.62963,     0.63063,     0.63163,     0.63263,     0.63363,     0.63463,     0.63564,     0.63664,     0.63764,     0.63864,     0.63964,     0.64064,     0.64164,     0.64264,     0.64364,     0.64464,     0.64565,     0.64665,     0.64765,
               0.64865,     0.64965,     0.65065,     0.65165,     0.65265,     0.65365,     0.65465,     0.65566,     0.65666,     0.65766,     0.65866,     0.65966,     0.66066,     0.66166,     0.66266,     0.66366,     0.66466,     0.66567,     0.66667,     0.66767,     0.66867,     0.66967,     0.67067,     0.67167,
               0.67267,     0.67367,     0.67467,     0.67568,     0.67668,     0.67768,     0.67868,     0.67968,     0.68068,     0.68168,     0.68268,     0.68368,     0.68468,     0.68569,     0.68669,     0.68769,     0.68869,     0.68969,     0.69069,     0.69169,     0.69269,     0.69369,     0.69469,      0.6957,
                0.6967,      0.6977,      0.6987,      0.6997,      0.7007,      0.7017,      0.7027,      0.7037,      0.7047,     0.70571,     0.70671,     0.70771,     0.70871,     0.70971,     0.71071,     0.71171,     0.71271,     0.71371,     0.71471,     0.71572,     0.71672,     0.71772,     0.71872,     0.71972,
               0.72072,     0.72172,     0.72272,     0.72372,     0.72472,     0.72573,     0.72673,     0.72773,     0.72873,     0.72973,     0.73073,     0.73173,     0.73273,     0.73373,     0.73473,     0.73574,     0.73674,     0.73774,     0.73874,     0.73974,     0.74074,     0.74174,     0.74274,     0.74374,
               0.74474,     0.74575,     0.74675,     0.74775,     0.74875,     0.74975,     0.75075,     0.75175,     0.75275,     0.75375,     0.75475,     0.75576,     0.75676,     0.75776,     0.75876,     0.75976,     0.76076,     0.76176,     0.76276,     0.76376,     0.76476,     0.76577,     0.76677,     0.76777,
               0.76877,     0.76977,     0.77077,     0.77177,     0.77277,     0.77377,     0.77477,     0.77578,     0.77678,     0.77778,     0.77878,     0.77978,     0.78078,     0.78178,     0.78278,     0.78378,     0.78478,     0.78579,     0.78679,     0.78779,     0.78879,     0.78979,     0.79079,     0.79179,
               0.79279,     0.79379,     0.79479,      0.7958,      0.7968,      0.7978,      0.7988,      0.7998,      0.8008,      0.8018,      0.8028,      0.8038,      0.8048,     0.80581,     0.80681,     0.80781,     0.80881,     0.80981,     0.81081,     0.81181,     0.81281,     0.81381,     0.81481,     0.81582,
               0.81682,     0.81782,     0.81882,     0.81982,     0.82082,     0.82182,     0.82282,     0.82382,     0.82482,     0.82583,     0.82683,     0.82783,     0.82883,     0.82983,     0.83083,     0.83183,     0.83283,     0.83383,     0.83483,     0.83584,     0.83684,     0.83784,     0.83884,     0.83984,
               0.84084,     0.84184,     0.84284,     0.84384,     0.84484,     0.84585,     0.84685,     0.84785,     0.84885,     0.84985,     0.85085,     0.85185,     0.85285,     0.85385,     0.85485,     0.85586,     0.85686,     0.85786,     0.85886,     0.85986,     0.86086,     0.86186,     0.86286,     0.86386,
               0.86486,     0.86587,     0.86687,     0.86787,     0.86887,     0.86987,     0.87087,     0.87187,     0.87287,     0.87387,     0.87487,     0.87588,     0.87688,     0.87788,     0.87888,     0.87988,     0.88088,     0.88188,     0.88288,     0.88388,     0.88488,     0.88589,     0.88689,     0.88789,
               0.88889,     0.88989,     0.89089,     0.89189,     0.89289,     0.89389,     0.89489,      0.8959,      0.8969,      0.8979,      0.8989,      0.8999,      0.9009,      0.9019,      0.9029,      0.9039,      0.9049,     0.90591,     0.90691,     0.90791,     0.90891,     0.90991,     0.91091,     0.91191,
               0.91291,     0.91391,     0.91491,     0.91592,     0.91692,     0.91792,     0.91892,     0.91992,     0.92092,     0.92192,     0.92292,     0.92392,     0.92492,     0.92593,     0.92693,     0.92793,     0.92893,     0.92993,     0.93093,     0.93193,     0.93293,     0.93393,     0.93493,     0.93594,
               0.93694,     0.93794,     0.93894,     0.93994,     0.94094,     0.94194,     0.94294,     0.94394,     0.94494,     0.94595,     0.94695,     0.94795,     0.94895,     0.94995,     0.95095,     0.95195,     0.95295,     0.95395,     0.95495,     0.95596,     0.95696,     0.95796,     0.95896,     0.95996,
               0.96096,     0.96196,     0.96296,     0.96396,     0.96496,     0.96597,     0.96697,     0.96797,     0.96897,     0.96997,     0.97097,     0.97197,     0.97297,     0.97397,     0.97497,     0.97598,     0.97698,     0.97798,     0.97898,     0.97998,     0.98098,     0.98198,     0.98298,     0.98398,
               0.98498,     0.98599,     0.98699,     0.98799,     0.98899,     0.98999,     0.99099,     0.99199,     0.99299,     0.99399,     0.99499,       0.996,       0.997,       0.998,       0.999,           1]), array([[    0.61702,     0.61702,     0.71339, ...,           0,           0,           0],
           [    0.52007,     0.52007,     0.63067, ...,           0,           0,           0],
           [    0.46148,     0.46148,     0.57267, ...,           0,           0,           0],
           [    0.56829,     0.56829,     0.67928, ...,           0,           0,           0],
           [    0.61495,     0.61495,     0.72558, ...,           0,           0,           0],
           [     0.5913,      0.5913,       0.681, ...,           0,           0,           0]]), 'Confidence', 'F1'], [array([          0,    0.001001,    0.002002,    0.003003,    0.004004,    0.005005,    0.006006,    0.007007,    0.008008,    0.009009,     0.01001,    0.011011,    0.012012,    0.013013,    0.014014,    0.015015,    0.016016,    0.017017,    0.018018,    0.019019,     0.02002,    0.021021,    0.022022,    0.023023,
              0.024024,    0.025025,    0.026026,    0.027027,    0.028028,    0.029029,     0.03003,    0.031031,    0.032032,    0.033033,    0.034034,    0.035035,    0.036036,    0.037037,    0.038038,    0.039039,     0.04004,    0.041041,    0.042042,    0.043043,    0.044044,    0.045045,    0.046046,    0.047047,
              0.048048,    0.049049,     0.05005,    0.051051,    0.052052,    0.053053,    0.054054,    0.055055,    0.056056,    0.057057,    0.058058,    0.059059,     0.06006,    0.061061,    0.062062,    0.063063,    0.064064,    0.065065,    0.066066,    0.067067,    0.068068,    0.069069,     0.07007,    0.071071,
              0.072072,    0.073073,    0.074074,    0.075075,    0.076076,    0.077077,    0.078078,    0.079079,     0.08008,    0.081081,    0.082082,    0.083083,    0.084084,    0.085085,    0.086086,    0.087087,    0.088088,    0.089089,     0.09009,    0.091091,    0.092092,    0.093093,    0.094094,    0.095095,
              0.096096,    0.097097,    0.098098,    0.099099,      0.1001,      0.1011,      0.1021,      0.1031,      0.1041,     0.10511,     0.10611,     0.10711,     0.10811,     0.10911,     0.11011,     0.11111,     0.11211,     0.11311,     0.11411,     0.11512,     0.11612,     0.11712,     0.11812,     0.11912,
               0.12012,     0.12112,     0.12212,     0.12312,     0.12412,     0.12513,     0.12613,     0.12713,     0.12813,     0.12913,     0.13013,     0.13113,     0.13213,     0.13313,     0.13413,     0.13514,     0.13614,     0.13714,     0.13814,     0.13914,     0.14014,     0.14114,     0.14214,     0.14314,
               0.14414,     0.14515,     0.14615,     0.14715,     0.14815,     0.14915,     0.15015,     0.15115,     0.15215,     0.15315,     0.15415,     0.15516,     0.15616,     0.15716,     0.15816,     0.15916,     0.16016,     0.16116,     0.16216,     0.16316,     0.16416,     0.16517,     0.16617,     0.16717,
               0.16817,     0.16917,     0.17017,     0.17117,     0.17217,     0.17317,     0.17417,     0.17518,     0.17618,     0.17718,     0.17818,     0.17918,     0.18018,     0.18118,     0.18218,     0.18318,     0.18418,     0.18519,     0.18619,     0.18719,     0.18819,     0.18919,     0.19019,     0.19119,
               0.19219,     0.19319,     0.19419,      0.1952,      0.1962,      0.1972,      0.1982,      0.1992,      0.2002,      0.2012,      0.2022,      0.2032,      0.2042,     0.20521,     0.20621,     0.20721,     0.20821,     0.20921,     0.21021,     0.21121,     0.21221,     0.21321,     0.21421,     0.21522,
               0.21622,     0.21722,     0.21822,     0.21922,     0.22022,     0.22122,     0.22222,     0.22322,     0.22422,     0.22523,     0.22623,     0.22723,     0.22823,     0.22923,     0.23023,     0.23123,     0.23223,     0.23323,     0.23423,     0.23524,     0.23624,     0.23724,     0.23824,     0.23924,
               0.24024,     0.24124,     0.24224,     0.24324,     0.24424,     0.24525,     0.24625,     0.24725,     0.24825,     0.24925,     0.25025,     0.25125,     0.25225,     0.25325,     0.25425,     0.25526,     0.25626,     0.25726,     0.25826,     0.25926,     0.26026,     0.26126,     0.26226,     0.26326,
               0.26426,     0.26527,     0.26627,     0.26727,     0.26827,     0.26927,     0.27027,     0.27127,     0.27227,     0.27327,     0.27427,     0.27528,     0.27628,     0.27728,     0.27828,     0.27928,     0.28028,     0.28128,     0.28228,     0.28328,     0.28428,     0.28529,     0.28629,     0.28729,
               0.28829,     0.28929,     0.29029,     0.29129,     0.29229,     0.29329,     0.29429,      0.2953,      0.2963,      0.2973,      0.2983,      0.2993,      0.3003,      0.3013,      0.3023,      0.3033,      0.3043,     0.30531,     0.30631,     0.30731,     0.30831,     0.30931,     0.31031,     0.31131,
               0.31231,     0.31331,     0.31431,     0.31532,     0.31632,     0.31732,     0.31832,     0.31932,     0.32032,     0.32132,     0.32232,     0.32332,     0.32432,     0.32533,     0.32633,     0.32733,     0.32833,     0.32933,     0.33033,     0.33133,     0.33233,     0.33333,     0.33433,     0.33534,
               0.33634,     0.33734,     0.33834,     0.33934,     0.34034,     0.34134,     0.34234,     0.34334,     0.34434,     0.34535,     0.34635,     0.34735,     0.34835,     0.34935,     0.35035,     0.35135,     0.35235,     0.35335,     0.35435,     0.35536,     0.35636,     0.35736,     0.35836,     0.35936,
               0.36036,     0.36136,     0.36236,     0.36336,     0.36436,     0.36537,     0.36637,     0.36737,     0.36837,     0.36937,     0.37037,     0.37137,     0.37237,     0.37337,     0.37437,     0.37538,     0.37638,     0.37738,     0.37838,     0.37938,     0.38038,     0.38138,     0.38238,     0.38338,
               0.38438,     0.38539,     0.38639,     0.38739,     0.38839,     0.38939,     0.39039,     0.39139,     0.39239,     0.39339,     0.39439,      0.3954,      0.3964,      0.3974,      0.3984,      0.3994,      0.4004,      0.4014,      0.4024,      0.4034,      0.4044,     0.40541,     0.40641,     0.40741,
               0.40841,     0.40941,     0.41041,     0.41141,     0.41241,     0.41341,     0.41441,     0.41542,     0.41642,     0.41742,     0.41842,     0.41942,     0.42042,     0.42142,     0.42242,     0.42342,     0.42442,     0.42543,     0.42643,     0.42743,     0.42843,     0.42943,     0.43043,     0.43143,
               0.43243,     0.43343,     0.43443,     0.43544,     0.43644,     0.43744,     0.43844,     0.43944,     0.44044,     0.44144,     0.44244,     0.44344,     0.44444,     0.44545,     0.44645,     0.44745,     0.44845,     0.44945,     0.45045,     0.45145,     0.45245,     0.45345,     0.45445,     0.45546,
               0.45646,     0.45746,     0.45846,     0.45946,     0.46046,     0.46146,     0.46246,     0.46346,     0.46446,     0.46547,     0.46647,     0.46747,     0.46847,     0.46947,     0.47047,     0.47147,     0.47247,     0.47347,     0.47447,     0.47548,     0.47648,     0.47748,     0.47848,     0.47948,
               0.48048,     0.48148,     0.48248,     0.48348,     0.48448,     0.48549,     0.48649,     0.48749,     0.48849,     0.48949,     0.49049,     0.49149,     0.49249,     0.49349,     0.49449,      0.4955,      0.4965,      0.4975,      0.4985,      0.4995,      0.5005,      0.5015,      0.5025,      0.5035,
                0.5045,     0.50551,     0.50651,     0.50751,     0.50851,     0.50951,     0.51051,     0.51151,     0.51251,     0.51351,     0.51451,     0.51552,     0.51652,     0.51752,     0.51852,     0.51952,     0.52052,     0.52152,     0.52252,     0.52352,     0.52452,     0.52553,     0.52653,     0.52753,
               0.52853,     0.52953,     0.53053,     0.53153,     0.53253,     0.53353,     0.53453,     0.53554,     0.53654,     0.53754,     0.53854,     0.53954,     0.54054,     0.54154,     0.54254,     0.54354,     0.54454,     0.54555,     0.54655,     0.54755,     0.54855,     0.54955,     0.55055,     0.55155,
               0.55255,     0.55355,     0.55455,     0.55556,     0.55656,     0.55756,     0.55856,     0.55956,     0.56056,     0.56156,     0.56256,     0.56356,     0.56456,     0.56557,     0.56657,     0.56757,     0.56857,     0.56957,     0.57057,     0.57157,     0.57257,     0.57357,     0.57457,     0.57558,
               0.57658,     0.57758,     0.57858,     0.57958,     0.58058,     0.58158,     0.58258,     0.58358,     0.58458,     0.58559,     0.58659,     0.58759,     0.58859,     0.58959,     0.59059,     0.59159,     0.59259,     0.59359,     0.59459,      0.5956,      0.5966,      0.5976,      0.5986,      0.5996,
                0.6006,      0.6016,      0.6026,      0.6036,      0.6046,     0.60561,     0.60661,     0.60761,     0.60861,     0.60961,     0.61061,     0.61161,     0.61261,     0.61361,     0.61461,     0.61562,     0.61662,     0.61762,     0.61862,     0.61962,     0.62062,     0.62162,     0.62262,     0.62362,
               0.62462,     0.62563,     0.62663,     0.62763,     0.62863,     0.62963,     0.63063,     0.63163,     0.63263,     0.63363,     0.63463,     0.63564,     0.63664,     0.63764,     0.63864,     0.63964,     0.64064,     0.64164,     0.64264,     0.64364,     0.64464,     0.64565,     0.64665,     0.64765,
               0.64865,     0.64965,     0.65065,     0.65165,     0.65265,     0.65365,     0.65465,     0.65566,     0.65666,     0.65766,     0.65866,     0.65966,     0.66066,     0.66166,     0.66266,     0.66366,     0.66466,     0.66567,     0.66667,     0.66767,     0.66867,     0.66967,     0.67067,     0.67167,
               0.67267,     0.67367,     0.67467,     0.67568,     0.67668,     0.67768,     0.67868,     0.67968,     0.68068,     0.68168,     0.68268,     0.68368,     0.68468,     0.68569,     0.68669,     0.68769,     0.68869,     0.68969,     0.69069,     0.69169,     0.69269,     0.69369,     0.69469,      0.6957,
                0.6967,      0.6977,      0.6987,      0.6997,      0.7007,      0.7017,      0.7027,      0.7037,      0.7047,     0.70571,     0.70671,     0.70771,     0.70871,     0.70971,     0.71071,     0.71171,     0.71271,     0.71371,     0.71471,     0.71572,     0.71672,     0.71772,     0.71872,     0.71972,
               0.72072,     0.72172,     0.72272,     0.72372,     0.72472,     0.72573,     0.72673,     0.72773,     0.72873,     0.72973,     0.73073,     0.73173,     0.73273,     0.73373,     0.73473,     0.73574,     0.73674,     0.73774,     0.73874,     0.73974,     0.74074,     0.74174,     0.74274,     0.74374,
               0.74474,     0.74575,     0.74675,     0.74775,     0.74875,     0.74975,     0.75075,     0.75175,     0.75275,     0.75375,     0.75475,     0.75576,     0.75676,     0.75776,     0.75876,     0.75976,     0.76076,     0.76176,     0.76276,     0.76376,     0.76476,     0.76577,     0.76677,     0.76777,
               0.76877,     0.76977,     0.77077,     0.77177,     0.77277,     0.77377,     0.77477,     0.77578,     0.77678,     0.77778,     0.77878,     0.77978,     0.78078,     0.78178,     0.78278,     0.78378,     0.78478,     0.78579,     0.78679,     0.78779,     0.78879,     0.78979,     0.79079,     0.79179,
               0.79279,     0.79379,     0.79479,      0.7958,      0.7968,      0.7978,      0.7988,      0.7998,      0.8008,      0.8018,      0.8028,      0.8038,      0.8048,     0.80581,     0.80681,     0.80781,     0.80881,     0.80981,     0.81081,     0.81181,     0.81281,     0.81381,     0.81481,     0.81582,
               0.81682,     0.81782,     0.81882,     0.81982,     0.82082,     0.82182,     0.82282,     0.82382,     0.82482,     0.82583,     0.82683,     0.82783,     0.82883,     0.82983,     0.83083,     0.83183,     0.83283,     0.83383,     0.83483,     0.83584,     0.83684,     0.83784,     0.83884,     0.83984,
               0.84084,     0.84184,     0.84284,     0.84384,     0.84484,     0.84585,     0.84685,     0.84785,     0.84885,     0.84985,     0.85085,     0.85185,     0.85285,     0.85385,     0.85485,     0.85586,     0.85686,     0.85786,     0.85886,     0.85986,     0.86086,     0.86186,     0.86286,     0.86386,
               0.86486,     0.86587,     0.86687,     0.86787,     0.86887,     0.86987,     0.87087,     0.87187,     0.87287,     0.87387,     0.87487,     0.87588,     0.87688,     0.87788,     0.87888,     0.87988,     0.88088,     0.88188,     0.88288,     0.88388,     0.88488,     0.88589,     0.88689,     0.88789,
               0.88889,     0.88989,     0.89089,     0.89189,     0.89289,     0.89389,     0.89489,      0.8959,      0.8969,      0.8979,      0.8989,      0.8999,      0.9009,      0.9019,      0.9029,      0.9039,      0.9049,     0.90591,     0.90691,     0.90791,     0.90891,     0.90991,     0.91091,     0.91191,
               0.91291,     0.91391,     0.91491,     0.91592,     0.91692,     0.91792,     0.91892,     0.91992,     0.92092,     0.92192,     0.92292,     0.92392,     0.92492,     0.92593,     0.92693,     0.92793,     0.92893,     0.92993,     0.93093,     0.93193,     0.93293,     0.93393,     0.93493,     0.93594,
               0.93694,     0.93794,     0.93894,     0.93994,     0.94094,     0.94194,     0.94294,     0.94394,     0.94494,     0.94595,     0.94695,     0.94795,     0.94895,     0.94995,     0.95095,     0.95195,     0.95295,     0.95395,     0.95495,     0.95596,     0.95696,     0.95796,     0.95896,     0.95996,
               0.96096,     0.96196,     0.96296,     0.96396,     0.96496,     0.96597,     0.96697,     0.96797,     0.96897,     0.96997,     0.97097,     0.97197,     0.97297,     0.97397,     0.97497,     0.97598,     0.97698,     0.97798,     0.97898,     0.97998,     0.98098,     0.98198,     0.98298,     0.98398,
               0.98498,     0.98599,     0.98699,     0.98799,     0.98899,     0.98999,     0.99099,     0.99199,     0.99299,     0.99399,     0.99499,       0.996,       0.997,       0.998,       0.999,           1]), array([[    0.44892,     0.44892,     0.55874, ...,           1,           1,           1],
           [    0.35252,     0.35252,     0.46393, ...,           1,           1,           1],
           [    0.30072,     0.30072,     0.40288, ...,           1,           1,           1],
           [        0.4,         0.4,     0.52008, ...,           1,           1,           1],
           [    0.44535,     0.44535,     0.57233, ...,           1,           1,           1],
           [    0.42015,     0.42015,      0.5169, ...,           1,           1,           1]]), 'Confidence', 'Precision'], [array([          0,    0.001001,    0.002002,    0.003003,    0.004004,    0.005005,    0.006006,    0.007007,    0.008008,    0.009009,     0.01001,    0.011011,    0.012012,    0.013013,    0.014014,    0.015015,    0.016016,    0.017017,    0.018018,    0.019019,     0.02002,    0.021021,    0.022022,    0.023023,
              0.024024,    0.025025,    0.026026,    0.027027,    0.028028,    0.029029,     0.03003,    0.031031,    0.032032,    0.033033,    0.034034,    0.035035,    0.036036,    0.037037,    0.038038,    0.039039,     0.04004,    0.041041,    0.042042,    0.043043,    0.044044,    0.045045,    0.046046,    0.047047,
              0.048048,    0.049049,     0.05005,    0.051051,    0.052052,    0.053053,    0.054054,    0.055055,    0.056056,    0.057057,    0.058058,    0.059059,     0.06006,    0.061061,    0.062062,    0.063063,    0.064064,    0.065065,    0.066066,    0.067067,    0.068068,    0.069069,     0.07007,    0.071071,
              0.072072,    0.073073,    0.074074,    0.075075,    0.076076,    0.077077,    0.078078,    0.079079,     0.08008,    0.081081,    0.082082,    0.083083,    0.084084,    0.085085,    0.086086,    0.087087,    0.088088,    0.089089,     0.09009,    0.091091,    0.092092,    0.093093,    0.094094,    0.095095,
              0.096096,    0.097097,    0.098098,    0.099099,      0.1001,      0.1011,      0.1021,      0.1031,      0.1041,     0.10511,     0.10611,     0.10711,     0.10811,     0.10911,     0.11011,     0.11111,     0.11211,     0.11311,     0.11411,     0.11512,     0.11612,     0.11712,     0.11812,     0.11912,
               0.12012,     0.12112,     0.12212,     0.12312,     0.12412,     0.12513,     0.12613,     0.12713,     0.12813,     0.12913,     0.13013,     0.13113,     0.13213,     0.13313,     0.13413,     0.13514,     0.13614,     0.13714,     0.13814,     0.13914,     0.14014,     0.14114,     0.14214,     0.14314,
               0.14414,     0.14515,     0.14615,     0.14715,     0.14815,     0.14915,     0.15015,     0.15115,     0.15215,     0.15315,     0.15415,     0.15516,     0.15616,     0.15716,     0.15816,     0.15916,     0.16016,     0.16116,     0.16216,     0.16316,     0.16416,     0.16517,     0.16617,     0.16717,
               0.16817,     0.16917,     0.17017,     0.17117,     0.17217,     0.17317,     0.17417,     0.17518,     0.17618,     0.17718,     0.17818,     0.17918,     0.18018,     0.18118,     0.18218,     0.18318,     0.18418,     0.18519,     0.18619,     0.18719,     0.18819,     0.18919,     0.19019,     0.19119,
               0.19219,     0.19319,     0.19419,      0.1952,      0.1962,      0.1972,      0.1982,      0.1992,      0.2002,      0.2012,      0.2022,      0.2032,      0.2042,     0.20521,     0.20621,     0.20721,     0.20821,     0.20921,     0.21021,     0.21121,     0.21221,     0.21321,     0.21421,     0.21522,
               0.21622,     0.21722,     0.21822,     0.21922,     0.22022,     0.22122,     0.22222,     0.22322,     0.22422,     0.22523,     0.22623,     0.22723,     0.22823,     0.22923,     0.23023,     0.23123,     0.23223,     0.23323,     0.23423,     0.23524,     0.23624,     0.23724,     0.23824,     0.23924,
               0.24024,     0.24124,     0.24224,     0.24324,     0.24424,     0.24525,     0.24625,     0.24725,     0.24825,     0.24925,     0.25025,     0.25125,     0.25225,     0.25325,     0.25425,     0.25526,     0.25626,     0.25726,     0.25826,     0.25926,     0.26026,     0.26126,     0.26226,     0.26326,
               0.26426,     0.26527,     0.26627,     0.26727,     0.26827,     0.26927,     0.27027,     0.27127,     0.27227,     0.27327,     0.27427,     0.27528,     0.27628,     0.27728,     0.27828,     0.27928,     0.28028,     0.28128,     0.28228,     0.28328,     0.28428,     0.28529,     0.28629,     0.28729,
               0.28829,     0.28929,     0.29029,     0.29129,     0.29229,     0.29329,     0.29429,      0.2953,      0.2963,      0.2973,      0.2983,      0.2993,      0.3003,      0.3013,      0.3023,      0.3033,      0.3043,     0.30531,     0.30631,     0.30731,     0.30831,     0.30931,     0.31031,     0.31131,
               0.31231,     0.31331,     0.31431,     0.31532,     0.31632,     0.31732,     0.31832,     0.31932,     0.32032,     0.32132,     0.32232,     0.32332,     0.32432,     0.32533,     0.32633,     0.32733,     0.32833,     0.32933,     0.33033,     0.33133,     0.33233,     0.33333,     0.33433,     0.33534,
               0.33634,     0.33734,     0.33834,     0.33934,     0.34034,     0.34134,     0.34234,     0.34334,     0.34434,     0.34535,     0.34635,     0.34735,     0.34835,     0.34935,     0.35035,     0.35135,     0.35235,     0.35335,     0.35435,     0.35536,     0.35636,     0.35736,     0.35836,     0.35936,
               0.36036,     0.36136,     0.36236,     0.36336,     0.36436,     0.36537,     0.36637,     0.36737,     0.36837,     0.36937,     0.37037,     0.37137,     0.37237,     0.37337,     0.37437,     0.37538,     0.37638,     0.37738,     0.37838,     0.37938,     0.38038,     0.38138,     0.38238,     0.38338,
               0.38438,     0.38539,     0.38639,     0.38739,     0.38839,     0.38939,     0.39039,     0.39139,     0.39239,     0.39339,     0.39439,      0.3954,      0.3964,      0.3974,      0.3984,      0.3994,      0.4004,      0.4014,      0.4024,      0.4034,      0.4044,     0.40541,     0.40641,     0.40741,
               0.40841,     0.40941,     0.41041,     0.41141,     0.41241,     0.41341,     0.41441,     0.41542,     0.41642,     0.41742,     0.41842,     0.41942,     0.42042,     0.42142,     0.42242,     0.42342,     0.42442,     0.42543,     0.42643,     0.42743,     0.42843,     0.42943,     0.43043,     0.43143,
               0.43243,     0.43343,     0.43443,     0.43544,     0.43644,     0.43744,     0.43844,     0.43944,     0.44044,     0.44144,     0.44244,     0.44344,     0.44444,     0.44545,     0.44645,     0.44745,     0.44845,     0.44945,     0.45045,     0.45145,     0.45245,     0.45345,     0.45445,     0.45546,
               0.45646,     0.45746,     0.45846,     0.45946,     0.46046,     0.46146,     0.46246,     0.46346,     0.46446,     0.46547,     0.46647,     0.46747,     0.46847,     0.46947,     0.47047,     0.47147,     0.47247,     0.47347,     0.47447,     0.47548,     0.47648,     0.47748,     0.47848,     0.47948,
               0.48048,     0.48148,     0.48248,     0.48348,     0.48448,     0.48549,     0.48649,     0.48749,     0.48849,     0.48949,     0.49049,     0.49149,     0.49249,     0.49349,     0.49449,      0.4955,      0.4965,      0.4975,      0.4985,      0.4995,      0.5005,      0.5015,      0.5025,      0.5035,
                0.5045,     0.50551,     0.50651,     0.50751,     0.50851,     0.50951,     0.51051,     0.51151,     0.51251,     0.51351,     0.51451,     0.51552,     0.51652,     0.51752,     0.51852,     0.51952,     0.52052,     0.52152,     0.52252,     0.52352,     0.52452,     0.52553,     0.52653,     0.52753,
               0.52853,     0.52953,     0.53053,     0.53153,     0.53253,     0.53353,     0.53453,     0.53554,     0.53654,     0.53754,     0.53854,     0.53954,     0.54054,     0.54154,     0.54254,     0.54354,     0.54454,     0.54555,     0.54655,     0.54755,     0.54855,     0.54955,     0.55055,     0.55155,
               0.55255,     0.55355,     0.55455,     0.55556,     0.55656,     0.55756,     0.55856,     0.55956,     0.56056,     0.56156,     0.56256,     0.56356,     0.56456,     0.56557,     0.56657,     0.56757,     0.56857,     0.56957,     0.57057,     0.57157,     0.57257,     0.57357,     0.57457,     0.57558,
               0.57658,     0.57758,     0.57858,     0.57958,     0.58058,     0.58158,     0.58258,     0.58358,     0.58458,     0.58559,     0.58659,     0.58759,     0.58859,     0.58959,     0.59059,     0.59159,     0.59259,     0.59359,     0.59459,      0.5956,      0.5966,      0.5976,      0.5986,      0.5996,
                0.6006,      0.6016,      0.6026,      0.6036,      0.6046,     0.60561,     0.60661,     0.60761,     0.60861,     0.60961,     0.61061,     0.61161,     0.61261,     0.61361,     0.61461,     0.61562,     0.61662,     0.61762,     0.61862,     0.61962,     0.62062,     0.62162,     0.62262,     0.62362,
               0.62462,     0.62563,     0.62663,     0.62763,     0.62863,     0.62963,     0.63063,     0.63163,     0.63263,     0.63363,     0.63463,     0.63564,     0.63664,     0.63764,     0.63864,     0.63964,     0.64064,     0.64164,     0.64264,     0.64364,     0.64464,     0.64565,     0.64665,     0.64765,
               0.64865,     0.64965,     0.65065,     0.65165,     0.65265,     0.65365,     0.65465,     0.65566,     0.65666,     0.65766,     0.65866,     0.65966,     0.66066,     0.66166,     0.66266,     0.66366,     0.66466,     0.66567,     0.66667,     0.66767,     0.66867,     0.66967,     0.67067,     0.67167,
               0.67267,     0.67367,     0.67467,     0.67568,     0.67668,     0.67768,     0.67868,     0.67968,     0.68068,     0.68168,     0.68268,     0.68368,     0.68468,     0.68569,     0.68669,     0.68769,     0.68869,     0.68969,     0.69069,     0.69169,     0.69269,     0.69369,     0.69469,      0.6957,
                0.6967,      0.6977,      0.6987,      0.6997,      0.7007,      0.7017,      0.7027,      0.7037,      0.7047,     0.70571,     0.70671,     0.70771,     0.70871,     0.70971,     0.71071,     0.71171,     0.71271,     0.71371,     0.71471,     0.71572,     0.71672,     0.71772,     0.71872,     0.71972,
               0.72072,     0.72172,     0.72272,     0.72372,     0.72472,     0.72573,     0.72673,     0.72773,     0.72873,     0.72973,     0.73073,     0.73173,     0.73273,     0.73373,     0.73473,     0.73574,     0.73674,     0.73774,     0.73874,     0.73974,     0.74074,     0.74174,     0.74274,     0.74374,
               0.74474,     0.74575,     0.74675,     0.74775,     0.74875,     0.74975,     0.75075,     0.75175,     0.75275,     0.75375,     0.75475,     0.75576,     0.75676,     0.75776,     0.75876,     0.75976,     0.76076,     0.76176,     0.76276,     0.76376,     0.76476,     0.76577,     0.76677,     0.76777,
               0.76877,     0.76977,     0.77077,     0.77177,     0.77277,     0.77377,     0.77477,     0.77578,     0.77678,     0.77778,     0.77878,     0.77978,     0.78078,     0.78178,     0.78278,     0.78378,     0.78478,     0.78579,     0.78679,     0.78779,     0.78879,     0.78979,     0.79079,     0.79179,
               0.79279,     0.79379,     0.79479,      0.7958,      0.7968,      0.7978,      0.7988,      0.7998,      0.8008,      0.8018,      0.8028,      0.8038,      0.8048,     0.80581,     0.80681,     0.80781,     0.80881,     0.80981,     0.81081,     0.81181,     0.81281,     0.81381,     0.81481,     0.81582,
               0.81682,     0.81782,     0.81882,     0.81982,     0.82082,     0.82182,     0.82282,     0.82382,     0.82482,     0.82583,     0.82683,     0.82783,     0.82883,     0.82983,     0.83083,     0.83183,     0.83283,     0.83383,     0.83483,     0.83584,     0.83684,     0.83784,     0.83884,     0.83984,
               0.84084,     0.84184,     0.84284,     0.84384,     0.84484,     0.84585,     0.84685,     0.84785,     0.84885,     0.84985,     0.85085,     0.85185,     0.85285,     0.85385,     0.85485,     0.85586,     0.85686,     0.85786,     0.85886,     0.85986,     0.86086,     0.86186,     0.86286,     0.86386,
               0.86486,     0.86587,     0.86687,     0.86787,     0.86887,     0.86987,     0.87087,     0.87187,     0.87287,     0.87387,     0.87487,     0.87588,     0.87688,     0.87788,     0.87888,     0.87988,     0.88088,     0.88188,     0.88288,     0.88388,     0.88488,     0.88589,     0.88689,     0.88789,
               0.88889,     0.88989,     0.89089,     0.89189,     0.89289,     0.89389,     0.89489,      0.8959,      0.8969,      0.8979,      0.8989,      0.8999,      0.9009,      0.9019,      0.9029,      0.9039,      0.9049,     0.90591,     0.90691,     0.90791,     0.90891,     0.90991,     0.91091,     0.91191,
               0.91291,     0.91391,     0.91491,     0.91592,     0.91692,     0.91792,     0.91892,     0.91992,     0.92092,     0.92192,     0.92292,     0.92392,     0.92492,     0.92593,     0.92693,     0.92793,     0.92893,     0.92993,     0.93093,     0.93193,     0.93293,     0.93393,     0.93493,     0.93594,
               0.93694,     0.93794,     0.93894,     0.93994,     0.94094,     0.94194,     0.94294,     0.94394,     0.94494,     0.94595,     0.94695,     0.94795,     0.94895,     0.94995,     0.95095,     0.95195,     0.95295,     0.95395,     0.95495,     0.95596,     0.95696,     0.95796,     0.95896,     0.95996,
               0.96096,     0.96196,     0.96296,     0.96396,     0.96496,     0.96597,     0.96697,     0.96797,     0.96897,     0.96997,     0.97097,     0.97197,     0.97297,     0.97397,     0.97497,     0.97598,     0.97698,     0.97798,     0.97898,     0.97998,     0.98098,     0.98198,     0.98298,     0.98398,
               0.98498,     0.98599,     0.98699,     0.98799,     0.98899,     0.98999,     0.99099,     0.99199,     0.99299,     0.99399,     0.99499,       0.996,       0.997,       0.998,       0.999,           1]), array([[    0.98639,     0.98639,     0.98639, ...,           0,           0,           0],
           [    0.99113,     0.99113,     0.98448, ...,           0,           0,           0],
           [    0.99151,     0.99151,     0.98981, ...,           0,           0,           0],
           [    0.98105,     0.98105,     0.97895, ...,           0,           0,           0],
           [    0.99317,     0.99317,     0.99089, ...,           0,           0,           0],
           [    0.99774,     0.99774,     0.99774, ...,           0,           0,           0]]), 'Confidence', 'Recall']]
    fitness: 0.797522748299497
    keys: ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']
    maps: array([    0.70798,     0.69084,     0.75625,      0.7391,     0.88858,     0.87917])
    names: {0: '1-open', 1: '2-short', 2: '3-mousebite', 3: '4-spur', 4: '5-copper', 5: '6-pin-hole'}
    plot: True
    results_dict: {'metrics/precision(B)': 0.96731090792701, 'metrics/recall(B)': 0.9519108350931859, 'metrics/mAP50(B)': 0.9823471218058711, 'metrics/mAP50-95(B)': 0.7769867067987888, 'fitness': 0.797522748299497}
    save_dir: PosixPath('runs/detect/train8')
    speed: {'preprocess': 0.5950196584065754, 'inference': 4.76120842827691, 'loss': 0.015925301445855033, 'postprocess': 2.4289830525716147}
    task: 'detect'
:::
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

::::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":17}" executionInfo="{\"elapsed\":1327,\"status\":\"ok\",\"timestamp\":1701714934674,\"user\":{\"displayName\":\"Diego Borro\",\"userId\":\"13584858323109103783\"},\"user_tz\":-60}" id="EFMoVawQHGU1" outputId="6fd82d9a-fb66-41bc-e6e2-83751ce5a0c4"}
``` python
from google.colab import files

# no funciona con zip: A UTF-8 locale is required. Got ANSI_X3.4-1968
#!zip -r 'train.zip' '/content/runs/detect/train'

shutil.make_archive("train", 'zip', '/content/runs/detect/train')
files.download("train.zip")
```

::: {.output .display_data}
    <IPython.core.display.Javascript object>
:::

::: {.output .display_data}
    <IPython.core.display.Javascript object>
:::
:::::

::: {.cell .markdown id="f9A8cxarQQ8o"}
## Training Evaluation
:::

::: {.cell .code id="AECtvTEH0zLr"}
``` python
RUNS_PATH = '/content/runs/detect/'
```
:::

:::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":618}" executionInfo="{\"elapsed\":2372,\"status\":\"ok\",\"timestamp\":1701714416065,\"user\":{\"displayName\":\"Diego Borro\",\"userId\":\"13584858323109103783\"},\"user_tz\":-60}" id="u-anbRuUQQ8o" outputId="fa46bea5-48fa-4ec5-c4f8-2adf0147a2c9"}
``` python
metrics = cv2.imread(RUNS_PATH+'train/results.png')
metrics = cv2.cvtColor(metrics, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(15,12))
plt.imshow(metrics)
plt.axis('off')
plt.show()
```

::: {.output .display_data}
![](0e7a3b84570ecb860f17e58767f97a178f440e1f.png)
:::
::::

:::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":908}" executionInfo="{\"elapsed\":2074,\"status\":\"ok\",\"timestamp\":1701714484937,\"user\":{\"displayName\":\"Diego Borro\",\"userId\":\"13584858323109103783\"},\"user_tz\":-60}" id="T0_T6WAaQQ8p" outputId="a398539a-559a-48e1-92db-f711f0a66b98"}
``` python
metrics = cv2.imread(RUNS_PATH+'train/confusion_matrix_normalized.png')
metrics = cv2.cvtColor(metrics, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(15,12))
plt.imshow(metrics)
plt.axis('off')
plt.show()
```

::: {.output .display_data}
![](d7f80833d27e0d70adb25ccde7868931f5963c51.png)
:::
::::

::: {.cell .markdown id="ZMhhQHnWQQ8p"}
## Model Inference
:::

:::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\"}" executionInfo="{\"elapsed\":531,\"status\":\"ok\",\"timestamp\":1701714579481,\"user\":{\"displayName\":\"Diego Borro\",\"userId\":\"13584858323109103783\"},\"user_tz\":-60}" id="aUBuhvntQQ8p" outputId="f3288027-d41e-4ad2-d325-62ae0d7ea55d"}
``` python
load_model = YOLO(RUNS_PATH+'train/weights/best.pt')

path_1 = BASE_PATH+'val/images/00041003.jpg'
path_2 = BASE_PATH+'val/images/12100189.jpg'

predict_1 = load_model.predict(path_1)
predict_2 = load_model.predict(path_2)
```

::: {.output .stream .stdout}

    image 1/1 /content/sample_data/val/images/00041003.jpg: 640x640 1 1-open, 2 3-mousebites, 1 5-copper, 2 6-pin-holes, 8.7ms
    Speed: 1.6ms preprocess, 8.7ms inference, 2.2ms postprocess per image at shape (1, 3, 640, 640)

    image 1/1 /content/sample_data/val/images/12100189.jpg: 640x640 1 1-open, 1 2-short, 2 3-mousebites, 2 5-coppers, 1 6-pin-hole, 13.9ms
    Speed: 1.9ms preprocess, 13.9ms inference, 2.3ms postprocess per image at shape (1, 3, 640, 640)
:::
::::

:::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":605}" executionInfo="{\"elapsed\":1466,\"status\":\"ok\",\"timestamp\":1701714605872,\"user\":{\"displayName\":\"Diego Borro\",\"userId\":\"13584858323109103783\"},\"user_tz\":-60}" id="6XKsd6ogQQ8p" outputId="092db2f2-4f80-4f57-9ba6-3ce0964cf4bd"}
``` python
image = cv2.imread(path_1)
h,w,c = image.shape

with open(BASE_PATH+'val/labels/00041003.txt', 'r') as file:
    details = file.readlines()

for detail in details:

    splitted = detail.split()
    xc = int(float(splitted[1])*w)
    yc = int(float(splitted[2])*h)
    wy = int((float(splitted[3])*w)/2)
    hy = int((float(splitted[4])*h)/2)
    label = str(int(splitted[0])+1)

    rectangle_image = cv2.rectangle(image, (xc - wy,yc - hy), (xc + wy,yc + hy), (0,255,0), 2)
    cv2.putText(image, label, (xc+wy, yc+hy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)

fig, axs = plt.subplots(1,2, figsize=(20,20))


axs[0].imshow(rectangle_image)
axs[1].imshow(predict_1[0].plot())
plt.show()
```

::: {.output .display_data}
![](0648ae34aaa341aa870ec387e6828e846280a30c.png)
:::
::::

:::: {.cell .code colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":605}" executionInfo="{\"elapsed\":2148,\"status\":\"ok\",\"timestamp\":1701714624948,\"user\":{\"displayName\":\"Diego Borro\",\"userId\":\"13584858323109103783\"},\"user_tz\":-60}" id="s5JRNsl1QQ8p" outputId="f57ee300-c90e-4d29-a830-ccd0d3736600"}
``` python
image = cv2.imread(path_2)
h,w,c = image.shape

with open(BASE_PATH+'val/labels/12100189.txt', 'r') as file:
    details = file.readlines()

for detail in details:

    splitted = detail.split()
    xc = int(float(splitted[1])*w)
    yc = int(float(splitted[2])*h)
    wy = int((float(splitted[3])*w)/2)
    hy = int((float(splitted[4])*h)/2)
    label = str(int(splitted[0])+1)

    rectangle_image = cv2.rectangle(image, (xc - wy,yc - hy), (xc + wy,yc + hy), (0,255,0), 2)
    cv2.putText(image, label, (xc+wy, yc+hy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)

fig, axs = plt.subplots(1,2, figsize=(20,20))


axs[0].imshow(rectangle_image)
axs[1].imshow(predict_2[0].plot())
plt.show()
```

::: {.output .display_data}
![](58403b585a5e2572befa6942288f47408af5484b.png)
:::
::::

::: {.cell .markdown id="E_rMAnpsIrZc"}
#Assignment

Objetivos:

-   Plotear las diferentes métricas de train que aparecen en la carpeta
    de runs
-   Explicar en una celda de texto cada métrica que se plotee
-   Hacer el test de evaluacón en todas las fotos

Deadline:
:::
