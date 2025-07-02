import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import time

# Defina aqui o caminho base onde estão os arquivos (ajuste conforme sua pasta)
base_path = r"C:\Users\igoro\OneDrive\Área de Trabalho\IA\yolo-coco-data"

# Listar arquivos da pasta base (opcional, só pra conferir)
for dirname, _, filenames in os.walk(base_path):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Carregar nomes das classes
names_path = os.path.join(base_path, "coco.names")
with open(names_path) as f:
    names = f.read().strip().split("\n")
print(names)
print("Total de classes:", len(names))

# Caminhos do modelo
weights_path = os.path.join(base_path, "yolov3.weights")
configuration_path = os.path.join(base_path, "yolov3.cfg")

# Parâmetros
pro_min = 0.5
threshold = 0.3

# Carregar rede
net = cv2.dnn.readNetFromDarknet(configuration_path, weights_path)

# Camadas de saída
layers = net.getLayerNames()
output_layers = [layers[i[0] - 1] for i in net.getUnconnectedOutLayers()]
print("Output layers:", output_layers)

# Carregar imagem (coloque aqui o caminho da sua imagem)
image_path = r"C:\Users\igoro\OneDrive\Área de Trabalho\IA\imager1\image4.jpg"
image = cv2.imread(image_path)
if image is None:
    raise ValueError("Erro ao carregar imagem.")
Height, Width = image.shape[:2]

# Exibir imagem original
plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Imagem Original")
plt.axis("off")
plt.show()

# Criar blob
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
print("Blob shape:", blob.shape)

# Visualizar blob
blob_to_show = blob[0].transpose(1, 2, 0)
plt.figure(figsize=(5, 5))
plt.imshow(blob_to_show)
plt.title("Blob da Imagem")
plt.axis("off")
plt.show()

# Inference
net.setInput(blob)
start = time.time()
outputs = net.forward(output_layers)
end = time.time()
print(f"YOLO levou {end - start:.2f} segundos")

# Processar saídas
classes = []
confidences = []
boxes = []

for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > pro_min:
            center_x, center_y, width, height = (detection[0:4] * np.array([Width, Height, Width, Height])).astype('int')
            x = int(center_x - width / 2)
            y = int(center_y - height / 2)

            boxes.append([x, y, width, height])
            confidences.append(float(confidence))
            classes.append(class_id)

# Non-Maximum Suppression
indices = cv2.dnn.NMSBoxes(boxes, confidences, pro_min, threshold)
colours = np.random.randint(0, 255, size=(len(names), 3), dtype='uint8')

# Desenhar caixas
for i in indices.flatten():
    x, y, w, h = boxes[i]
    color = [int(c) for c in colours[classes[i]]]

    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    label = f"{names[classes[i]]}: {confidences[i]:.2f}"
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Mostrar imagem final
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Imagem com Detecções")
plt.axis("off")
plt.show()
