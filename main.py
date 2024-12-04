import cv2
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 1. Cargar el conjunto de datos Iris
iris = load_iris()
X = iris.data  # Las características (longitud y anchura de sépalos y pétalos)
y = iris.target  # Las etiquetas (especies de flores)

# 2. Dividir el conjunto de datos en entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Crear y entrenar el modelo de árbol de decisión
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# 4. Función para extraer características básicas de la imagen de la flor
def extract_flower_features(image_path):
    # Cargar la imagen de la flor
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir de BGR a RGB

    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Usar un umbral para segmentar los pétalos y sépalos
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Encontrar los contornos en la imagen
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrar los contornos que corresponden a las flores
    flower_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]

    # Suponiendo que el pétalo más grande es el que queremos medir
    max_contour = max(flower_contours, key=cv2.contourArea)

    # Obtener el bounding box del contorno máximo (que representa el pétalo principal)
    x, y, w, h = cv2.boundingRect(max_contour)

    # Calcular características como la longitud y el ancho de los pétalos (a partir del bounding box)
    petal_length = w / 10.0  # Ajuste de escala para longitud
    petal_width = h / 10.0  # Ajuste de escala para anchura

    # Características simuladas de sépalo para fines de prueba
    sepal_length = 5.0  # Esto debería ser extraído también
    sepal_width = 3.0   # Esto debería ser extraído también

    return np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# 5. Función para clasificar la flor basada en la imagen subida
def classify_flower(image_path):
    # Extraer características de la imagen
    features = extract_flower_features(image_path)

    # Realizar la predicción
    prediction = clf.predict(features)
    species = iris.target_names[prediction][0]
    return species, features[0]

# 6. Función para mostrar la imagen y su clasificación
def display_flower_with_classification(image_path):
    # Clasificar la flor y obtener sus características
    species, _ = classify_flower(image_path)

    # Mostrar la imagen
    img = mpimg.imread(image_path)
    
    # Crear la figura
    plt.figure(figsize=(8, 8))

    # Mostrar la imagen a la izquierda
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis('off')  # Ocultar los ejes
    plt.title(f'Flor: {species}', fontsize=16)  # Título con la especie predicha

    # Mostrar el gráfico
    plt.show()

# 7. Ejemplo de uso
image_path = 'Setosa.jpg'  # Reemplaza con la ruta de tu imagen

# Mostrar la imagen y la clasificación
display_flower_with_classification(image_path)
