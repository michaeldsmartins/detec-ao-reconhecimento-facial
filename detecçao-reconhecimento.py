!pip install deepface

import cv2
import matplotlib.pyplot as plt
from google.colab import files
from deepface import DeepFace
import numpy as np


face_name = input("Digite o nome da face que deseja reconhecer: ")


print("Faça o upload da imagem de referência para detectar a face:")
uploaded = files.upload()
image_path = list(uploaded.keys())[0]

reference_image = cv2.imread(image_path)
reference_image_rgb = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


reference_image_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(reference_image_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

if len(faces) == 0:
    print("Nenhuma face detectada na imagem de referência. Tente novamente com outra imagem.")
else:
    print(f"Face detectada e associada ao nome: {face_name}")


    for (x, y, w, h) in faces:
        cv2.rectangle(reference_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(reference_image, face_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    plt.figure(figsize=(10, 10))
    plt.imshow(reference_image_rgb)
    plt.axis('off')
    plt.title('Detecção Facial')
    plt.show()

    
    print("Faça o upload da imagem para reconhecimento:")
    uploaded = files.upload()
    recognition_image_path = list(uploaded.keys())[0]

   
    recognition_image = cv2.imread(recognition_image_path)
    recognition_image_rgb = cv2.cvtColor(recognition_image, cv2.COLOR_BGR2RGB)

  
    recognition_image_gray = cv2.cvtColor(recognition_image, cv2.COLOR_BGR2GRAY)
    recognition_faces = face_cascade.detectMultiScale(recognition_image_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(recognition_faces) == 0:
        print("Nenhuma face detectada na imagem de reconhecimento.")
    else:
       
        for (x, y, w, h) in recognition_faces:
            recognition_face = recognition_image[y:y+h, x:x+w]
            
         
            cv2.imwrite('recognized_face.jpg', recognition_face)
            
            try:
                result = DeepFace.verify('recognized_face.jpg', image_path)
                if result['verified']:
                    cv2.putText(recognition_image, face_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(recognition_image, "Desconhecido", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            except Exception as e:
                print(f"Erro ao comparar as faces: {e}")

            cv2.rectangle(recognition_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(recognition_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(' Reconhecimento  Facial')
        plt.show()
