import ArcFace
import VGGFace
import matplotlib.pyplot as plt
from PIL import Image
from deepface.commons import distance as dst
from deepface.commons import functions
from deepface import DeepFace
import numpy as np 
import json, codecs
from os import listdir
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image



model_A = ArcFace.loadModel()
model_A.load_weights("./weights/arcface_weights.h5") 

model_V,vgg_face_descriptor = VGGFace.loadVggface()

arcF=''
vggF = ''

# ESCREVER ARQUIVO JSON 
def escrever_json(dados):
   dados_l = dados
   json.dump(dados_l, codecs.open("caracteristicas.json", 'w', encoding='utf-8'), 
          separators=(',', ':'), 
          sort_keys=True, 
          indent=4)

# PREPOSCESS VGG
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))#224
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


# ARCFACE 
def extractorArc(img1_path , arcF):
   hasface = False
   img1 = functions.preprocess_face(img1_path, target_size = (112, 112),enforce_detection=False)
   
   npA = np.all(img1)
   if bool(npA):
      hasface = True
      arcF = model_A.predict(img1)[0]

   return (hasface , arcF )


#  VGG
def extractorVGG(img1_path,vggF):
   hasface = True
   img1_pre = preprocess_image(img1_path)
   # img1_representation = vgg_face_descriptor.predict(img1_pre)[0,:]
   vggF = vgg_face_descriptor.predict(img1_pre)[0,:]

   return (hasface, vggF)

all_info =[] 
can_insert = False
employee_pictures ="./aligned_images"
employees = dict()
aux=0

for file in listdir(employee_pictures):
   employee, extension = file.split(".")
   print('\n',employee, extension,"\n")
   if extension =='png' or extension =='jpg':
      verif1,arcF= extractorArc('aligned_images\%s.%s' %(employee,extension),arcF)
      
      print("Encontrou face: ",verif1)
      verif2,vggF = extractorVGG('aligned_images\%s.%s' %(employee,extension) ,vggF)
   
   if verif1 and verif2:
      aux=aux+1
      all_info.append({'arc':arcF.tolist() , 'vgg':vggF.tolist()})


print("employee representations retrieved successfully\n")
print(aux," imagens encontraram rosto")

escrever_json(all_info)
plt.show(block=True)