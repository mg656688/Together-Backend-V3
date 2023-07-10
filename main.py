import os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from flask import Flask, jsonify, request, render_template
from modules.database import collection as db
from datetime import datetime
import pickle
import torchvision.transforms as transforms
import torch
from PIL import Image
import torch.nn as nn
from modules.models_architcture import Flowers, Cotton, Leaves
import tensorflow as tf
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

# Load Paddy ResNet34 Model
model_paddy = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
model_paddy.fc = nn.Linear(512, 10)

# Load the saved model state dictionary
model_paddy.load_state_dict(torch.load('modules/backed/paddy.pt', map_location=torch.device('cpu')))

# Set the model in evaluation mode
model_paddy.eval()

# Load Flowers ResNet50 Model

# Load the model checkpoint
model_path = 'modules/backed/flowers.pt'
flower_model = torch.load(model_path)

# Create an instance of the model
model_flower = Flowers.ResNet50Model()

# Load the state dict of the model
model_flower.load_state_dict(flower_model)

# Set the model to evaluation mode
model_flower.eval()

# Load Cotton ResNet50 Model
cotton_model_path = 'modules/backed/Cotton_resnet50_checkpoint.pth'
# Load the saved checkpoint
checkpoint = torch.load(cotton_model_path)

# Create an instance of the ResNet-50 model
model_cotton = Cotton.ResNet50Model()

# Load the model's state_dict from the checkpoint
model_cotton.load_state_dict(checkpoint)

# Set the model to evaluation mode
model_cotton.eval()

# Load the Seedlings EfficientNetB2 model
model_seedlings = load_model('modules/backed/EfficientNetB2Seed.h5')

# Load the RF Crop Recommendation model
Random_Forest_crop_model = 'modules/backed/CropRandomForest.pkl'
with open(Random_Forest_crop_model, 'rb') as f:
    pickle_model_crop = pickle.load(f)

# # Load the LDA Leaves model
# LDA_leaves_model = 'modules/backed/LDA_leaf.pkl'
# with open(LDA_leaves_model, 'rb') as f:
#     pickle_model_leaves = pickle.load(f)


tflite_model_file_leaves = 'modules/backed/leaf.tflite'
with open(tflite_model_file_leaves, 'rb') as fid:
    tflite_model_leaves = fid.read()


target_img = os.path.join(os.getcwd(), 'modules/backed/static/images')


# Function to load and prepare the images in the right shape
def read_image_seed(filename, size):
    img = load_img(filename, target_size=size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def read_image_weed(filename):
    img = Image.open(filename).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    x = transform(img)
    x = x.unsqueeze(0)
    return x


def read_image_flower(filename):
    img = Image.open(filename).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to (224, 224)
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    x = transform(img)
    x = x.unsqueeze(0)
    return x


def read_image_leave(filename, size):
    img = load_img(filename, target_size=size, color_mode="grayscale")
     # Preprocess the image
    img = img.resize((192, 1))  # Resize the image to 192x1 pixels
    img = np.asarray(img)  # Convert the image to a numpy array
    img = img.astype('float32')  # Convert the data type to float32
    img = (img - 127.5) / 127.5  # Normalize the image
    x = preprocess_input(img)
    return x



def transform_image(image_bytes):
    # Define the transformations
    my_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Open the image using PIL
    img = Image.open(image_bytes)

    # Apply the transformations
    preprocessed_image = my_transforms(img)

    # Add a batch dimension to match the model's input shape
    preprocessed_image = preprocessed_image.unsqueeze(0)

    return preprocessed_image


def calculate_npk_requirements(nitrogen, phosphorus, potassium, acres_num, source):
    global req_n_final, req_p_final, req_k_final
    nitrogen_readings = nitrogen
    phosphorus_readings = phosphorus
    potassium_readings = potassium
    acres = acres_num
    nitrogen_source = source

    # Set the ideal nutrient levels and acceptable ranges
    I_N = 50  # Ideal nitrogen levels (ppm)
    I_P = 50  # Ideal phosphorus levels (ppm)
    I_K = 40  # Ideal potassium levels (ppm)
    range_min = 0.90
    range_max = 1.00

    # Calculate input nutrient percentages
    input_n_percentage = nitrogen_readings / I_N
    input_p_percentage = phosphorus_readings / I_P
    input_k_percentage = potassium_readings / I_K

    # Check if nitrogen levels are within the acceptable range
    if range_min <= input_n_percentage <= range_max:
        print("Your nitrogen levels are good, no need for additional nitrogen.")
    else:
        # Calculate nitrogen shortage percentage
        n_shortage_percentage = (I_N - nitrogen_readings) / I_N * 100

        # Calculate nitrogen required in pounds
        req_n = n_shortage_percentage / 100 * acres

        # Check if the calculated nitrogen requirement is negative
        if req_n < 0:
            req_n = 0

        # Calculate the final required nitrogen amount based on the selected source
        if nitrogen_source == "Urea":
            req_n_final = req_n * 2
            print(f"Your land requires the addition of {req_n_final} pounds of Urea.")
        elif nitrogen_source == "Compost":
            req_n_final = req_n
            print(f"Your land requires the addition of {req_n_final} pounds of Compost.")
        else:
            print("Invalid nitrogen source selection. Please choose either Urea or Compost.")

    # Check if phosphorus levels are within the acceptable range
    if range_min <= input_p_percentage <= range_max:
        print("Your phosphorus levels are good, no need for additional phosphorus.")
    else:
        # Calculate phosphorus shortage percentage
        p_shortage_percentage = (I_P - phosphorus_readings) / I_P * 100

        # Calculate phosphorus required in pounds
        req_p = p_shortage_percentage / 100 * acres

        # Check if the calculated phosphorus requirement is negative
        if req_p < 0:
            req_p = 0

        # Calculate the final required phosphorus amount
        req_p_final = req_p * 2
        print(f"Your land requires the addition of {req_p_final} pounds of Super Phosphate.")

    # Check if potassium levels are within the acceptable range
    if range_min <= input_k_percentage <= range_max:
        print("Your potassium levels are good, no need for additional potassium.")
    else:
        # Calculate potassium shortage percentage
        k_shortage_percentage = (I_K - potassium_readings) / I_K * 100

        # Calculate potassium required in pounds
        req_k = k_shortage_percentage / 100 * acres

        # Check if the calculated potassium requirement is negative
        if req_k < 0:
            req_k = 0

        # Calculate the final required potassium amount
        req_k_final = req_k * 2
        print(f"Your land requires the addition of {req_k_final} pounds of Potassium Sulphate.")

    return req_n_final, req_p_final, req_k_final


# Allow files with extension png, jpg and jpeg
ALLOWED_EXT = {'jpg', 'jpeg', 'png'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT


@app.route('/seedlings', methods=['POST'])
def predict_seedlings():
    global seed
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):  # Checking file format
            filename = file.filename
            file_path = os.path.join('modules/backed/static/images/seedlings', filename)
            file.save(file_path)
            img_size = (260, 260)
            img = read_image_seed(file_path, img_size)  # preprocessing method

            # Perform inference using the model
            class_prediction = model_seedlings.predict(img)
            class_x = np.argmax(class_prediction)
            print(class_prediction)
            print(class_x)

            # Define a dictionary mapping class indices to seed names
            class_to_seed = {
                0: "Scentless Mayweed",
                1: "Common wheat",
                2: "Charlock",
                3: "Black grass",
                4: "Sugar beet",
                5: "Loose Silky-bent",
                6: "Maize",
                7: "Cleavers",
                8: "Common Chickweed",
                9: "Fat Hen",
                10: "Small-flowered Cranesbill",
                11: "Shepherd’s Purse"
            }

            # Get the seed name based on the class index
            seed = class_to_seed[class_x]

            db.addSeedlingImage(
                file.filename,
                seed,
                datetime.now(),
                target_img + '/seedlings' + file.filename)

            return jsonify(prediction=seed, user_image=file_path)
        else:
            return "Unable to read the file. Please check file extension"


@app.route('/flowers', methods=['POST'])
def predict_flowers():
    global flower
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):  # Checking file format
            filename = file.filename
            file_path = os.path.join('modules/backed/static/images/flowers', filename)
            file.save(file_path)
            img = read_image_flower(file_path)  # preprocessing method

            # Perform inference using the model
            with torch.no_grad():
                output = model_flower(img)
                _, predicted = torch.max(output, 1)
                class_x = predicted.item()

            flower_mapping = {
                0: "Astilbe",
                1: "Bell flower",
                2: "Black Eyed Susan",
                3: "Calendula",
                4: "California Poppy",
                5: "Carnation",
                6: "Common Daisy",
                7: "Coreopsis",
                8: "Daffodil",
                9: "Dandelion",
                10: "Iris",
                11: "Magnolia",
                12: "Rose",
                13: "Sunflower",
                14: "Tulip",
                15: "Water Lily"
            }

            flower = flower_mapping.get(class_x, "Unknown Flower")

            db.addFlowerImage(
                file.filename,
                flower,
                datetime.now(),
                target_img + '/flowers' + file.filename)

            return jsonify(prediction=flower, user_image=file_path)
        else:
            return "Unable to read the file. Please check file extension"


@app.route('/leaves', methods=['POST'])
def predict_leaves():
    global leaf
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):  # Checking file format
            filename = file.filename
            file_path = os.path.join('modules/backed/static/images/leaves', filename)
            file.save(file_path)

             # Reshape the image to the input shape of the model
            input_shape = (1, 192)  # Assuming 1-dimensional input
            img = read_image_leave(file_path, input_shape)

            # Load the TFLite model and allocate tensors
            interpreter = tf.lite.Interpreter(model_content=tflite_model_leaves)
            interpreter.allocate_tensors()

            # Get input and output tensors
            input_details = interpreter.get_input_details()[0]['index']
            output_details = interpreter.get_output_details()[0]['index']
            class_prediction = []

            interpreter.set_tensor(input_details, img)
            interpreter.invoke()
            class_prediction.append(interpreter.get_tensor(output_details))

            class_x = np.argmax(class_prediction)


            leaf_mapping = {
                0: 'Acer Capillipes',
                1: 'Acer Circinatum',
                2: 'Acer Mono',
                3: 'Acer Opalus',
                4: 'Acer Palmatum',
                5: 'Acer Pictum',
                6: 'Acer Platanoids',
                7: 'Acer Rubrum',
                8: 'Acer Rufinerve',
                9: 'Acer Saccharinum',
                10: 'Alnus Cordata',
                11: 'Alnus Maximowiczii',
                12: 'Alnus Rubra',
                13: 'Alnus Sieboldiana',
                14: 'Alnus Viridis',
                15: 'Arundinaria Simonii',
                16: 'Betula Austrosinensis',
                17: 'Betula Pendula',
                18: 'Callicarpa Bodinieri',
                19: 'Castanea Sativa',
                20: 'Celtis Koraiensis',
                21: 'Cercis Siliquastrum',
                22: 'Cornus Chinensis',
                23: 'Cornus Controversa',
                24: 'Cornus Macrophylla',
                25: 'Cotinus Coggygria',
                26: 'Crataegus Monogyna',
                27: 'Cytisus Battandieri',
                28: 'Eucalyptus Glaucescens',
                29: 'Eucalyptus Neglecta',
                30: 'Eucalyptus Urnigera',
                31: 'Fagus Sylvatica',
                32: 'Ginkgo Biloba',
                33: 'Ilex Aquifolium',
                34: 'Ilex Cornuta',
                35: 'Liquidambar Styraciflua',
                36: 'Liriodendron Tulipifera',
                37: 'Lithocarpus Cleistocarpus',
                38: 'Lithocarpus Edulis',
                39: 'Magnolia Heptapeta',
                40: 'Magnolia Salicifolia',
                41: 'MorusNigra',
                42: 'Olea Europaea',
                43: 'Phildelphus',
                44: 'Populus Adenopoda',
                45: 'Populus Grandidentata',
                46: 'Populus Nigra',
                47: 'Prunus Avium',
                48: 'Prunus x Shmittii',
                49: 'Pterocarya Stenoptera',
                50: 'Quercus Afares',
                51: 'Quercus Agrifolia',
                52: 'Quercus Alnifolia',
                53: 'Quercus Brantii',
                54: 'Quercus Canariensis',
                55: 'Quercus Castaneifolia',
                56: 'Quercus Cerris',
                57: 'Quercus Chrysolepis',
                58: 'Quercus Coccifera',
                59: 'Quercus Coccinea',
                60: 'Quercus Crassifolia',
                61: 'Quercus Ellipsoidalis',
                62: 'Quercus Gregorii',
                63: 'Quercus Hartwissiana',
                64: 'Quercus Ilex',
                65: 'Quercus Imbricaria',
                66: 'Quercus Infectoria sub',
                67: 'Quercus Kewensis',
                68: 'Quercus Nigra',
                69: 'Quercus Palustris',
                70: 'Quercus Petraca',
                71: 'Quercus Phellos',
                72: 'Quercus Pubescens',
                73: 'Quercus Pyrenaica',
                74: 'Quercus Rhysophylla',
                75: 'Quercus Robur',
                76: 'Quercus Rubra',
                77: 'Quercus Semecarpifolia',
                78: 'Quercus Shumardii',
                79: 'Quercus Sigillata',
                80: 'Quercus Suber',
                81: 'Quercus Texana',
                82: 'Quercus Trojana',
                83: 'Quercus Variabilis',
                84: 'Quercus Virginiana',
                85: 'Quercus x Hispanica',
                86: 'Quercus x Turneri',
                87: 'Rhododendron x Rostatum',
                88: 'Salix Fragilis',
                89: 'Salix Interior',
                90: 'Sorbus Aria',
                91: 'Tilia Tomentosa',
                92: 'Ulmus Minor',
                93: 'Viburnum Tinus',
                94: 'Viburnum x Rhytidophylloides',
                95: 'Zelkova Serrata',
                96: 'Betula Pubescens',
                97: 'Quercus x Tlittiae',
                98: 'Quercus Imbricaria'
            }

            leaf = leaf_mapping[class_x]
            print(leaf)

            db.addLeaveImage(
                file.filename,
                leaf,
                datetime.now(),
                target_img + '/leaves' + file.filename)

            return jsonify(prediction=leaf, user_image=file_path)
        else:
            return "Unable to read the file. Please check file extension"


@app.route('/paddy', methods=['POST'])
def predict_paddy():
    global disease
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):  # Checking file format
            filename = file.filename
            file_path = os.path.join('modules/backed/static/images/paddy', filename)
            file.save(file_path)

            # preprocessing method
            img = transform_image(file)

            # Perform inference using the model
            with torch.no_grad():
                output = model_paddy(img)
                _, predicted = torch.max(output, 1)
                class_x = predicted.item()

            disease_mapping = {
                0: "Bacterial Leaf Blight",
                1: "Bacterial Leaf Streak",
                2: "Bacterial Panicle Blight",
                3: "Blast",
                4: "Brown Spot",
                5: "Dead Heart",
                6: "Downy Mildew",
                7: "Hispa",
                8: "Normal",
                9: "Tungro"
            }

            disease = disease_mapping.get(class_x, "Unknown Disease")

            db.addPaddyImage(
                file.filename,
                disease,
                datetime.now(),
                target_img + '/paddy' + file.filename)

            return jsonify(prediction=disease, user_image=file_path)
        else:
            return "Unable to read the file. Please check file extension"


@app.route('/weeds', methods=['POST'])
def predict_weeds():
    global weed
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):  # Checking file format
            filename = file.filename
            file_path = os.path.join('modules/backed/static/images/weed', filename)
            file.save(file_path)
            img = read_image_weed(file_path)  # preprocessing method

            # Perform inference using the model
            with torch.no_grad():
                output = model_cotton(img)
                _, predicted = torch.max(output, 1)
                class_x = predicted.item()

            weed_mapping = {
                0: "Nutsedge",
                1: "Sicklepod",
                2: "Morningglory",
                3: "Ragweed",
                4: "Palmer Amaranth",
                5: "Waterhemp",
                6: "Crabgrass",
                7: "Swinecress",
                8: "Prickly Sida",
                9: "Carpet weeds",
                10: "Spotted Spurge",
                11: "SpurredAnoda",
                12: "Eclipta",
                13: "Goosegrass",
                14: "Purslane"
            }

            weed = weed_mapping.get(class_x, "Unknown Weed")

            db.addWeedImage(
                file.filename,
                weed,
                datetime.now(),
                target_img + '/weed' + file.filename)

            return jsonify(prediction=weed, user_image=file_path)
        else:
            return "Unable to read the file. Please check file extension"


@app.route('/crop', methods=['POST'])
def predict_crop():
    global classes_x, crop, reqN, reqP, reqK, r_N
    if request.method == 'POST':
        data = request.get_json()
        print(data)
        N = data['nitrogen']
        P = data['phosphorus']
        K = data['potassium']
        temp = data['temp']
        humidity = data['humidity']
        Ph = data['ph']
        rainFall = data['rainFall']
        acres_number = data['acres']
        source = data['source']

        query = np.array([N, P, K, temp, humidity, Ph, rainFall], dtype=np.float32)
        input_data = query.reshape(1, 7)

        # Make predictions using the loaded model
        class_prediction = pickle_model_crop.predict(input_data)
        classes_x = np.argmax(class_prediction)

        reqN, reqP, reqK = calculate_npk_requirements(N, P, K, acres_number, source)

    if classes_x == 0:
        crop = 'Apple'
    elif classes_x == 1:
        crop = 'Banana'
    elif classes_x == 2:
        crop = 'Blackgram'
    elif classes_x == 3:
        crop = 'Chickpea'
    elif classes_x == 4:
        crop = 'Coconut'
    elif classes_x == 5:
        crop = 'Coffee'
    elif classes_x == 6:
        crop = 'Cotton'
    elif classes_x == 7:
        crop = 'Grapes'
    elif classes_x == 8:
        crop = 'jute'
    elif classes_x == 9:
        crop = 'Kidneybeans'
    elif classes_x == 10:
        crop = 'Lentil'
    elif classes_x == 11:
        crop = 'Maize'
    elif classes_x == 12:
        crop = 'Mango'
    elif classes_x == 13:
        crop = 'Mothbeans'
    elif classes_x == 14:
        crop = 'Mungbean'
    elif classes_x == 15:
        crop = 'Muskmelon'
    elif classes_x == 16:
        crop = 'Orange'
    elif classes_x == 17:
        crop = 'Papaya'
    elif classes_x == 18:
        crop = 'Pigeonpeas'
    elif classes_x == 19:
        crop = 'Pomegranate'
    elif classes_x == 20:
        crop = 'Rice'
    elif classes_x == 21:
        crop = 'Watermelon'

    db.cropRecommendation(
        crop,
        datetime.now())

    return jsonify(prediction=crop, req_N=reqN, req_P=reqP, req_K=reqK)


@app.route('/report', methods=['POST'])
def report():
    location = request.json['location']
    pollutionImage = request.json['file']
    db.pollutionReport(
        pollutionImage,
        datetime.now(),
        location)
    return jsonify(location=location)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=9874)
