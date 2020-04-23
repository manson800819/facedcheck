from scipy.spatial import distance
from multiprocessing.dummy import Pool
import imutils
from imutils.face_utils import FaceAligner
import time
import cv2
import os
import glob
import numpy as np
import dlib
from keras.models import load_model
import  tensorflow as tf
ready_to_detect_identity = True

predictor_path = "/Users/youqinfeng/face/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)
net = cv2.dnn.readNet('/Users/youqinfeng/face/tiny-yolo-azface-fddb_82000.weights',
                      '/Users/youqinfeng/face/tiny-yolo-azface-fddb.cfg')

# Load Yolo
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
model_path = '/Users/youqinfeng/face/model/facenet_keras.h5'
FRmodel = load_model(model_path)
print(123)
global graph
graph = tf.get_default_graph()

def calc_embs(imgs):
    embs = FRmodel.predict_on_batch(imgs)
    return embs
def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

def img_path_to_encoding(img1, model):
    return img_to_encoding(img1, model)
def img_to_encoding(image, model):
    image = cv2.resize(image, (160, 160))
    img = image[..., ::-1]
    img = np.around(img / 255.0, decimals=12)
    x_train = np.array([img])
    with graph.as_default():

        embedding = model.predict_on_batch(x_train)
    return l2_normalize(embedding)

def prepare_database():
    database = {}

    # load all the images of individuals to recognize into the database
    for file in glob.glob("/Users/youqinfeng/untitled3/face/images/*"):
        identity = os.path.splitext(os.path.basename(file))[0]
        img1 = cv2.imread(file, 1)
        img1 = process_frame1(img1,net)
        database[identity] = img_path_to_encoding(img1, FRmodel)
    return database

def prepare_database1(img):
    img = process_frame1(img, net)
    feature = img_path_to_encoding(img, FRmodel)
    return feature

def webcam_face_recognizer(database,frame):
    """
    Runs a loop that extracts images from the computer's webcam and determines whether or not
    it contains the face of a person in our database.

    If it contains a face, an audio message will be played welcoming the user.
    If not, the program will process the next frame from the webcam
    """
    global ready_to_detect_identity
    # print("start scan")
    img = frame
    img,name = process_frame(img, frame,net, database, FRmodel)
    return img,name
def process_frame1(img,net=net):
    """
    Determine whether the current frame contains the faces of people from our database
    """
    img = imutils.resize(img, width=800)
    global ready_to_detect_identity
    def detecting_one_image(net, output_layers, img):
        # Detecting objects
        # cv::dnn::blobFromImage (InputArray image, double scalefactor=1.0, const Size &size=Size(), const Scalar &mean=Scalar(), bool swapRB=false, bool crop=false, int ddepth=CV_32F)
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
        return outs
    # Load Yolo
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    outs = detecting_one_image(net, output_layers, img)
    height, width, channels = img.shape
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.6:
                fa = FaceAligner(predictor)
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3]* height)
                # Rectangle coordinates
                x = int(center_x - w / 2) if int(center_x - w / 2)>=0 else 0
                y = int(center_y - h / 2) if int(center_y - h / 2)>=0 else 0
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faceAligned = fa.align(img, gray, dlib.rectangle(x, y, x + w, y + h))


                return faceAligned

def process_frame(img, frame,net, database, FRmodel):
    """
    Determine whether the current frame contains the faces of people from our database
    """
    identity=None
    img = imutils.resize(img, width=800)
    frame = imutils.resize(frame, width=800)
    global ready_to_detect_identity
    def detecting_one_image(net, output_layers, img):
        # Detecting objects
        # cv::dnn::blobFromImage (InputArray image, double scalefactor=1.0, const Size &size=Size(), const Scalar &mean=Scalar(), bool swapRB=false, bool crop=false, int ddepth=CV_32F)
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
        return outs
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    outs = detecting_one_image(net, output_layers, img)
    height, width, channels = img.shape
    identities = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            # print(scores)
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.8:
                fa = FaceAligner(predictor)
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2) if int(center_x - w / 2)>=0 else 0
                y = int(center_y - h / 2) if int(center_y - h / 2)>=0 else 0
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faceAligned = fa.align(img, gray, dlib.rectangle(x, y, x + w, y + h))
                img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0), 2)
                identity = find_identity(faceAligned, database, FRmodel)
                img  = cv2.putText(img, identity, (x, y + h - 5), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 1)
                if identity is not None:
                    identities.append(identity)
    return img,identity
def find_identity(frame, database, FRmodel):

    part_image = frame

    return who_is_it(part_image, database, FRmodel)

def who_is_it(image, database, model):
    """
    Implements face recognition for the happy house by finding who is the person on the image_path image.
    
    Arguments:
    image_path -- path to an image
    database -- database containing image encodings along with the name of the person on the image
    model -- your Inception model instance in Keras
    
    Returns:
    min_dist -- the minimum distance between image_path encoding and the encodings from the database
    identity -- string, the name prediction for the person on image_path
    """
    # print(image.shape)
    encoding = img_to_encoding(image, model)
    
    min_dist = 100
    identity = None
    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():
        
        # Compute L2 distance between the target "encoding" and the current "emb" from the database.
        dist = distance.euclidean(db_enc,encoding)

        # print('distance for %s is %s' %(name, dist))

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name
        if dist < min_dist:
            min_dist = dist
            identity = name
    
    if min_dist > 0.8:
        return None
    else:
        return str(identity)

def welcome_users(identities):
    """ Outputs a welcome audio message to the users """
    global ready_to_detect_identity
    welcome_message = 'Welcome '

    if len(identities) == 1:
        welcome_message += '%s, have a nice day.' % identities[0]
    else:
        for identity_id in range(len(identities)-1):
            welcome_message += '%s, ' % identities[identity_id]
        welcome_message += 'and %s, ' % identities[-1]
        welcome_message += 'have a nice day!'
    # windows10_voice_interface.Speak(welcome_message)
    # Allow the program to start detecting identities again
    ready_to_detect_identity = True

# if __name__ == "__main__":
#     database = prepare_database()
#     webcam_face_recognizer(database)





# ### References:
# 
# - Florian Schroff, Dmitry Kalenichenko, James Philbin (2015). [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/pdf/1503.03832.pdf)
# - Yaniv Taigman, Ming Yang, Marc'Aurelio Ranzato, Lior Wolf (2014). [DeepFace: Closing the gap to human-level performance in face verification](https://research.fb.com/wp-content/uploads/2016/11/deepface-closing-the-gap-to-human-level-performance-in-face-verification.pdf) 
# - The pretrained model we use is inspired by Victor Sy Wang's implementation and was loaded using his code: https://github.com/iwantooxxoox/Keras-OpenFace.
# - Our implementation also took a lot of inspiration from the official FaceNet github repository: https://github.com/davidsandberg/facenet 
# 
