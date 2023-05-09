import tensorflow as tf
import numpy as np
import cv2
import dlib
import os
import argparse
def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)
def detect_face(image_paths, save_detected_at, default_max_size=800, size=300, padding=0.25):
    cnn_face_detector = dlib.cnn_face_detection_model_v1('dlib_models/mmod_human_face_detector.dat')
    sp = dlib.shape_predictor('dlib_models/shape_predictor_5_face_landmarks.dat')
    base = 2000

    for index, image_path in enumerate(image_paths):
        if index % 1000 == 0:
            print('---%d/%d---' % (index, len(image_paths)))
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        old_height, old_width, _ = img.shape

        if old_width > old_height:
            new_width, new_height = default_max_size, int(default_max_size * old_height / old_width)
        else:
            new_width, new_height = int(default_max_size * old_width / old_height), default_max_size
        img = cv2.resize(img, (new_width, new_height))

        dets = cnn_face_detector(img, 1)
        num_faces = len(dets)

        if num_faces == 0:
            print("Sorry, there were no faces found in '{}'".format(image_path))
            continue

        faces = dlib.full_object_detections()

        for detection in dets:
            rect = detection.rect
            faces.append(sp(img, rect))

        images = dlib.get_face_chips(img, faces, size=size, padding=padding)

        for idx, image in enumerate(images):
            img_name = image_path.split("/")[-1]
            path_sp = img_name.split(".")
            face_name = os.path.join(save_detected_at, path_sp[0] + "_" + "face" + str(idx) + "." + path_sp[-1])
            cv2.imwrite(face_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import dlib

def predict_age_gender_race(save_prediction_at, imgs_path='cropped_faces/'):
    img_names = [os.path.join(imgs_path, x) for x in os.listdir(imgs_path)]
    device = 'gpu' if tf.test.is_gpu_available() else 'cpu'

    model_fair_7 = tf.keras.applications.ResNet50(
        include_top=True,
        weights=None,
        input_shape=(224, 224, 3),
        classes=18
    )
    model_fair_7.load_weights('/home/kalpit/Downloads/FairFace-master/res34_fair_align_multi_7_20190809.h5')

    model_fair_4 = tf.keras.applications.ResNet50(
        include_top=True,
        weights=None,
        input_shape=(224, 224, 3),
        classes=4
    )
    model_fair_4.load_weights('/home/kalpit/Downloads/FairFace-master/res34_fair_align_multi_4_20190809.h5')

    trans = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Resizing(224, 224),
        tf.keras.layers.experimental.preprocessing.Normalization(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    face_names = []
    race_scores_fair = []
    gender_scores_fair = []
    age_scores_fair = []
    race_preds_fair = []
    gender_preds_fair = []
    age_preds_fair = []
    race_scores_fair_4 = []
    race_preds_fair_4 = []

    for index, img_name in enumerate(img_names):
        if index % 1000 == 0:
            print("Predicting... {}/{}".format(index, len(img_names)))

        face_names.append(img_name)
        image = dlib.load_rgb_image(img_name)
        image = trans(image)
        image = np.expand_dims(image, axis=0)

        # fair
        outputs = model_fair_7(image)
        outputs = tf.squeeze(outputs)
        outputs = outputs.numpy()

        race_outputs = outputs[:7]
        gender_outputs = outputs[7:9]
        age_outputs = outputs[9:18]

        race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
        gender_score = np.exp(gender_outputs) / np.sum(np.exp(gender_outputs))
        age_score = np.exp(age_outputs) / np.sum(np.exp(age_outputs))

        race_pred = np.argmax(race_score)
        gender_pred = np.argmax(gender_score)
        age_pred = np.argmax(age_score)

        race_scores_fair.append(race_score)
        gender_scores_fair.append(gender_score)
        age_scores_fair.append(age_score)

        race_preds_fair.append(race_pred)
        gender_preds_fair.append(gender_pred)
        age_preds_fair.append(age_pred)

        # fair 4 class
        outputs = model_fair_4(image)
        outputs = tf.squeeze(outputs)
        outputs = outputs.numpy()

        race_outputs = outputs[:4]
        race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
        race_pred = np.argmax(race_score)

        race_scores_fair_4.append(race_score)
        race_preds_fair_4.append(race_pred)

    result = pd.DataFrame([face_names,
                           race_preds_fair,
                           race_preds_fair_4,
                           gender_preds_fair,
                           age_preds_fair,
                           race_scores_fair, race_scores_fair_4,
                           gender_scores_fair,
                           age_scores_fair, ]).T
    result.columns = ['face_name_align',
                      'race_preds_fair',
                      'race_preds_fair_4',
                      'gender_preds_fair',
                      'age_preds_fair',
                      'race_scores_fair',
                      'race_scores_fair_4',
                      'gender_scores_fair',
                      'age_scores_fair']
    result.loc[result['race_preds_fair'] == 0, 'race'] = 'White'
    result.loc[result['race_preds_fair'] == 1, 'race'] = 'Black'
    result.loc[result['race_preds_fair'] == 2, 'race'] = 'Latino_Hispanic'
    result.loc[result['race_preds_fair'] == 3, 'race'] = 'East Asian'
    result.loc[result['race_preds_fair'] == 4, 'race'] = 'Southeast Asian'
    result.loc[result['race_preds_fair'] == 5, 'race'] = 'Indian'
    result.loc[result['race_preds_fair'] == 6, 'race'] = 'Middle Eastern'

    # race fair 4

    result.loc[result['race_preds_fair_4'] == 0, 'race4'] = 'White'
    result.loc[result['race_preds_fair_4'] == 1, 'race4'] = 'Black'
    result.loc[result['race_preds_fair_4'] == 2, 'race4'] = 'Asian'
    result.loc[result['race_preds_fair_4'] == 3, 'race4'] = 'Indian'

    # gender
    result.loc[result['gender_preds_fair'] == 0, 'gender'] = 'Male'
    result.loc[result['gender_preds_fair'] == 1, 'gender'] = 'Female'

    # age
    result.loc[result['age_preds_fair'] == 0, 'age'] = '0-2'
    result.loc[result['age_preds_fair'] == 1, 'age'] = '3-9'
    result.loc[result['age_preds_fair'] == 2, 'age'] = '10-19'
    result.loc[result['age_preds_fair'] == 3, 'age'] = '20-29'
    result.loc[result['age_preds_fair'] == 4, 'age'] = '30-39'
    result.loc[result['age_preds_fair'] == 5, 'age'] = '40-49'
    result.loc[result['age_preds_fair'] == 6, 'age'] = '50-59'
    result.loc[result['age_preds_fair'] == 7, 'age'] = '60-69'
    result.loc[result['age_preds_fair'] == 8, 'age'] = '70+'

    result[['face_name_align',
            'race', 'race4',
            'gender', 'age',
            'race_scores_fair', 'race_scores_fair_4',
            'gender_scores_fair', 'age_scores_fair']].to_csv(save_prediction_at, index=False)

    print("saved results at ", save_prediction_at)


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
                          
if __name__ == "__main__":
    #Please create a csv with one column 'img_path', contains the full paths of all images to be analyzed.
    #Also please change working directory to this file.
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', dest='input_csv', action='store',
                        help='csv file of image path where col name for image path is "img_path')
    dlib.DLIB_USE_CUDA = True
    print("using CUDA?: %s" % dlib.DLIB_USE_CUDA)
    args = parser.parse_args()
    SAVE_DETECTED_AT = "detected_faces"
    ensure_dir(SAVE_DETECTED_AT)
    imgs = pd.read_csv(args.input_csv)['img_path']
    detect_face(imgs, SAVE_DETECTED_AT)
    print("detected faces are saved at ", SAVE_DETECTED_AT)
    #Please change test_outputs.csv to actual name of output csv. 
    predict_age_gender_race("test_outputs.csv", SAVE_DETECTED_AT)