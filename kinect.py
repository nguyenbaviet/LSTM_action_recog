import cv2
import json
import os
import uuid
import shutil
from os.path import join
from sklearn.model_selection import train_test_split
import numpy as np

BASE_LINK = '/home/hoang/datasets/se7en11'
# FOLDER = ['basket_out', 'packing', 'paying', 'scanning', 'go_out']
FOLDER = ['basket_out', 'packing', 'paying', 'scanning']
BBOX = [[840, 350, 1470, 1080], [850, 350, 1470, 1000], [400, 430, 1000, 1080], [840, 350, 1500, 1080], [0, 0, 1920, 1080]]

LABELS = {'basket_out': 0, 'packing': 1, 'paying': 2, 'scanning': 3}
SELECTED_KEYPOINTS = [4, 5, 6, 7, 8, 9, 10, 11]

class Kinect:
    """
    Use data from kinect's camera to get new form json, which serve for training later.
    """
    def __init__(self, vid_link, json_link, selected_keypoints, bbox, type, save_vid = False):
        #selected_keypoints is an array of keypoints used

        self.vid_link = vid_link
        self.json_link = json_link
        self.selected_keypoints = selected_keypoints
        self.name = vid_link.split('/')[-1].split('.')[0]
        self.bbox = bbox
        self.type = type
        self.dir = BASE_LINK + '/record/' + self.type + '/' + self.name
        self.save_vid = save_vid

        self.data = self.read_vid()

    def read_json(self):

        with open(self.json_link) as f:
            file = json.load(f)
        json_file = {}
        images_arr = []
        annotations_arr = []

        categories = {}
        cat = []
        categories['id'] = '0'
        categories['name'] = 'person_0'
        categories['supercategory'] = 'person'
        kp = {}
        kp['0'] = 'spinebase'
        kp['1'] = 'spinemid'
        kp['2'] = 'neck'
        kp['3'] = 'head'
        kp['4'] = 'shoulderleft'
        kp['5'] = 'elbowleft'
        kp['6'] = 'wristleft'
        kp['7'] = 'handleft'
        kp['8'] = 'shoulderright'
        kp['9'] = 'elbowright'
        kp['10'] = 'wristright'
        kp['11'] = 'handright'
        kp['12'] = 'hipleft'
        kp['13'] = 'kneeleft'
        kp['14'] = 'ankleleft'
        kp['15'] = 'footleft'
        kp['16'] = 'hipright'
        kp['17'] = 'kneeright'
        kp['18'] = 'ankleright'
        kp['19'] = 'footright'
        kp['20'] = 'spineshoulder'
        kp['21'] = 'handtileft'
        kp['22'] = 'thumbleft'
        kp['23'] = 'handtiright'
        kp['24'] = 'thumbright'

        categories['keypoints'] = kp

        skeleton = {}
        skeleton['0'] = [3, 2]
        skeleton['1'] = [2, 20]
        skeleton['2'] = [20, 1]
        skeleton['3'] = [1, 0]
        skeleton['4'] = [20, 8]
        skeleton['5'] = [8, 9]
        skeleton['6'] = [9, 10]
        skeleton['7'] = [10, 11]
        skeleton['8'] = [11, 23]
        skeleton['9'] = [11, 24]
        skeleton['10'] = [20, 4]
        skeleton['11'] = [4, 5]
        skeleton['12'] = [5, 6]
        skeleton['13'] = [6, 7]
        skeleton['14'] = [7, 21]
        skeleton['15'] = [7, 22]
        skeleton['16'] = [0, 12]
        skeleton['17'] = [0, 16]
        skeleton['18'] = [16, 17]
        skeleton['19'] = [17, 18]
        skeleton['20'] = [18, 19]
        skeleton['21'] = [12, 13]
        skeleton['22'] = [13, 14]
        skeleton['23'] = [14, 15]

        categories['skeletons'] = skeleton

        categories['selected_keypoint'] = self.selected_keypoints
        cat.append(categories)
        for id, f in enumerate(file):
            image = {}
            image['rights_holder'] = '---bestmonster---'
            image['license'] = '0'
            image['filename'] = self.name + '_%04d' %id + '.jpg'
            image['url'] = BASE_LINK + '/' + image['filename']
            image['height'] = 1920
            image['width'] = 1080
            image['id'] = id

            images_arr.append(image)

            for keypoints in f['bodies']:
                if keypoints['tracked']:
                    annotation = {}
                    annotation['image_id'] = id
                    annotation['iscrowd'] = 0
                    annotation['bbox'] = self.bbox
                    kp = []
                    num_keypoint = 25
                    for keypoint in keypoints['joints']:
                        confident = 2 if keypoint['jointType'] in self.selected_keypoints else 0
                        if keypoint['colorX'] is None:
                            num_keypoint -= 1
                            confident = 0
                            keypoint['colorX'] = 0
                            keypoint['colorY'] = 0
                        kp.append(round(keypoint['colorX'] * 1920))
                        kp.append(round(keypoint['colorY'] * 1080))
                        kp.append(confident)
                    annotation['num_keypoint'] = num_keypoint
                    annotation['keypoints'] = kp
                    annotation['category_id'] = '0'
                    annotation['id'] = str(uuid.uuid1())
                    annotation['area'] = 1024

                    annotations_arr.append(annotation)

                    break      # 1 person was tracked in video

        license_arr = []
        lic_arr = {}
        lic_arr['url'] = '---bestmonster---'
        lic_arr['id'] = '0'
        lic_arr['name'] = 'bestmonster'
        license_arr.append(lic_arr)

        json_file['images'] = images_arr
        json_file['licenses'] = license_arr
        json_file['annotations'] = annotations_arr
        json_file['categories'] = cat
        return json_file

    def read_vid(self, thr=0.01):
        cap = cv2.VideoCapture(self.vid_link)
        dir = self.dir
        if not os.path.exists(dir):
            os.makedirs(dir)

        if self.save_vid:
            os.chdir(dir)
            out_vid = cv2.VideoWriter(self.name + '.avi', cv2.VideoWriter_fourcc(*"MJPG"), 10, (1920, 1080))
        save_img = False if os.path.exists(dir + '/images') else True
        if save_img:
            img_dir = dir + '/images'
            os.makedirs(img_dir)
            os.chdir(img_dir)
        data = self.read_json()
        for id,keypoint in enumerate(data['annotations']):
            keypoint = keypoint['keypoints']
            _, img = cap.read()
            if img is None:
                break
            img = cv2.resize(img, (1920, 1080))
            if save_img:

                name = self.name + '_%04d' %(id) + '.jpg'
                cv2.imwrite(name, img)
            if self.save_vid:
                BODY_PARTS_KPT_IDS = []
                skeleton = data['categories'][0]['skeletons']
                for i in range(24):
                    BODY_PARTS_KPT_IDS.append(skeleton[str(i)])

                for i in self.selected_keypoints:
                    coordinate = (int(keypoint[3*i]), int(keypoint[3*i + 1]))
                    img = cv2.circle(img, coordinate, 5, (0, 0, 255), 3)
                for id in range(len(BODY_PARTS_KPT_IDS)):
                    part = BODY_PARTS_KPT_IDS[id]
                    if (keypoint[3 * part[0] + 2] < thr or keypoint[3*part[1] + 2] < thr):
                        continue
                    coordinate1 = (keypoint[3 * part[0]], keypoint[3 * part[0] + 1])
                    coordinate2 = (keypoint[3 * part[1]], keypoint[3 * part[1] + 1])
                    cv2.line(img, coordinate1, coordinate2, (0, 255, 255), 2)
                out_vid.write(img)

        num_img = len(os.listdir(self.dir + '/images'))

        data['images'] = data['images'][:num_img]
        data['annotations'] = data['annotations'][:num_img]
        data['categories'] = data['categories']

        return data

    def create_json(self):
        os.chdir(self.dir)
        with open(self.name + '.json', 'w') as f:
            json.dump(self.data, f)

    def create_label(self, n_seq):
        # n_seq = 174
        label = LABELS[self.type]
        data = np.zeros(n_seq * len(self.selected_keypoints) * 2)
        len_sl_kp = len(self.selected_keypoints)

        for id in range(len(self.data['annotations'])):
            for i in range(len_sl_kp):
                data[2 * id * len_sl_kp + 2*i] = self.data['annotations'][id]['keypoints'][self.selected_keypoints[i] * 3]
                data[2 * id * len_sl_kp + 2*i + 1] = self.data['annotations'][id]['keypoints'][
                    self.selected_keypoints[i] * 3 + 1]
        return label, data

def xxx(action):
    if action == 'kp':
        label = []
        data = []
        for id, folder in enumerate(FOLDER):
            base_link = '/home/vietnguyen/LSTM_keypoint/database/Kinect v2 joints/' + folder

            link = [join(base_link, f) for f in os.listdir(base_link)]

            link.sort()
            a = int(len(link) / 3)
            vid_folder = link[:a]
            json_folder = link[a:][0::2]

            assert len(vid_folder) == len(json_folder)

            for i in range(len(vid_folder)):
                vid_link = vid_folder[i]
                json_link = json_folder[i]
                kinect = Kinect(vid_link, json_link, SELECTED_KEYPOINTS, BBOX[id], type=folder, save_vid=False)
                kinect.create_json()
                l, d = kinect.create_label(n_seq= 174)
                label.append(l)
                data.append(d.tolist())
        os.chdir(BASE_LINK)
        recog = {}
        recog['label'] = label
        recog['data'] = data
        with open('action_recognition.json', 'w') as f:
            json.dump(recog, f)
    elif action == 'sum':
        sum = 0
        for folder in FOLDER:
            link = BASE_LINK + '/record/' + folder
            for l in os.listdir(link):
                file_link = link + '/' + l + '/images'
                sum = len(os.listdir(file_link)) if sum < len(os.listdir(file_link)) else sum
                if(len(os.listdir(file_link)) == 174):
                    print(file_link)
        print(sum)
    elif action == 'bbox':
        img = cv2.imread('record/scanning/color1580956167912/images/color1580956167912_0001.jpg')
        img = cv2.resize(img, (960, 540))
        cv2.imshow('hi', img)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
    elif action == 'video':
        out_vid = cv2.VideoWriter('test.avi', cv2.VideoWriter_fourcc(*"MJPG"), 10, (1920, 1080))
        os.chdir(BASE_LINK)
        for img in ['record/scanning/color1580956167912/images/' + l for l in
                    os.listdir('record/scanning/color1580956167912/images')]:
            img = cv2.imread(img)
            img = cv2.rectangle(img, (840, 350), (1500, 1270), (0, 0, 255), 3)
            cv2.imshow('hi', img)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
            out_vid.write(img)
        cv2.destroyAllWindows()
    elif action == 'append':
        image_arr = []
        annotations_arr = []
        for folder in FOLDER:
            for l in os.listdir(BASE_LINK + '/record/' + folder):
                #append json
                link = BASE_LINK + '/record/' + folder + '/' + l
                with open(link + '/' + l + '.json') as f:
                    data = json.load(f)
                img = data['images']
                annos = data['annotations']
                for i in range(len(img)):
                    image_arr.append(img[i])
                    annotations_arr.append(annos[i])

                #append images
                if not os.path.exists(BASE_LINK + '/images'):
                    os.makedirs(BASE_LINK + '/images')

                link = BASE_LINK + '/record/' + folder + '/' + l + '/images'
                for l in os.listdir(link):
                    shutil.copy(link + '/' + l, BASE_LINK + '/images')
        final_json = {}
        final_json['images'] = image_arr
        final_json['annotations'] = annotations_arr
        final_json['categories'] = data['categories']

        os.chdir(BASE_LINK)
        with open('annotations.json', 'w') as f:
            json.dump(final_json, f)

def split(json_link, img_link):
    img = [link for link in os.listdir(img_link)]

    img_train, img_test = train_test_split(img, test_size=0.3)

    with open(json_link) as f:
        data = json.load(f)

    #save train data
    train = {}
    dest = BASE_LINK + '/train/images'
    if not os.path.exists(dest):
        os.makedirs(dest)
    train_imgs = []
    train_annos = []
    for img in img_train:
        shutil.copy(img_link + '/' + img, dest)
        for id in range(len(data['images'])):
            if data['images'][id]['filename'] == img:
                train_imgs.append(data['images'][id])
                train_annos.append(data['annotations'][id])
    train['images'] = train_imgs
    train['annotations'] = train_annos
    train['categories'] = data['categories']
    os.chdir(BASE_LINK)
    with open('train/annotations.json', 'w') as f:
        json.dump(train, f)

    #save test data
    test = {}
    dest = BASE_LINK + '/test/images'
    if not os.path.exists(dest):
        os.makedirs(dest)
    test_imgs = []
    test_annos = []
    for img in img_test:
        shutil.copy(img_link + '/' + img, dest)
        for id in range(len(data['images'])):
            if data['images'][id]['filename'] == img:
                test_imgs.append(data['images'][id])
                test_annos.append(data['annotations'][id])
    test['images'] = test_imgs
    test['annotations'] = test_annos
    test['categories'] = data['categories']
    os.chdir(BASE_LINK)
    with open('test/annotations.json', 'w') as f:
        json.dump(test, f)



if __name__=='__main__':

    xxx('kp')
    xxx('append')
    img_link = BASE_LINK + '/images'
    json_link = BASE_LINK + '/annotations.json'
    split(json_link, img_link)
    # xxx('sum')


