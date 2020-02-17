import cv2
import json
import os
import uuid
import shutil
from os.path import join
from sklearn.model_selection import train_test_split
import numpy as np

BASE_LINK = os.getcwd()
# FOLDER = ['basket_out', 'packing', 'paying', 'scanning', 'go_out']
FOLDER = ['basket_out', 'packing', 'paying', 'scanning']
BBOX = [[840, 350, 1470, 1080], [850, 350, 1470, 1000], [400, 430, 1000, 1080], [840, 350, 1500, 1080], [0, 0, 1920, 1080]]

LABELS = {'basket_out': 0, 'packing': 1, 'paying': 2, 'scanning': 3}
SELECTED_KEYPOINTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

class Kinect:
    """
    Use data from kinect's camera to get new form json, which serve for training later.
    """
    def __init__(self, vid_link, json_link, selected_keypoints, bbox):
        #selected_keypoints is an array of keypoints used

        self.vid_link = vid_link
        self.json_link = json_link
        self.selected_keypoints = selected_keypoints
        self.name = vid_link.split('/')[-1].split('.')[0]
        self.bbox = bbox
        self.dir = BASE_LINK + '/record/DATN/' + self.name

        self.num_img = self.get_num_img()
        self.data = self.read_json()

    def get_num_img(self):
        img_dir = self.dir + '/images'
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
            os.chdir(img_dir)
            cap = cv2.VideoCapture(self.vid_link)
            id = 0
            while True:
                _, img = cap.read()
                if img is None:
                    break
                img = cv2.resize(img, (1280, 720))
                name = self.name + '_%04d' % (id) + '.jpg'
                cv2.imwrite(name, img)
                id += 1
        os.chdir(BASE_LINK)
        return len(os.listdir(img_dir))

    def read_json(self):
        # read data from kinect'camera and re-format it into json file

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
        file = file[0::2]
        # for id, f in enumerate(file[-self.num_img:]):
        for id, f in enumerate(file[:self.num_img]):
            image = {}
            image['rights_holder'] = '---bestmonster---'
            image['license'] = '0'
            image['file_name'] = self.name + '_%04d' %id + '.jpg'
            image['url'] = BASE_LINK + '/' + image['file_name']
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

    def draw_kps(self, thr=0.01):
        # automatic save img (after capture from video), save video ( with draw keypoints) is optionally

        imgs = [self.dir + '/images/' + l for l in os.listdir(self.dir + '/images')]
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        os.chdir(self.dir)
        out_vid = cv2.VideoWriter(self.name + '.avi', cv2.VideoWriter_fourcc(*"MJPG"), 5, (1920, 1080))

        for id,keypoint in enumerate(self.data['annotations']):
            keypoint = keypoint['keypoints']
            img = cv2.imread(imgs[id])
            img = cv2.resize(img, (1920, 1080))
            BODY_PARTS_KPT_IDS = []
            skeleton = self.data['categories'][0]['skeletons']
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

    def create_json(self):
        # save json file

        os.chdir(self.dir)
        with open('annotations.json', 'w') as f:
            json.dump(self.data, f)

def action(action):
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
    elif action == 'bbox':
        for f in FOLDER:
            img = cv2.imread('/home/nbviet/Desktop/record/hihi/color' + f + '/images/color' + f +'_0000.jpg')
            img = cv2.resize(img, (960, 540))
            cv2.imshow('hi', img)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
    elif action == 'video':
        out_vid = cv2.VideoWriter('test.avi', cv2.VideoWriter_fourcc(*"MJPG"), 10, (1920, 1080))
        os.chdir(BASE_LINK)
        for id, f in enumerate(FOLDER):
            for img in ['/home/nbviet/Desktop/record/hihi/color' + f + '/images/' + l for l in
                        os.listdir('/home/nbviet/Desktop/record/hihi/color'+ f +'/images')]:
                img = cv2.imread(img)
                img = cv2.rectangle(img, (BBOX[id][0], BBOX[id][1]), (BBOX[id][2], BBOX[id][3]), (0, 0, 255), 3)
                cv2.imshow('hi', img)
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
                out_vid.write(img)
        cv2.destroyAllWindows()
    elif action == 'append':
        image_arr = []
        annotations_arr = []
        img_id = 0
        for id,f in enumerate(os.listdir(BASE_LINK)):
            with open(BASE_LINK + f + '/annotations.json') as file:
                data = json.load(file)
            img = data['images']
            annos = data['annotations']
            for i in range(len(img)):
                img[i]['id'] = img_id
                annos[i]['image_id'] = img_id
                annos[i]['bbox'] = BBOX[id]
                image_arr.append(img[i])
                annotations_arr.append(annos[i])
                img_id += 1

            # append images
            if not os.path.exists(BASE_LINK + '/images'):
                os.makedirs(BASE_LINK + '/images')

            # link = BASE_LINK + '/record/' + folder + '/' + l + '/images'
            link = BASE_LINK + '/record/hihi/' + l + '/images'
            for l in os.listdir(link):
                shutil.copy(link + '/' + l, BASE_LINK + '/images')
        final_json = {}
        final_json['images'] = image_arr
        final_json['annotations'] = annotations_arr
        final_json['categories'] = data['categories']

        os.chdir('/home/nbviet/Desktop')
        with open('annotations.json', 'w') as f:
            json.dump(final_json, f)

def split(json_link, img_link):
    """
    input: img_link: link to img's folder
           json_link: link to annotations's file
    output: train set and test set, each set contains img's folder and annotations's file
    """
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
            if data['images'][id]['file_name'] == img:
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
            if data['images'][id]['file_name'] == img:
                test_imgs.append(data['images'][id])
                test_annos.append(data['annotations'][id])
    test['images'] = test_imgs
    test['annotations'] = test_annos
    test['categories'] = data['categories']
    os.chdir(BASE_LINK)
    with open('test/annotations.json', 'w') as f:
        json.dump(test, f)

def create_data(vid_folder, json_folder, draw_kps=False):
    """
    input: an array of vid_link, an array of corresponding json_link
    output: img's folder and annotations for each link
    """
    for i in range(len(vid_folder)):
        kinect = Kinect(vid_folder[i], json_folder[i], SELECTED_KEYPOINTS, BBOX[0])
        kinect.create_json()
        if draw_kps:
            kinect.draw_kps()


if __name__=='__main__':

    # FOLDER = ['1581647146889', '1581648779050', '1581649157582', '1581649440125', '1581650686624', '1581650766949', '1581650923659']
    # BBOX = [[620, 220, 1360, 960], [650, 220, 1360, 960], [560, 220, 1360, 960], [580, 260, 1360, 960], [640, 280, 1360, 960], [640, 280, 1360, 960], [640, 280, 1360, 960]]
    # link = BASE_LINK + '/database/DATN/'
    # a = os.listdir(link)
    # vid_folder = [link + l for l in a[0::2]]
    # json_folder = [link + l for l in a[1::2]]
    vid_folder = ['/media/nbviet/UBUNTU 18_0/color1581907338881.webm']
    json_folder = ['/media/nbviet/UBUNTU 18_0/skeleton1581907338881.json']
    create_data(vid_folder, json_folder, True)