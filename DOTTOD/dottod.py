import argparse
import random
import cv2
import numpy as np
import torch
from torch import nn
import os

AGENT_RES = 64
TASK_MODEL_RES = 200


# ----------------------------------------------------------------------------------------------------------------------
# DOT
# ----------------------------------------------------------------------------------------------------------------------


class PolicyNetDot(nn.Module):
    """
    This class contain the implementation of the policy net.
    """

    def __init__(self, actions=2):
        super(PolicyNetDot, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.action_space = np.arange(actions)
        self.nb_actions = actions

        # The feature extractor
        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=7, stride=3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16, 32, kernel_size=5, stride=2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64, 128, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.Flatten()
        )

        # the mlp
        self.head = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, actions)
        )

        self.backbone.to(self.device)
        self.head.to(self.device)

    def prepare_data(self, state):
        """
        prepare the data in a format allowed by the model. Here it transform a tensor of cv2/numpy img into a PIL tensor
        format tensor image.
        @param state: the state given by the environment.
        @return: the transformed tensor
        """
        return state.permute(0, 3, 1, 2)

    def forward(self, state):
        """
        the surcharged forward method.
        @param state: a tensor of state.
        @return: the probabilities of taking an action for the state.
        """
        x = self.backbone(state)
        return self.head(x)


class DOT:
    """
    The class DOT (Detection of target). it is basically a policy gradient descent.
    """

    def __init__(self, environment):
        self.environment = environment
        self.environment.agent = self
        self.policy = PolicyNetDot()

    def load(self, weights):
        """
        this method can be used to load weights already trained.
        @param weights: the path to the weights.
        """
        self.policy.load_state_dict(torch.load(weights))

    def __call__(self, S):
        """
        this method allow the agent to exploit on the environment.
        @param S: the first state given by the environment.
        @return: the sum of rewards during the episode.
        """
        while True:
            # State preprocess
            S = torch.from_numpy(S).float()
            S = S.unsqueeze(0).to(self.policy.device)
            S = self.policy.prepare_data(S)

            with torch.no_grad():
                action_probs = self.policy(S)
                # no need to explore anymore.
                A = np.argmax(action_probs)

            S_prime, is_terminal = self.environment.take_action(A)

            S = S_prime
            if is_terminal:
                break


# ----------------------------------------------------------------------------------------------------------------------
# TOD
# ----------------------------------------------------------------------------------------------------------------------


class PolicyNetTod(nn.Module):
    def __init__(self, nb_actions=6, classes=5):
        super(PolicyNetTod, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.action_space = np.arange(nb_actions)
        self.nb_actions = nb_actions
        self.classes = classes

        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=7, stride=3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16, 32, kernel_size=5, stride=2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64, 128, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.Flatten()
        )

        self.policy_head = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(32),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, self.nb_actions)
        )

        self.class_head = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(32),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, classes)
        )

        self.conf_head = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1)
        )

        self.backbone.to(self.device)
        self.policy_head.to(self.device)
        self.class_head.to(self.device)
        self.conf_head.to(self.device)

    def get_class(self, class_preds):
        proba = torch.nn.functional.softmax(class_preds, dim=1).squeeze()
        pred = torch.argmax(proba).item()
        return pred

    def prepare_data(self, state):
        return state.permute(0, 3, 1, 2)

    def forward(self, state):
        x = self.backbone(state)
        preds = self.policy_head(x)
        class_preds = self.class_head(x)
        conf = self.conf_head(x)
        return preds, conf, class_preds


class TOD:

    def __init__(self, environment, classes):
        self.environment = environment
        self.policy = PolicyNetTod(classes=classes)

    def load(self, weights):
        self.policy.load_state_dict(torch.load(weights))

    def __call__(self, S):
        while True:

            S = torch.from_numpy(S).float()
            S = S.unsqueeze(0).to(self.policy.device)
            S = self.policy.prepare_data(S)

            with torch.no_grad():
                action_probs, conf, class_preds = self.policy(S)
                conf = conf.item()

                action_probs = action_probs.detach().cpu().numpy()[0]
                A = np.argmax(action_probs)

                label = self.policy.get_class(class_preds)

            S_prime, is_terminal = self.environment.take_action_tod(A, conf, label)

            S = S_prime
            if is_terminal:
                break


# ----------------------------------------------------------------------------------------------------------------------
# ENVIRONMENT
# ----------------------------------------------------------------------------------------------------------------------


class Environment:
    """
    this class implement a problem where the agent must mark the place where he have found boat.
    He must not mark place where there is house.
    """

    def __init__(self, nb_class=4, conf_threshold=0.2):
        self.bboxes = None
        self.nb_actions_taken_tod = None
        self.current_bbox = None
        self.full_img = None
        self.base_img = None
        self.nb_actions_taken = 0
        self.tod = None
        self.nb_classes = nb_class
        self.best_bbox = None
        self.conf_threshold = conf_threshold

        self.colors = []
        for i in range(self.nb_classes + 1):
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            self.colors.append([r, g, b])

    def reload_env(self, img):
        """
        reload the environment on a new image.
        @param img: the img filename.
        @param bb: the bounding box filename.
        :return: the first state of the environment.
        """

        # prepare the image for the environment.
        self.prepare_img(img)
        # prepare the coordinates for the environment.
        self.bboxes = []
        self.nb_actions_taken = 0
        return self.get_state()

    def reload_env_tod(self, bb):
        """
        reload the environment to train tod.
        @param bb: the bounding box that will be optimised by tod.
        @return: the first state for tod.
        """
        bb_x, bb_y = bb
        bb_w, bb_h = 64, 64

        if bb_x + bb_w >= 200:
            bb_x = 136

        if bb_y + bb_h >= 200:
            bb_y = 136

        self.current_bbox = {
            'x': bb_x,
            'y': bb_y,
            'w': bb_w,
            'h': bb_h,
            'conf': 0.,
            'label': 0.
        }

        self.best_bbox = self.current_bbox.copy()
        self.nb_actions_taken_tod = 0

        return self.get_tod_state()

    def prepare_img(self, img):
        """
        prepare the image for the environment.
        @param img: the image used to train.
        """
        self.full_img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
        if self.full_img is None or 4 > len(self.full_img.shape) != 3:
            raise ValueError("Image provided must be a single BGR cv2 image")

        self.base_img = self.full_img.copy()


    def get_current_coord(self):
        """
        using the current steps to calculate the current coordinate of dot.
        @return: the current x and y.
        """
        x = int(self.nb_actions_taken % 10)
        y = int(self.nb_actions_taken / 10)
        pad = int((200 - AGENT_RES) / 10)

        return x * pad, y * pad

    def get_state(self):
        """
        give the current state for the agent.
        @return: a 64x64 image.
        """

        x, y = self.get_current_coord()

        temp = self.full_img[y: y + AGENT_RES, x: x + AGENT_RES]
        temp = cv2.resize(temp, (AGENT_RES, AGENT_RES)) / 255.

        return temp

    def get_tod_state(self):
        """
        give the current state for the tod agent.
        @return: a 64x64 image.
        """
        temp = self.base_img[self.current_bbox['y']: self.current_bbox['y'] + self.current_bbox['h'],
               self.current_bbox['x']: self.current_bbox['x'] + self.current_bbox['w']]
        temp = cv2.resize(temp, (AGENT_RES, AGENT_RES)) / 255.
        return temp

    # https://gist.github.com/meyerjo/dd3533edc97c81258898f60d8978eddc
    def intersection_over_union(self, boxA, boxB):
        """
        This method calculate the intersection over union of 2 bounding boxes.
        @param boxA: a bounding box given by (x, y, w, h).
        @param boxB: a bounding box given by (x, y, w, h).
        @return: the iou.
        """

        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2] + boxA[0], boxB[2] + boxB[0])
        yB = min(boxA[3] + boxA[1], boxB[3] + boxB[1])

        # compute the area of intersection rectangle
        interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
        if interArea == 0:
            return 0
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]

        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

    def non_max_suppression(self, boxes, iou_threshold=0.1):
        """
        eliminate the bounding box with a confidence below the threshold and bounding boxes that are specifying the
        same object.
        @param boxes: the list of bounding boxes
        @param conf_threshold: the confidence threshold. BB under that threshold are removed.
        @param iou_threshold: bounding box that collide with an iou over that threshold are removed.
        @return: the new list of bounding boxes.
        """
        bbox_list_thresholded = []
        bbox_list_new = []

        boxes_sorted = sorted(boxes, reverse=True, key=lambda x: x[4])
        for box in boxes_sorted:
            if box[4] >= self.conf_threshold:
                bbox_list_thresholded.append(box)
            else:
                break

        while len(bbox_list_thresholded) > 0:
            current_box = bbox_list_thresholded.pop(0)
            bbox_list_new.append(current_box)
            for box in bbox_list_thresholded:
                iou = self.intersection_over_union(current_box[:4], box[:4])
                if iou > iou_threshold:
                    bbox_list_thresholded.remove(box)

        return bbox_list_new

    def take_action(self, action):
        """
        update the environment given the action chosen.
        @param action: the action chose by the agent.
        @return: the next state, the current reward and if the state is terminal.
        """

        # --------------------------------------------------------------------------------------------------------------
        # CALLING TOD IF TARGET IS DETECTED
        # --------------------------------------------------------------------------------------------------------------
        if action:
            # --------------------------------------------------------------------------------------------------------------
            # Get the current coordinate of the agent.
            x, y = self.get_current_coord()

            # reload the environment for tod.
            first_state_tod = self.reload_env_tod((x, y))
            self.tod(first_state_tod)
            # append the bbox to the bboxes found by tod.
            self.bboxes.append((self.best_bbox['x'], self.best_bbox['y'],
                                self.best_bbox['w'], self.best_bbox['h'],
                                self.best_bbox['conf'], self.best_bbox['label']))

        # --------------------------------------------------------------------------------------------------------------
        # Prepare the new state.
        self.nb_actions_taken += 1
        # Get the new coordinate to give the new state.
        S_prime = self.get_state()

        is_terminal = False
        if self.nb_actions_taken >= 100:
            is_terminal = True

        return S_prime, is_terminal

    def take_action_tod(self, A, conf, label_pred):
        """
        update the environment given the action chosen.
        @param A: the action chose by the agent tod.
        @param conf: the iou that tod as predicted.
        @param label_pred: the label that tod has predicted.
        @return: the next state, the current reward, if the state is terminal, the iou and the label of the state.
        """

        if conf > self.best_bbox['conf']:
            self.best_bbox = self.current_bbox.copy()

        self.nb_actions_taken_tod += 1
        ratio = 0.1
        pad = int(ratio * np.mean([self.current_bbox['w'], self.current_bbox['h']]))
        pad_dep = pad

        if A == 0:
            if self.current_bbox['x'] + self.current_bbox['w'] < 200 - pad_dep:
                self.current_bbox['x'] += pad_dep
        elif A == 1:

            if self.current_bbox['y'] + self.current_bbox['h'] < 200 - pad_dep:
                self.current_bbox['y'] += pad_dep
        elif A == 2:
            if self.current_bbox['x'] >= pad_dep:
                self.current_bbox['x'] -= pad_dep
        elif A == 3:
            if self.current_bbox['y'] >= pad_dep:
                self.current_bbox['y'] -= pad_dep
        elif A == 4:
            if self.current_bbox['w'] - 2 * pad >= 32 and self.current_bbox['h'] - 2 * pad >= 32:
                self.current_bbox['w'] -= 2 * pad
                self.current_bbox['x'] += pad
                self.current_bbox['h'] -= 2 * pad
                self.current_bbox['y'] += pad
        elif A == 5:
            if self.current_bbox['x'] + self.current_bbox['w'] < 200 - pad and self.current_bbox['x'] >= pad and \
                    self.current_bbox['y'] + self.current_bbox['h'] < 200 - pad and self.current_bbox['y'] >= pad:
                self.current_bbox['w'] += 2 * pad
                self.current_bbox['x'] -= pad
                self.current_bbox['h'] += 2 * pad
                self.current_bbox['y'] -= pad

        is_terminal = False
        if self.nb_actions_taken_tod >= 50:
            is_terminal = True

        self.current_bbox['conf'] = conf
        self.current_bbox['label'] = label_pred

        next_state = self.get_tod_state()

        return next_state, is_terminal

    def TOD_history(self):

        bboxes = self.non_max_suppression(self.bboxes)
        history_img = self.base_img.copy()

        for bb in bboxes:
            x, y, w, h, conf, label = bb

            if label == self.nb_classes:
                continue

            p1 = (int(x), int(y))
            p2 = (int(x + w), int(y + h))
            history_img = cv2.rectangle(history_img, p1, p2, self.colors[label], 2)
            history_img = cv2.putText(history_img,
                                      str(label) + ' ' + str(round(conf, 2)),
                                      (int(x + 5), int(y + 10)),
                                      cv2.FONT_HERSHEY_SIMPLEX,
                                      0.3,
                                      self.colors[label], 1, cv2.LINE_AA)

        return history_img


# ----------------------------------------------------------------------------------------------------------------------
# DOTTOD
# ----------------------------------------------------------------------------------------------------------------------

class DotTod:
    """
    this class implement a problem where the agent must mark the place where he have found boat.
    He must not mark place where there is house.
    """

    def __init__(self, nb_class=4, tod_weights="tod-weights.pt", dot_weights="dot-weights.pt",
                 iou_threshold=0.2, return_img=False):
        self.env = Environment(nb_class=nb_class, conf_threshold=iou_threshold)

        filepath = os.path.join(os.path.dirname(__file__), "weights")
        dot_path = os.path.join(filepath, dot_weights)
        tod_path = os.path.join(filepath, tod_weights)

        self.tod = TOD(self.env, classes=nb_class)
        self.tod.load(tod_path)
        self.env.tod = self.tod
        self.dot = DOT(self.env)
        self.dot.load(dot_path)
        self.return_img = return_img

    def __call__(self, image):

        S = self.env.reload_env(image)
        self.dot(S)

        bounding_boxes = []
        for bb in self.env.bboxes:
            bounding_box = {
                'x': bb[0],
                'y': bb[1],
                'w': bb[2],
                'h': bb[3],
                'conf': bb[4],
                'label': bb[5],
            }

            bounding_boxes.append(bounding_box)

        if self.return_img:
            return bounding_box, self.env.TOD_history()

        return bounding_boxes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='This program allow user to train DOT/TOD on a specific dataset. The weight and result are saved '
                    'in the current directory')
    parser.add_argument('-img', '--images_path',
                        help='the path to the data.')
    parser.add_argument('-nb', '--nb_class', help='number of class')
    parser.add_argument('-plt', '--return_image', action='store_true')
    parser.add_argument('-conf', '--conf_threshold', default=0.2)

    args = parser.parse_args()

    dottod = DotTod(nb_class=int(args.nb_class), iou_threshold=float(args.conf_threshold), return_img=args.return_image)

    images_list = os.listdir(args.images_path)
    results = []
    for filename in images_list:
        if args.return_image:
            bboxes, img = dottod(os.path.join(args.images_path, filename))
            cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        else:
            bboxes = dottod(os.path.join(args.images_path, filename))
        result = {
            "filename": filename,
            "bboxes": bboxes
        }
        results.append(result)

    print(results)
