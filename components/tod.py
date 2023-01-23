import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms


class PolicyNet(nn.Module):
    """
    This class contain the implementation of the policy net for the tod agent.
    """
    def __init__(self, nb_actions=6, classes=5, epsilon=0.4):
        super(PolicyNet, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.action_space = np.arange(nb_actions)
        self.nb_actions = nb_actions
        self.classes = classes
        self.epsilon = epsilon

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

        self.policy_head.apply(self.init_weights)

    def follow_policy(self, probs):
        """
        this method allow the agent to choose an action randomly (for exploration) but with respect of the parameter e.
        @param probs: the probabilities returned by the model.
        @return: an action include in the action space.
        """
        p = np.random.random()
        if p < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            return np.argmax(probs)

    def get_class(self, class_preds):
        """
        get the class prediction of the model.
        @param class_preds: the prediction for each class.
        @return: the maximum class prediction.
        """
        proba = torch.nn.functional.softmax(class_preds, dim=1).squeeze()
        pred = torch.argmax(proba).item()
        return pred

    def init_weights(self, m):
        """
        This method allow to init the weight of the model.
        @param m: the torch.nn.sequential class that need to be initialised.
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

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
        @return: the probabilities of taking an action for the state, the confidence of a bbox and the class prediction.
        """
        x = self.backbone(state)
        preds = self.policy_head(x)
        class_preds = self.class_head(x)
        conf = self.conf_head(x)
        return preds, conf, class_preds


class TOD:
    """
    The Tod class. (Tiny Object Detection). 
    """

    def __init__(self, environment, learning_rate=0.0005, gamma=0.1, epsilon=0.4,
                 lr_gamma=0.9, pa_dataset_size=3000, pa_batch_size=50, nb_class=5):

        self.IOU_pa_batch = None
        self.gamma = gamma
        self.environment = environment
        self.environment.tod = self

        self.policy = PolicyNet(classes=nb_class, epsilon=epsilon)

        self.pa_dataset_size = pa_dataset_size
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.class_optimizer = torch.optim.Adam(self.policy.parameters(), lr=0.001)
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=lr_gamma)

        self.pa_batch_size = pa_batch_size

        # Past Actions Buffer
        self.S_pa_batch = None
        self.A_pa_batch = None
        self.G_pa_batch = None

    def save(self, file):
        torch.save(self.policy.state_dict(), file)

    def load(self, weights):
        self.policy.load_state_dict(torch.load(weights))

    def model_summary(self):
        print("RUNNING ON {0}".format(self.policy.device))
        print(self.policy)
        print("TOTAL PARAMS: {0}".format(sum(p.numel() for p in self.policy.parameters())))

    def prepare_ds(self):


        temp_x = self.S_pa_batch[self.LABEL_pa_batch >= 0]
        temp_y = self.LABEL_pa_batch[self.LABEL_pa_batch >= 0]
        _, count = torch.unique(temp_y, return_counts=True, dim=-1)

        min_dim = torch.min(count).item()
        equilibrate_X = []
        equilibrate_Y = []
        for i in range(len(count)):
            temp = temp_x[temp_y == i]
            equilibrate_X.append(temp[:min_dim])
            temp = temp_y[temp_y == i]
            equilibrate_Y.append(temp[:min_dim])

        temp_x = torch.concat(equilibrate_X)
        temp_y = torch.concat(equilibrate_Y)

        split = int(len(temp_y) / 10)
        self.Y_test, self.Y = torch.split(temp_y, [split, len(temp_y) - split])
        self.X_test, self.X = torch.split(temp_x, [split, len(temp_y) - split])

    def update_class_head(self):

        for i, param in enumerate(self.policy.class_head.parameters()):
            param.requires_grad = True
        for i, param in enumerate(self.policy.backbone.parameters()):
            param.requires_grad = False
        for i, param in enumerate(self.policy.policy_head.parameters()):
            param.requires_grad = False
        for i, param in enumerate(self.policy.conf_head.parameters()):
            param.requires_grad = False

        shuffle_index = torch.randperm(len(self.X))
        self.X = self.X[shuffle_index]
        self.Y = self.Y[shuffle_index]
        X = self.X[:int(len(self.Y) / 2)]
        Y = self.Y[:int(len(self.Y) / 2)]

        transform = transforms.RandomChoice(
            [transforms.RandomHorizontalFlip(),
             transforms.RandomVerticalFlip()]
        )

        X = transform(X)

        self.class_optimizer.zero_grad()

        _, _, class_preds = self.policy(X)
        class_loss = torch.nn.functional.cross_entropy(class_preds.squeeze(), Y.squeeze())

        l2_lambda = 0.001
        l2_norm = sum(p.pow(2.0).sum() for p in self.policy.parameters())
        #class_loss = class_loss + l2_lambda * l2_norm
        class_loss.backward()
        self.class_optimizer.step()

        with torch.no_grad():
            _, _, class_preds = self.policy(self.X_test)

        class_loss_test = torch.nn.functional.cross_entropy(class_preds.squeeze(), self.Y_test.squeeze())

        return class_loss.item(), class_loss_test.item()

    def update_policy(self):

        for i, param in enumerate(self.policy.class_head.parameters()):
            param.requires_grad = False
        for i, param in enumerate(self.policy.backbone.parameters()):
            param.requires_grad = True
        for i, param in enumerate(self.policy.policy_head.parameters()):
            param.requires_grad = True
        for i, param in enumerate(self.policy.conf_head.parameters()):
            param.requires_grad = True

        if self.A_pa_batch is None or len(self.A_pa_batch) < self.pa_batch_size:
            return 0., 0, 0

        shuffle_index = torch.randperm(len(self.A_pa_batch))
        self.A_pa_batch = self.A_pa_batch[shuffle_index]
        self.G_pa_batch = self.G_pa_batch[shuffle_index]
        self.S_pa_batch = self.S_pa_batch[shuffle_index]
        self.IOU_pa_batch = self.IOU_pa_batch[shuffle_index]
        self.LABEL_pa_batch = self.LABEL_pa_batch[shuffle_index]


        S = self.S_pa_batch[:self.pa_batch_size]
        A = self.A_pa_batch[:self.pa_batch_size]
        G = self.G_pa_batch[:self.pa_batch_size]
        IOU = self.IOU_pa_batch[:self.pa_batch_size]

        #S, A, G, IOU, LABEL = batch

        # Calculate loss
        self.optimizer.zero_grad()
        action_probs, conf, _ = self.policy(S)

        #log_probs = torch.log(action_probs)
        #log_probs = torch.nan_to_num(log_probs)
        action_prob = torch.gather(action_probs, 1, A.unsqueeze(1))
        #selected_log_probs = torch.gather(log_probs, 1, A.unsqueeze(1))
        #entropy_loss = - 0.001 * (action_probs * log_probs).sum(1).mean()

        #loss = - (G.unsqueeze(1) * selected_log_probs).mean()
        loss = torch.nn.functional.mse_loss(action_prob.squeeze(), G.squeeze())
        loss.backward(retain_graph=True)

        conf_loss = torch.nn.functional.mse_loss(conf.squeeze(), IOU.squeeze())
        conf_loss.backward()

        #conf_loss = torch.nn.functional.mse_loss(conf.squeeze(), IOU.squeeze())
       # conf_loss.backward(retain_graph=True)

        self.optimizer.step()
        self.scheduler.step()

        return loss.item(), conf_loss.item(), 0#, conf_loss.item(), class_loss.item()

    def fit_one_episode(self, S):

        # ------------------------------------------------------------------------------------------------------
        # EPISODE PREPARATION
        # ------------------------------------------------------------------------------------------------------
        S_batch = []
        R_batch = []
        A_batch = []
        IOU_batch = []
        LABEL_batch = []

        # ------------------------------------------------------------------------------------------------------
        # EPISODE REALISATION
        # ------------------------------------------------------------------------------------------------------
        counter = 0
        sum_reward = 0

        counter += 1
        # State preprocess

        while True:
            S = torch.from_numpy(S).float()
            S = S.unsqueeze(0).to(self.policy.device)
            S = self.policy.prepare_data(S)
            with torch.no_grad():
                action_probs, conf, class_preds = self.policy(S)

                action_probs = action_probs.detach().cpu().numpy()[0]
                A = self.policy.follow_policy(action_probs)
                #conf = action_probs[A].item()

                label_pred = self.policy.get_class(class_preds)

            S_prime, R, is_terminal, iou, label = self.environment.take_action_tod(A, conf.item(), label_pred)

            S_batch.append(S)
            A_batch.append(A)
            R_batch.append(R)
            IOU_batch.append(iou)
            LABEL_batch.append(label)
            sum_reward += R

            S = S_prime

            if is_terminal:
                break

        for i in reversed(range(1, len(R_batch))):
            R_batch[i - 1] += self.gamma * R_batch[i]

        # ------------------------------------------------------------------------------------------------------
        # BATCH PREPARATION
        # ------------------------------------------------------------------------------------------------------
        S_batch = torch.concat(S_batch).to(self.policy.device)
        A_batch = torch.LongTensor(A_batch).to(self.policy.device)
        G_batch = torch.FloatTensor(R_batch).to(self.policy.device)
        IOU_batch = torch.FloatTensor(IOU_batch).to(self.policy.device)
        LABEL_batch = torch.LongTensor(LABEL_batch).to(self.policy.device)


        # ------------------------------------------------------------------------------------------------------
        # PAST ACTION DATASET PREPARATION
        # ------------------------------------------------------------------------------------------------------
        # Append the past action batch to the current batch if possible

        #if self.A_pa_batch is not None and len(self.A_pa_batch) > self.pa_batch_size:
        #    batch = (torch.cat((self.S_pa_batch[0:self.pa_batch_size], S_batch), 0),
        #             torch.cat((self.A_pa_batch[0:self.pa_batch_size], A_batch), 0),
        #             torch.cat((self.G_pa_batch[0:self.pa_batch_size], G_batch), 0),
        #             torch.cat((self.IOU_pa_batch[0:self.pa_batch_size], IOU_batch), 0),
        #             torch.cat((self.LABEL_pa_batch[0:self.pa_batch_size], LABEL_batch), 0),)
        #else:
        #    batch = (S_batch, A_batch, G_batch, IOU_batch, LABEL_batch)

        # Add some experiences to the buffer with respect of TD error
        nb_new_memories = 5

        #weights = G_batch + 1
        #weights /= torch.sum(weights)
        #idx = torch.multinomial(weights, nb_new_memories)
        idx = torch.randperm(len(A_batch))[:nb_new_memories]

        if self.A_pa_batch is None:
            self.A_pa_batch = A_batch[idx]
            self.S_pa_batch = S_batch[idx]
            self.G_pa_batch = G_batch[idx]
            self.IOU_pa_batch = IOU_batch[idx]
            self.LABEL_pa_batch = LABEL_batch[idx]
        else:
            self.A_pa_batch = torch.cat((self.A_pa_batch, A_batch[idx]), 0)
            self.S_pa_batch = torch.cat((self.S_pa_batch, S_batch[idx]), 0)
            self.G_pa_batch = torch.cat((self.G_pa_batch, G_batch[idx]), 0)
            self.IOU_pa_batch = torch.cat((self.IOU_pa_batch, IOU_batch[idx]), 0)
            self.LABEL_pa_batch = torch.cat((self.LABEL_pa_batch, LABEL_batch[idx]), 0)

        # clip the buffer if it's to big
        if len(self.A_pa_batch) > self.pa_dataset_size:
            # shuffling the batch

            #shuffle_index = torch.randperm(len(self.A_pa_batch))
            #self.A_pa_batch = self.A_pa_batch[shuffle_index]
            #self.G_pa_batch = self.G_pa_batch[shuffle_index]
            #self.S_pa_batch = self.S_pa_batch[shuffle_index]
            #self.IOU_pa_batch = self.IOU_pa_batch[shuffle_index]
            #self.LABEL_pa_batch = self.LABEL_pa_batch[shuffle_index]

            # dataset clipping
            surplus = len(self.A_pa_batch) - self.pa_dataset_size
            _, self.A_pa_batch = torch.split(self.A_pa_batch, [surplus, self.pa_dataset_size])
            _, self.G_pa_batch = torch.split(self.G_pa_batch, [surplus, self.pa_dataset_size])
            _, self.S_pa_batch = torch.split(self.S_pa_batch, [surplus, self.pa_dataset_size])
            _, self.IOU_pa_batch = torch.split(self.IOU_pa_batch, [surplus, self.pa_dataset_size])
            _, self.LABEL_pa_batch = torch.split(self.LABEL_pa_batch, [surplus, self.pa_dataset_size])

        # ------------------------------------------------------------------------------------------------------
        # MODEL OPTIMISATION
        # ------------------------------------------------------------------------------------------------------

        return iou, sum_reward, 0, 0,0

    def exploit_one_episode(self, S):
        sum_reward = 0
        while True:
            # State preprocess

            S = torch.from_numpy(S).float()
            S = S.unsqueeze(0).to(self.policy.device)
            S = self.policy.prepare_data(S)

            with torch.no_grad():
                action_probs, conf, class_preds = self.policy(S)
                #class_preds = self.class_net(S)
                conf = conf.item()

                action_probs = action_probs.detach().cpu().numpy()[0]
                A = self.policy.follow_policy(action_probs)

                label = self.policy.get_class(class_preds)

            S_prime, R, is_terminal, iou, label = self.environment.take_action_tod(A, conf, label)

            S = S_prime
            sum_reward += R
            if is_terminal:
                break

        return iou, sum_reward
