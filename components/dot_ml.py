import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR


class DotNet(nn.Module):
    """
    This class contain the implementation of the policy net.
    """
    def __init__(self, classes=2):
        super(DotNet, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, kernel_size=7, stride=3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(8),
            torch.nn.Conv2d(8, 16, kernel_size=5, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 64, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.Flatten()
        )

        self.head = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(32),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 3)
        )

        #self.backbone.to(self.device)
        self.head.to(self.device)

        self.head.apply(self.init_weights)


    def init_weights(self, m):
        """
        This method allow to init the weight of the model.
        @param m: the torch.nn.sequential class that need to be initialised.
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def prepare_data(self, S):
        """
        prepare the data in a format allowed by the model. Here it transform a tensor of cv2/numpy img into a PIL tensor
        format tensor image.
        @param state: the state given by the environment.
        @return: the transformed tensor
        """
        S = torch.from_numpy(S).float()
        S = S.to(self.device)
        print(S.shape)
        return S.permute(0, 3, 1, 2)

    def forward(self, state):
        """
        the surcharged forward method.
        @param state: a tensor of state.
        @return: the probabilities of taking an action for the state.
        """
        state = self.backbone(state)
        return self.head(state)

    def get_pred(self, preds):
        soft = torch.nn.functional.softmax(preds, dim=1)
        print(soft)
        pred = torch.argmax(soft).item()
        conf = soft[0][pred].item()
        return pred, conf


class DOT:
    """
    The class DOT (Detection of target). it is basically a policy gradient descent.
    """

    def __init__(self, environment, learning_rate=0.005, gamma=0.05,
                lr_gamma=0.7, pa_dataset_size=3000, pa_batch_size=200):

        self.gamma = gamma
        self.environment = environment

        self.policy = DotNet()

        self.pa_dataset_size = pa_dataset_size
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=lr_gamma)
        self.pa_batch_size = pa_batch_size

        # Past Actions Buffer
        self.X_batch = None
        self.Y_batch = None
        self.X_batch_test = None
        self.Y_batch_test = None

    def save(self, file):
        """
        this method allow user to save the weights of the policy net.
        @param file: the filename given to the .pt file.
        """
        torch.save(self.policy.state_dict(), file)

    def load(self, weights):
        """
        this method can be used to load weights already trained.
        @param weights: the path to the weights.
        """
        self.policy.load_state_dict(torch.load(weights))

    def model_summary(self):
        """
        print the summary of the model.
        """
        print("RUNNING ON {0}".format(self.policy.device))
        print(self.policy)
        print("TOTAL PARAMS: {0}".format(sum(p.numel() for p in self.policy.parameters())))

    def add_to_batch(self, X, Y, X_test, Y_test):
        X = self.policy.prepare_data(np.array(X))
        Y = np.array(Y)
        X_test = self.policy.prepare_data(np.array(X_test))
        Y_test = np.array(Y_test)

        if self.X_batch is None:
            self.X_batch = torch.FloatTensor(X).to(self.policy.device)
            self.Y_batch = torch.LongTensor(Y).to(self.policy.device)
            self.X_batch_test = torch.FloatTensor(X_test).to(self.policy.device)
            self.Y_batch_test = torch.LongTensor(Y_test).to(self.policy.device)
        else:
            self.X_batch = torch.cat((self.X_batch, torch.FloatTensor(X).to(self.policy.device)), 0)
            self.Y_batch = torch.cat((self.Y_batch, torch.LongTensor(Y).to(self.policy.device)), 0)
            self.X_batch_test = torch.cat((self.X_batch_test, torch.FloatTensor(X_test).to(self.policy.device)), 0)
            self.Y_batch_test = torch.cat((self.Y_batch_test, torch.LongTensor(Y_test).to(self.policy.device)), 0)

        if len(self.X_batch) > self.pa_dataset_size:
            surplus = len(self.X_batch) - self.pa_dataset_size
            _, self.X_batch = torch.split(self.X_batch, [surplus, self.pa_dataset_size])
            _, self.Y_batch = torch.split(self.Y_batch, [surplus, self.pa_dataset_size])
        if len(self.X_batch_test) > self.pa_dataset_size:
            surplus = len(self.X_batch) - self.pa_dataset_size
            _, self.X_batch_test = torch.split(self.X_batch_test, [surplus, self.pa_dataset_size])
            _, self.Y_batch_test = torch.split(self.Y_batch_test, [surplus, self.pa_dataset_size])

    def update_policy(self):

        shuffle_index = torch.randperm(len(self.X_batch))
        self.X_batch = self.X_batch[shuffle_index]
        self.Y_batch = self.Y_batch[shuffle_index]

        shuffle_index = torch.randperm(len(self.X_batch_test))
        self.X_batch_test = self.X_batch_test[shuffle_index]
        self.Y_batch_test = self.Y_batch_test[shuffle_index]
        print(len(self.X_batch))
        print(self.Y_batch)

        if len(self.X_batch) < self.pa_batch_size or len(self.X_batch_test) < self.pa_batch_size:
            return 0.

        X = self.X_batch[:self.pa_batch_size]
        Y = self.Y_batch[:self.pa_batch_size]
        X_test = self.X_batch_test[:self.pa_batch_size]
        Y_test = self.Y_batch_test[:self.pa_batch_size]

        self.optimizer.zero_grad()
        preds = self.policy(X)

        loss_train = torch.nn.functional.cross_entropy(preds, Y)
        loss_train.backward()
        self.optimizer.step()
        self.scheduler.step()

        with torch.no_grad():
            preds = self.policy(X_test)
            loss_test = torch.nn.functional.cross_entropy(preds, Y_test)

        return loss_train.item(), loss_test.item()

    def predict(self, S):
        S = self.policy.prepare_data(np.array([S]))
        preds = self.policy(S)
        return self.policy.get_pred(preds)
