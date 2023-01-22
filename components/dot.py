import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR


class PolicyNet(nn.Module):
    """
    This class contain the implementation of the policy net.
    """
    def __init__(self, epsilon=0.2, actions=2):
        super(PolicyNet, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.action_space = np.arange(actions)
        self.nb_actions = actions
        self.e = epsilon

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

        self.head.apply(self.init_weights)

    def follow_policy(self, probs):
        """
        this method allow the agent to choose an action randomly (for exploration) but with the respect of the
        probability given by the policy net
        @param probs: the probabilities returned by the model.
        @return: an action include in the action space.
        """
        p = np.random.random()
        if p < self.e:
            return np.random.choice(self.action_space)
        else:
            return np.argmax(probs)

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
        @return: the probabilities of taking an action for the state.
        """
        state = self.backbone(state)
        return self.head(state)


class DOT:
    """
    The class DOT (Detection of target). it is basically a policy gradient descent.
    """

    def __init__(self, environment, learning_rate=0.005, gamma=0.05,
                lr_gamma=0.7, pa_dataset_size=3000, pa_batch_size=100, epsilon=0.2):

        self.gamma = gamma
        self.environment = environment

        self.policy = PolicyNet(epsilon)

        self.pa_dataset_size = pa_dataset_size
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=lr_gamma)
        self.pa_batch_size = pa_batch_size

        # Past Actions Buffer
        self.S_pa_batch = None
        self.A_pa_batch = None
        self.G_pa_batch = None

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

    def update_policy(self):

        shuffle_index = torch.randperm(len(self.A_pa_batch))
        self.A_pa_batch = self.A_pa_batch[shuffle_index]
        self.G_pa_batch = self.G_pa_batch[shuffle_index]
        self.S_pa_batch = self.S_pa_batch[shuffle_index]

        if len(self.A_pa_batch) < self.pa_batch_size:
            return 0.

        S = self.S_pa_batch[:self.pa_batch_size]
        A = self.A_pa_batch[:self.pa_batch_size]
        G = self.G_pa_batch[:self.pa_batch_size]

        #S, A, G = batch

        # Calculate loss
        self.optimizer.zero_grad()
        action_probs = self.policy(S)

        log_probs = torch.log(action_probs)
        log_probs = torch.nan_to_num(log_probs)

        #selected_log_probs = torch.gather(log_probs, 1, A.unsqueeze(1))
        select_action = torch.gather(action_probs, 1, A.unsqueeze(1))
        #loss = - (G.unsqueeze(1) * selected_log_probs).mean()
        l2_lambda = 0.001
        l2_norm = sum(p.pow(2.0).sum() for p in self.policy.parameters())

        loss = torch.nn.functional.l1_loss(select_action.squeeze(), G.squeeze())
        #loss = loss + l2_lambda * l2_norm

        loss.backward(retain_graph=True)

        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def fit_one_episode(self, S):
        """
        this method allow the agent to be trained for one episode.
        @param S: The first state given by the environment.
        @return: the loss and the total reward during this episode.
        """

        # ------------------------------------------------------------------------------------------------------
        # EPISODE PREPARATION
        # ------------------------------------------------------------------------------------------------------
        S_batch = []
        R_batch = []
        A_batch = []

        # ------------------------------------------------------------------------------------------------------
        # EPISODE REALISATION
        # ------------------------------------------------------------------------------------------------------
        counter = 0
        sum_reward = 0

        counter += 1
        # State preprocess

        while True:  # while the state is not terminal.
            S = torch.from_numpy(S).float()
            S = S.unsqueeze(0).to(self.policy.device)
            S = self.policy.prepare_data(S)
            #  no need to go backward.
            with torch.no_grad():
                action_probs = self.policy(S)
                action_probs = action_probs.detach().cpu().numpy()[0]
                A = self.policy.follow_policy(action_probs)
                conf = action_probs[A]

            S_prime, R, is_terminal = self.environment.take_action(A, conf.item())

            # appending the state, the action and the reward to the batch.
            S_batch.append(S)
            A_batch.append(A)
            R_batch.append(R)
            sum_reward += R

            S = S_prime

            if is_terminal:
                break

        # ------------------------------------------------------------------------------------------------------
        # BATCH PREPARATION
        # ------------------------------------------------------------------------------------------------------
        # calculate the discounted reward.
        for i in reversed(range(1, len(R_batch))):
            R_batch[i - 1] += self.gamma * R_batch[i]

        S_batch = torch.concat(S_batch).to(self.policy.device)
        A_batch = torch.LongTensor(A_batch).to(self.policy.device)
        G_batch = torch.FloatTensor(R_batch).to(self.policy.device)

        # ------------------------------------------------------------------------------------------------------
        # PAST ACTION DATASET PREPARATION
        # ------------------------------------------------------------------------------------------------------

        # Append the past action batch to the current batch if possible
        #if self.A_pa_batch is not None and len(self.A_pa_batch) > self.pa_batch_size:
        #    batch = (torch.cat((self.S_pa_batch[0:self.pa_batch_size], S_batch), 0),
        #             torch.cat((self.A_pa_batch[0:self.pa_batch_size], A_batch), 0),
        #             torch.cat((self.G_pa_batch[0:self.pa_batch_size], G_batch), 0))
        #else:
        #    batch = (S_batch, A_batch, G_batch)

        # Add some experiences to the buffer with respect of TD error
        nb_new_memories = 50

        #idx = torch.randperm(len(A_batch))[:nb_new_memories]
        weights = G_batch + 1
        weights /= torch.sum(weights)
        idx = torch.multinomial(weights, nb_new_memories)

        if self.A_pa_batch is None:
            self.A_pa_batch = A_batch[idx]
            self.S_pa_batch = S_batch[idx]
            self.G_pa_batch = G_batch[idx]
        else:
            self.A_pa_batch = torch.cat((self.A_pa_batch, A_batch[idx]), 0)
            self.S_pa_batch = torch.cat((self.S_pa_batch, S_batch[idx]), 0)
            self.G_pa_batch = torch.cat((self.G_pa_batch, G_batch[idx]), 0)

        # clip the buffer if it's too big
        if len(self.A_pa_batch) > self.pa_dataset_size:
            # shuffling the batch

            # dataset clipping
            surplus = len(self.A_pa_batch) - self.pa_dataset_size
            _, self.A_pa_batch = torch.split(self.A_pa_batch, [surplus, self.pa_dataset_size])
            _, self.G_pa_batch = torch.split(self.G_pa_batch, [surplus, self.pa_dataset_size])
            _, self.S_pa_batch = torch.split(self.S_pa_batch, [surplus, self.pa_dataset_size])

        # ------------------------------------------------------------------------------------------------------
        # MODEL OPTIMISATION
        # ------------------------------------------------------------------------------------------------------
        loss = self.update_policy()

        return loss, sum_reward

    def exploit_one_episode(self, S):
        """
        this method allow the agent to exploit on the environment.
        @param S: the first state given by the environment.
        @return: the sum of rewards during the episode.
        """
        sum_reward = 0
        while True:
            # State preprocess

            S = torch.from_numpy(S).float()
            S = S.unsqueeze(0).to(self.policy.device)
            S = self.policy.prepare_data(S)

            with torch.no_grad():
                action_probs = self.policy(S)
                # no need to explore anymore.
                action_probs = action_probs.detach().cpu().numpy()[0]
                A = np.argmax(action_probs)
                conf = action_probs[A]

            S_prime, R, is_terminal = self.environment.take_action(A, conf)

            S = S_prime
            sum_reward += R

            if is_terminal:
                break

        return sum_reward
