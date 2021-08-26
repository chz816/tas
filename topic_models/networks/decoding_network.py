"""
This file is modified from the "contextualized_topic_models" package - decoding_network.py

PyTorch class for feed forward AVITM network.
"""

import torch
from torch import nn
from torch.nn import functional as F

from topic_models.networks.inference_network import CombinedInferenceNetwork, ContextualInferenceNetwork


class DecoderNetwork(nn.Module):
    """AVITM Network."""

    def __init__(self, vocab_size, bert_size, infnet, num_topics, model_type='prodLDA',
                 hidden_sizes=(100, 100), activation='softplus', dropout=0.2,
                 learn_priors=True):
        """
        Initialize InferenceNetwork.

        Args
            vocab_size : int, dimension of input
            num_topics : int, number of topic components, (default 10)
            model_type : string, 'prodLDA' or 'LDA' (default 'prodLDA')
            hidden_sizes : tuple, length = n_layers, (default (100, 100))
            activation : string, 'softplus', 'relu', (default 'softplus')
            learn_priors : bool, make priors learnable parameter
        """
        super(DecoderNetwork, self).__init__()
        assert isinstance(vocab_size, int), "vocab_size must by type int."
        assert isinstance(num_topics, int) and num_topics > 0, "num_topics must be type int > 0."
        assert model_type in ['prodLDA', 'LDA'], "model type must be 'prodLDA' or 'LDA'"
        assert isinstance(hidden_sizes, tuple), "hidden_sizes must be type tuple."
        assert activation in ['softplus', 'relu'], "activation must be 'softplus' or 'relu'."
        assert dropout >= 0, "dropout must be >= 0."

        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.model_type = model_type
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout
        self.learn_priors = learn_priors

        if infnet == "zeroshot":
            self.inf_net = ContextualInferenceNetwork(vocab_size, bert_size, num_topics, hidden_sizes, activation)
        elif infnet == "combined":
            self.inf_net = CombinedInferenceNetwork(vocab_size, bert_size, num_topics, hidden_sizes, activation)
        else:
            raise Exception('Missing infnet parameter, options are zeroshot and combined')

        # init prior parameters
        # \mu_1k = log \alpha_k + 1/K \sum_i log \alpha_i;
        # \alpha = 1 \forall \alpha
        topic_prior_mean = 0.0
        self.prior_mean = torch.tensor([topic_prior_mean] * num_topics)
        if torch.cuda.is_available():
            self.prior_mean = self.prior_mean.cuda()
        if self.learn_priors:
            self.prior_mean = nn.Parameter(self.prior_mean)

        # \Sigma_1kk = 1 / \alpha_k (1 - 2/K) + 1/K^2 \sum_i 1 / \alpha_k;
        # \alpha = 1 \forall \alpha
        topic_prior_variance = 1. - (1. / self.num_topics)
        self.prior_variance = torch.tensor(
            [topic_prior_variance] * num_topics)
        if torch.cuda.is_available():
            self.prior_variance = self.prior_variance.cuda()
        if self.learn_priors:
            self.prior_variance = nn.Parameter(self.prior_variance)

        self.topic_word = torch.Tensor(num_topics, vocab_size)
        if torch.cuda.is_available():
            self.topic_word = self.topic_word.cuda()
        self.topic_word = nn.Parameter(self.topic_word)
        nn.init.xavier_uniform_(self.topic_word)

        self.topic_word_batchnorm = nn.BatchNorm1d(vocab_size, affine=False)

        # dropout on h
        self.drop_h = nn.Dropout(p=self.dropout)

    @staticmethod
    def reparameterize(mu, logvar):
        """Reparameterize the h distribution."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x, latent_representation):
        """
        Forward pass
        - word_dist: batch_size x vocab_size
        """
        # batch_size x num_topics
        posterior_mu, posterior_log_sigma = self.inf_net(x, latent_representation)
        posterior_sigma = torch.exp(posterior_log_sigma)

        # generate samples from h - h.size: batch_size * num_topics
        h = F.softmax(self.reparameterize(posterior_mu, posterior_log_sigma), dim=1)
        h = self.drop_h(h)

        # prodLDA vs LDA
        if self.model_type == 'prodLDA':
            # in: batch_size x vocab_size x num_topics
            word_dist = F.softmax(self.topic_word_batchnorm(torch.matmul(h, self.topic_word)), dim=1)
        elif self.model_type == 'LDA':
            # simplex constrain on topic_word
            self.topic_word = F.softmax(self.topic_word_batchnorm(self.topic_word), dim=1)
            word_dist = torch.matmul(h, self.topic_word)

        return self.prior_mean, self.prior_variance, posterior_mu, posterior_sigma, posterior_log_sigma, word_dist, h

    def get_theta(self, x, latent_representation):
        with torch.no_grad():
            # batch_size x n_components
            posterior_mu, posterior_log_sigma = self.inf_net(x, latent_representation)
            posterior_sigma = torch.exp(posterior_log_sigma)

            # generate samples from theta
            theta = F.softmax(self.reparameterize(posterior_mu, posterior_log_sigma), dim=1)

            return theta
