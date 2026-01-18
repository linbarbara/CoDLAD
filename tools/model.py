import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.nn.functional import binary_cross_entropy_with_logits as bce_logits
from torch.nn.functional import mse_loss as mse
from tools.dataprocess import *
from torch_geometric import data as DATA
from torch_geometric.nn import GCNConv, GINConv, GATConv, ChebConv, GAE, global_mean_pool, global_max_pool

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# class GraphEncoder(nn.Module):
#     def __init__(self, input_dim: int, output_dim: int, dropout=0.2):
#         super(GraphEncoder, self).__init__()
#         self.conv1 = GATConv(input_dim, input_dim, heads=10)
#         self.conv2 = GCNConv(input_dim*10, input_dim*10)
#         self.fc_g1 = torch.nn.Linear(input_dim*10, 256)
#         self.fc_g2 = torch.nn.Linear(256, output_dim)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, data: DATA.data):
#         x, edge_index = data.x, data.edge_index
#         x = self.conv1(x, edge_index)
#         x = self.relu(x)
#         x = self.conv2(x, edge_index)
#         x = self.relu(x)
#         x = self.fc_g1(x)
#         x = self.dropout(x)
#         x = self.fc_g2(x)
#         return x




class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalGCNEncoder, self).__init__()
        # self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True) # cached only for transductive learning
        # self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
        # self.conv_logstd = GCNConv(2 * out_channels, out_channels, cached=True)
        self.conv1 = ChebConv(in_channels, 2 * out_channels, K=2)
        self.bn1 = nn.BatchNorm1d(2 * out_channels)
        self.conv_mu = ChebConv(2 * out_channels, out_channels, K=2)
        self.conv_logstd = ChebConv(2 * out_channels, out_channels, K=2)
        self.relu = nn.ReLU()
    def forward(self, data: DATA.data, batch=None):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.relu(x)
        mu, logstd = self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
        # mu = global_mean_pool(mu, batch=batch)
        # logstd = global_mean_pool(logstd, batch=batch)
        return mu, logstd


class gEncoder(nn.Module):
    def __init__(self, input_dim:int, output_dim:int, dropout=0.2):
        super(gEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.BatchNorm1d(input_dim//2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(input_dim//2, output_dim)
        )
    def forward(self, x):
        return self.net(x)


class gDecoder(nn.Module):
    def __init__(self, recon_dim:int, emb_dim:int, dropout=0.2):
        super(gDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, emb_dim*2),
            nn.BatchNorm1d(emb_dim*2),
            nn.ReLU(),
            # nn.Dropout(p=dropout),
            nn.Linear(emb_dim*2, recon_dim)
        )
    def forward(self, x):
        return self.net(x)


class GraphEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout=0.2):
        super(GraphEncoder, self).__init__()
        # self.conv1 = GATConv(input_dim, input_dim, heads=10)
        # self.conv2 = GCNConv(input_dim*10, input_dim*10)
        # self.fc_g1 = torch.nn.Linear(input_dim*10, 256)
        # self.fc_g2 = torch.nn.Linear(256, output_dim)
        self.conv1 = ChebConv(in_channels=input_dim, out_channels=128, K=2)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = ChebConv(in_channels=128, out_channels=128, K=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc_g1 = torch.nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc_g2 = torch.nn.Linear(64, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data: DATA.data, batch=None):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.fc_g1(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc_g2(x)
        x_mean = global_mean_pool(x, batch=batch)
        # x_mean = global_max_pool(x, batch=batch)
        return x, x_mean


class GraphDecoder(nn.Module):
    def __init__(self, recon_dim: int, emb_dim: int, dropout=0.2):
        super(GraphDecoder, self).__init__()
        self.fc_g1 = torch.nn.Linear(emb_dim, 1024)
        self.fc_g2 = torch.nn.Linear(1024, recon_dim)
        # self.conv1 = GCNConv(recon_dim, recon_dim)
        self.conv1 = ChebConv(recon_dim, recon_dim, K=2)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index:torch.Tensor):
        x = self.fc_g1(x)
        x = self.relu(x)
        x = self.fc_g2(x)
        x = self.relu(x)
        x = self.conv1(x, edge_index)
        # x = self.relu(x)
        # x = self.conv1(x)
        return x


class Edgeindexdecoder(nn.Module):
    def __init__(self, input_dim:int):
        super(Edgeindexdecoder, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, input_dim)
        self.fc2 = torch.nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.2)

    def forward(self, edge_index:torch.tensor):
        edge_index = self.fc1(edge_index)
        edge_index = self.relu(edge_index)
        edge_index = self.drop(edge_index)
        edge_index = self.fc2(edge_index)
        return edge_index



class Classify(nn.Module):
    def __init__(self, input_dim):
        super(Classify, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Linear(10, 1)
            # nn.Sigmoid()
        )

    def forward(self, x):
        return (self.net(x)).view(-1)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            # nn.BatchNorm1d(input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        # return (self.net(x)).view(-1)
        return self.net(x)

class VAE_Encoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(VAE_Encoder, self).__init__()
        self.linear = torch.nn.Linear(input_size, hidden_size)
        self.mu = torch.nn.Linear(hidden_size, latent_size)
        self.sigma = torch.nn.Linear(hidden_size, latent_size)
    def forward(self, x):# x: bs,input_size
        x = F.relu(self.linear(x)) #-> bs,hidden_size
        mu = self.mu(x) #-> bs,latent_size
        sigma = self.sigma(x)#-> bs,latent_size
        return mu,sigma

class VAE_Decoder(torch.nn.Module):
    def __init__(self, latent_size, hidden_size, output_size):
        super(VAE_Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(latent_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, output_size)
    def forward(self, x): # x:bs,latent_size
        x = F.relu(self.linear1(x)) #->bs,hidden_size
        # x = torch.sigmoid(self.linear2(x)) #->bs,output_size
        x = self.linear2(x)
        return x

class VAE(torch.nn.Module):
    def __init__(self, input_size, output_size, latent_size, hidden_size):
        super(VAE, self).__init__()
        self.encoder = VAE_Encoder(input_size, hidden_size, latent_size)
        self.decoder = VAE_Decoder(latent_size, hidden_size, output_size)


        # self.decoder1 = VAE_Decoder(64, 128, 1426)

    def forward(self, x): #x: bs,input_size
        mu,sigma = self.encoder(x) #mu,sigma: bs,latent_size
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)  #eps: bs,latent_size
        z = mu + eps*sigma  #z: bs,latent_size
        re_x = self.decoder(z) # re_x: bs,output_size
        return re_x,z,mu,sigma

def vaeloss(mu, sigma, re_x, x, alpha=0.1):
    mseloss = torch.nn.MSELoss()
    recon_loss = mseloss(re_x, x)
    KLD = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
    loss = alpha * KLD + recon_loss
    return loss

def info_nce_loss(z, labels, temperature=0.5, device='cuda', num_classes=None):
    batch_size = z.size(0)
    z = F.normalize(z, dim=1)
    sim_matrix = torch.matmul(z, z.T) / temperature
    
    if num_classes is None:
        num_classes = len(torch.unique(labels))
    if num_classes <= 1:
        print("Warning: Only one class in batch, skipping InfoNCE loss")
        return torch.tensor(0.0, requires_grad=True).to(device)
    
    labels_one_hot = F.one_hot(labels, num_classes=num_classes).float().to(device)
    positive_mask = torch.matmul(labels_one_hot, labels_one_hot.T)
    
    logits_mask = 1 - torch.eye(batch_size).to(device)
    exp_logits = torch.exp(sim_matrix) * logits_mask
    log_prob = sim_matrix - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
    
    mean_log_prob_pos = (positive_mask * log_prob).sum(1) / (positive_mask.sum(1) + 1e-8)
    loss = -mean_log_prob_pos.mean()
    
    return loss

def compute_samples_per_cls(data_loader, unique_labels, device='cuda'):
    label_counts = torch.zeros(len(unique_labels)).to(device)
    for _, labels in data_loader:
        labels = labels.to(device)
        label_counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float))
    return label_counts.cpu().numpy()

def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    """
    BCLoss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="none")
    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits)))
    loss = modulator * BCLoss
    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)
    focal_loss /= torch.sum(labels)
    return focal_loss

def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma, device='cuda'):
    weights = np.ones(no_of_classes, dtype=np.float32)
    
    non_zero_mask = samples_per_cls > 0
    effective_num = np.ones(no_of_classes, dtype=np.float32)
    effective_num[non_zero_mask] = 1.0 - np.power(beta, samples_per_cls[non_zero_mask])
    effective_num = np.where(effective_num == 0, 1e-6, effective_num)
    
    weights[non_zero_mask] = (1.0 - beta) / effective_num[non_zero_mask]
    weights_sum = np.sum(weights[non_zero_mask]) if np.sum(non_zero_mask) > 0 else 1.0
    weights = weights / (weights_sum + 1e-6) * no_of_classes
    
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    labels_one_hot = F.one_hot(labels, no_of_classes).float().to(device)

    weights = weights.unsqueeze(0).repeat(labels_one_hot.shape[0], 1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1).repeat(1, no_of_classes)
    
    if loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels_one_hot, weight=weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim=1)
        cb_loss = F.binary_cross_entropy(input=pred, target=labels_one_hot, weight=weights)

    return cb_loss

def cb_loss(labels, logits, samples_per_cls, unique_labels, loss_type="focal", beta=0.9999, gamma=2.0, device='cuda'):
    no_of_classes = len(unique_labels)

    if logits.size(1) != no_of_classes:
        linear_layer = nn.Linear(logits.size(1), no_of_classes).to(device)
        logits = linear_layer(logits) 
    
    cb_loss_value = CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma)
    
    return cb_loss_value
def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
