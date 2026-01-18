import os
import math
import copy
import itertools
import argparse
from collections import defaultdict

import torch
import torch.nn as nn
import torch.autograd as autograd
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from data import *  
from tools.dataprocess import *  
from tools.model import * 
from diffusion_process import create_diffusion 

device = "cuda" if torch.cuda.is_available() else "cpu"

def safemakedirs(folder: str):
    os.makedirs(folder, exist_ok=True)


def append_file(filename: str, data_dict: dict):
    import json
    with open(filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(data_dict) + "\n")


def ortho_loss(shared_z: torch.Tensor, private_z: torch.Tensor) -> torch.Tensor:
    s_l2_norm = torch.norm(shared_z, p=2, dim=1, keepdim=True).detach()
    s_l2 = shared_z.div(s_l2_norm.expand_as(shared_z) + 1e-6)
    p_l2_norm = torch.norm(private_z, p=2, dim=1, keepdim=True).detach()
    p_l2 = private_z.div(p_l2_norm.expand_as(private_z) + 1e-6)
    return torch.mean((s_l2.t().mm(p_l2)).pow(2))


def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    alpha = torch.rand((real_samples.shape[0], 1), device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    critic_interpolates = critic(interpolates)
    fakes = torch.ones((real_samples.shape[0], 1), device=device)

    gradients = autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=fakes,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

class LatentDiffusionModel(nn.Module):
    def __init__(self, latent_dim=32, hidden_dim=256, num_layers=3, num_classes=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.label_embed = nn.Linear(1, hidden_dim)

        self.input_proj = nn.Linear(latent_dim, hidden_dim)

        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
        self.mlp = nn.Sequential(*layers)

        self.output_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z_noisy, timesteps, label=None):
        t_emb = self.time_embed(timestep_embedding(timesteps, self.hidden_dim))

        if label is not None and self.label_embed is not None:
            label_emb = self.label_embed(label)
            cond = t_emb + label_emb
        else:
            cond = t_emb

        h = self.input_proj(z_noisy)
        h = h + cond
        h = self.mlp(h)
        h = self.output_proj(h)
        return h

def train_vae_discriminator(
    s_batch, t_batch,
    shared_vae, source_private_vae, target_private_vae,
    discrim, optimizer, scheduler
):
    loss_log = defaultdict(float)

    shared_vae.zero_grad(set_to_none=True)
    source_private_vae.zero_grad(set_to_none=True)
    target_private_vae.zero_grad(set_to_none=True)
    discrim.zero_grad(set_to_none=True)

    shared_vae.eval()
    source_private_vae.eval()
    target_private_vae.eval()
    discrim.train()

    optimizer.zero_grad(set_to_none=True)

    with torch.no_grad():
        _, _, ccle_mu, _ = shared_vae(s_batch)
        _, _, tcga_mu, _ = shared_vae(t_batch)
        _, _, pccle_mu, _ = source_private_vae(s_batch)
        _, _, ptcga_mu, _ = target_private_vae(t_batch)

        zs = torch.cat([ccle_mu, pccle_mu], dim=1)  # real (source)
        zt = torch.cat([tcga_mu, ptcga_mu], dim=1)  # fake (target)

    d_loss = torch.mean(discrim(zt)) - torch.mean(discrim(zs))
    g_p = compute_gradient_penalty(
        critic=discrim,
        real_samples=zs,
        fake_samples=zt,
        device=device
    )

    loss_log.update({
        "discrim_loss": float(d_loss.item()),
        "g_p": float(g_p.item()),
    })

    d_loss = d_loss + 10.0 * g_p
    d_loss.backward()
    optimizer.step()
    scheduler.step()

    discrim.eval()
    return loss_log

def train_vae_with_latent_diffusion(
    s_batch, t_batch, s_labels, t_labels,
    shared_vae, source_private_vae, target_private_vae,
    latent_diffusion, diffusion_process,
    discrim, optimizer, scheduler,
    unique_labels,
    samples_per_cls=None,
    device="cuda",
):
    loss_log = defaultdict(float)

    shared_vae.train()
    source_private_vae.train()
    target_private_vae.train()
    latent_diffusion.train()
    discrim.eval()

    optimizer.zero_grad(set_to_none=True)

    s_labels_float = s_labels.float().unsqueeze(1) if s_labels.dim() == 1 else s_labels.float()
    t_labels_float = t_labels.float().unsqueeze(1) if t_labels.dim() == 1 else t_labels.float()

    ccle_re_x, _, ccle_mu, ccle_sigma = shared_vae(s_batch)
    tcga_re_x, _, tcga_mu, tcga_sigma = shared_vae(t_batch)
    pccle_re_x, _, pccle_mu, pccle_sigma = source_private_vae(s_batch)
    ptcga_re_x, _, ptcga_mu, ptcga_sigma = target_private_vae(t_batch)

    ccle_vae_loss = vaeloss(ccle_mu, ccle_sigma, ccle_re_x, s_batch)
    tcga_vae_loss = vaeloss(tcga_mu, tcga_sigma, tcga_re_x, t_batch)
    pccle_vae_loss = vaeloss(pccle_mu, pccle_sigma, pccle_re_x, s_batch)
    ptcga_vae_loss = vaeloss(ptcga_mu, ptcga_sigma, ptcga_re_x, t_batch)

    vae_loss = ccle_vae_loss + tcga_vae_loss
    pvae_loss = pccle_vae_loss + ptcga_vae_loss
    o_loss = ortho_loss(ccle_mu, pccle_mu) + ortho_loss(tcga_mu, ptcga_mu)

    batch_size = s_batch.size(0)
    t_steps = torch.randint(0, diffusion_process.num_timesteps, (batch_size,), device=device).long()

    noise_s = torch.randn_like(ccle_mu)
    noise_t = torch.randn_like(tcga_mu)

    z_noisy_s = diffusion_process.q_sample(ccle_mu, t_steps, noise=noise_s)
    z_noisy_t = diffusion_process.q_sample(tcga_mu, t_steps, noise=noise_t)

    pred_noise_s = latent_diffusion(z_noisy_s, t_steps, label=s_labels_float)
    pred_noise_t = latent_diffusion(z_noisy_t, t_steps, label=t_labels_float)

    diffusion_loss_s = nn.functional.mse_loss(pred_noise_s, noise_s)
    diffusion_loss_t = nn.functional.mse_loss(pred_noise_t, noise_t)
    diffusion_loss = (diffusion_loss_s + diffusion_loss_t) / 2.0

    zs = torch.cat((ccle_mu, pccle_mu), dim=1)
    zt = torch.cat((tcga_mu, ptcga_mu), dim=1)
    g_loss = -torch.mean(discrim(zt))

    z = torch.cat((ccle_mu, tcga_mu), dim=0)
    labels = torch.cat((s_labels, t_labels), dim=0)

    max_label = labels.max().item()
    num_classes = max(int(max_label + 1), len(unique_labels))
    nce_loss = info_nce_loss(z, labels, temperature=0.5, device=device, num_classes=num_classes)

    if samples_per_cls is None:
        raise ValueError("samples_per_cls is required for cb_loss in GAN stage.")
    ccle_cb_loss = cb_loss(s_labels, ccle_mu, np.array(samples_per_cls["ccle"]), unique_labels)
    tcga_cb_loss = cb_loss(t_labels, tcga_mu, np.array(samples_per_cls["tcga"]), unique_labels)
    cb_loss_value = (ccle_cb_loss + tcga_cb_loss) / 2.0

    total_loss = (
        vae_loss * 1.0 +
        pvae_loss * 1.0 +
        o_loss * 0.5 +
        diffusion_loss * 0.5 +
        g_loss * 0.3 +
        nce_loss * 0.5 +
        cb_loss_value * 0.3
    )

    loss_log.update({
        "vae_loss": float(vae_loss.item()),
        "pvae_loss": float(pvae_loss.item()),
        "ortho_loss": float(o_loss.item()),
        "diffusion_loss": float(diffusion_loss.item()),
        "gen_loss": float(g_loss.item()),
        "total_loss": float(total_loss.item()),
    })

    total_loss.backward()
    optimizer.step()
    scheduler.step()

    return loss_log


def pretrain_vae_latent_diffusion(sourcedata, targetdata, param, parent_folder, batch_size=64):
    """VAE + latent diffusion pretraining (ablation: no prototype loss)"""
    print("Start VAE + latent diffusion pretraining (ablation: no prototype loss)")

    # Params
    pretrain_epochs = param["pretrain_num_epochs"]
    gan_epochs = param["train_num_epochs"]
    pretrain_lr = param["pretrain_learning_rate"]
    gan_lr = param["gan_learning_rate"]
    unique_labels = param["unique_labels"]

    vae_latent_dim = 32

    set_dir_name = (
        f"pt_epochs_{pretrain_epochs},"
        f"t_epochs_{gan_epochs},"
        f"Ptlr_{pretrain_lr},"
        f"tlr{gan_lr}_vae_latent_diffusion_ablation_no_proto"
    )
    pretrain_dir = os.path.join(parent_folder, set_dir_name)
    safemakedirs(pretrain_dir)

    trainloss_logfile = os.path.join(pretrain_dir, "pretrain_losslog.txt")
    evalloss_logfile = os.path.join(pretrain_dir, "pretrain_eval_losslog.txt")
    dloss_logfile = os.path.join(pretrain_dir, "d_losslog.txt")
    genloss_logfile = os.path.join(pretrain_dir, "g_losslog.txt")

    trainloss_logdict = defaultdict(float)
    evalloss_logdict = defaultdict(float)

    # Data (must include Test Set)
    sourcetrainloader, sourcetest = sourcedata
    targettrainloader, targettest = targetdata

    # CB loss counts
    samples_per_cls = {
        "ccle": compute_samples_per_cls(sourcetrainloader, unique_labels),
        "tcga": compute_samples_per_cls(targettrainloader, unique_labels),
    }

    # Models
    shared_vae = VAE(input_size=1426, output_size=1426, latent_size=vae_latent_dim, hidden_size=128).to(device)
    source_private_vae = VAE(input_size=1426, output_size=1426, latent_size=vae_latent_dim, hidden_size=128).to(device)
    target_private_vae = VAE(input_size=1426, output_size=1426, latent_size=vae_latent_dim, hidden_size=128).to(device)

    latent_diffusion = LatentDiffusionModel(
        latent_dim=vae_latent_dim,
        hidden_dim=256,
        num_layers=3,
        num_classes=len(unique_labels),
    ).to(device)

    diffusion_process = create_diffusion(num_diffusion_timesteps=500)
    discrim = Discriminator(input_dim=vae_latent_dim * 2).to(device)

    if pretrain_epochs > 0:
        print("Stage 1: Pretrain VAE + latent diffusion")

        pretrain_optimizer = torch.optim.Adam(
            list(shared_vae.parameters()) +
            list(source_private_vae.parameters()) +
            list(target_private_vae.parameters()) +
            list(latent_diffusion.parameters()),
            lr=pretrain_lr,
        )
        pretrain_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(pretrain_optimizer, pretrain_epochs)

        tolerance = 0
        max_tolerance = 50
        min_eval_loss = float("inf")
        best_pretrain_state = None

        for epoch in range(pretrain_epochs):
            if epoch % 20 == 0:
                print(f"pretrain epoch: {epoch}")

            shared_vae.train()
            source_private_vae.train()
            target_private_vae.train()
            latent_diffusion.train()

            train_metrics = defaultdict(float)
            steps = 0

            for (sdata, slabels), (tdata, tlabels) in zip(sourcetrainloader, targettrainloader):
                sdata, tdata = sdata.to(device), tdata.to(device)
                slabels, tlabels = slabels.to(device), tlabels.to(device)

                slabels_float = slabels.float().unsqueeze(1) if slabels.dim() == 1 else slabels.float()
                tlabels_float = tlabels.float().unsqueeze(1) if tlabels.dim() == 1 else tlabels.float()

                pretrain_optimizer.zero_grad(set_to_none=True)

                ccle_re_x, _, ccle_mu, ccle_sigma = shared_vae(sdata)
                tcga_re_x, _, tcga_mu, tcga_sigma = shared_vae(tdata)
                pccle_re_x, _, pccle_mu, pccle_sigma = source_private_vae(sdata)
                ptcga_re_x, _, ptcga_mu, ptcga_sigma = target_private_vae(tdata)

                vae_loss = (
                    vaeloss(ccle_mu, ccle_sigma, ccle_re_x, sdata) +
                    vaeloss(tcga_mu, tcga_sigma, tcga_re_x, tdata)
                )
                pvae_loss = (
                    vaeloss(pccle_mu, pccle_sigma, pccle_re_x, sdata) +
                    vaeloss(ptcga_mu, ptcga_sigma, ptcga_re_x, tdata)
                )
                o_loss = ortho_loss(ccle_mu, pccle_mu) + ortho_loss(tcga_mu, ptcga_mu)

                # diffusion denoise loss
                t_steps = torch.randint(0, 500, (sdata.size(0),), device=device).long()
                noise_s = torch.randn_like(ccle_mu)
                noise_t = torch.randn_like(tcga_mu)
                z_noisy_s = diffusion_process.q_sample(ccle_mu, t_steps, noise=noise_s)
                z_noisy_t = diffusion_process.q_sample(tcga_mu, t_steps, noise=noise_t)

                pred_noise_s = latent_diffusion(z_noisy_s, t_steps, label=slabels_float)
                pred_noise_t = latent_diffusion(z_noisy_t, t_steps, label=tlabels_float)
                diffusion_loss = (
                    nn.functional.mse_loss(pred_noise_s, noise_s) +
                    nn.functional.mse_loss(pred_noise_t, noise_t)
                ) / 2.0

                loss = vae_loss + pvae_loss + 0.5 * o_loss + 0.5 * diffusion_loss
                loss.backward()
                pretrain_optimizer.step()
                pretrain_scheduler.step()

                train_metrics["vae_loss"] += float(vae_loss.item())
                train_metrics["pvae_loss"] += float(pvae_loss.item())
                train_metrics["ortho_loss"] += float(o_loss.item())
                train_metrics["diffusion_loss"] += float(diffusion_loss.item())
                steps += 1

            # train log
            trainloss_logdict["epoch"] = epoch
            for k, v in train_metrics.items():
                trainloss_logdict[k] = v / max(1, steps)
            append_file(trainloss_logfile, trainloss_logdict)

            # eval
            shared_vae.eval()
            source_private_vae.eval()
            target_private_vae.eval()
            latent_diffusion.eval()

            ccle_test_loader = DataLoader(sourcetest, batch_size=batch_size, shuffle=False)
            tcga_test_loader = DataLoader(targettest, batch_size=batch_size, shuffle=False)

            eval_metrics = defaultdict(float)
            eval_batches = 0

            with torch.no_grad():
                for sdata, slabels in ccle_test_loader:
                    sdata, slabels = sdata.to(device), slabels.to(device)
                    slabels_float = slabels.float().unsqueeze(1) if slabels.dim() == 1 else slabels.float()

                    ccle_re_x, _, ccle_mu, ccle_sigma = shared_vae(sdata)
                    pccle_re_x, _, pccle_mu, pccle_sigma = source_private_vae(sdata)

                    eval_metrics["vae_loss"] += float(vaeloss(ccle_mu, ccle_sigma, ccle_re_x, sdata).item())
                    eval_metrics["pvae_loss"] += float(vaeloss(pccle_mu, pccle_sigma, pccle_re_x, sdata).item())

                    t_steps = torch.randint(0, 500, (sdata.size(0),), device=device).long()
                    noise = torch.randn_like(ccle_mu)
                    z_noisy = diffusion_process.q_sample(ccle_mu, t_steps, noise=noise)
                    pred = latent_diffusion(z_noisy, t_steps, label=slabels_float)
                    eval_metrics["diffusion_loss"] += float(nn.functional.mse_loss(pred, noise).item())
                    eval_batches += 1

                for tdata, tlabels in tcga_test_loader:
                    tdata, tlabels = tdata.to(device), tdata.to(device) if False else (tdata.to(device), tlabels.to(device))
                    tlabels_float = tlabels.float().unsqueeze(1) if tlabels.dim() == 1 else tlabels.float()

                    tcga_re_x, _, tcga_mu, tcga_sigma = shared_vae(tdata)
                    ptcga_re_x, _, ptcga_mu, ptcga_sigma = target_private_vae(tdata)

                    eval_metrics["vae_loss"] += float(vaeloss(tcga_mu, tcga_sigma, tcga_re_x, tdata).item())
                    eval_metrics["pvae_loss"] += float(vaeloss(ptcga_mu, ptcga_sigma, ptcga_re_x, tdata).item())

                    t_steps = torch.randint(0, 500, (tdata.size(0),), device=device).long()
                    noise = torch.randn_like(tcga_mu)
                    z_noisy = diffusion_process.q_sample(tcga_mu, t_steps, noise=noise)
                    pred = latent_diffusion(z_noisy, t_steps, label=tlabels_float)
                    eval_metrics["diffusion_loss"] += float(nn.functional.mse_loss(pred, noise).item())
                    eval_batches += 1

            for k in eval_metrics:
                eval_metrics[k] /= max(1, eval_batches)

            evalloss_logdict["epoch"] = epoch
            evalloss_logdict.update(eval_metrics)
            append_file(evalloss_logfile, evalloss_logdict)

            # early stop by eval loss
            evalloss = eval_metrics["vae_loss"] + eval_metrics["pvae_loss"] + eval_metrics["diffusion_loss"]
            if evalloss < min_eval_loss:
                min_eval_loss = evalloss
                tolerance = 0
                best_pretrain_state = {
                    "shared_vae": copy.deepcopy(shared_vae.state_dict()),
                    "source_private_vae": copy.deepcopy(source_private_vae.state_dict()),
                    "target_private_vae": copy.deepcopy(target_private_vae.state_dict()),
                    "latent_diffusion": copy.deepcopy(latent_diffusion.state_dict()),
                }
            else:
                tolerance += 1

            if tolerance >= max_tolerance:
                print(f"Pretrain early stop at epoch {epoch}")
                break

        if best_pretrain_state is not None:
            shared_vae.load_state_dict(best_pretrain_state["shared_vae"])
            source_private_vae.load_state_dict(best_pretrain_state["source_private_vae"])
            target_private_vae.load_state_dict(best_pretrain_state["target_private_vae"])
            latent_diffusion.load_state_dict(best_pretrain_state["latent_diffusion"])

        torch.save(shared_vae.state_dict(), os.path.join(pretrain_dir, "pretrain_shared_vae.pth"))
        torch.save(source_private_vae.state_dict(), os.path.join(pretrain_dir, "pretrain_source_private_vae.pth"))
        torch.save(target_private_vae.state_dict(), os.path.join(pretrain_dir, "pretrain_target_private_vae.pth"))
        torch.save(latent_diffusion.state_dict(), os.path.join(pretrain_dir, "pretrain_latent_diffusion.pth"))

    if not os.path.exists(os.path.join(pretrain_dir, "after_traingan_shared_vae.pth")):
        print("Stage 2: GAN adversarial training (WGAN-GP)")

        # Generator optimizer (VAE + diffusion)
        vae_diffusion_optimizer = torch.optim.AdamW(
            list(shared_vae.parameters()) +
            list(source_private_vae.parameters()) +
            list(target_private_vae.parameters()) +
            list(latent_diffusion.parameters()),
            lr=gan_lr,
        )
        vae_diffusion_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(vae_diffusion_optimizer, gan_epochs)

        # Critic optimizer (WGAN)
        discrim_optimizer = torch.optim.RMSprop(discrim.parameters(), lr=gan_lr)
        discrim_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(discrim_optimizer, gan_epochs)

        gan_best_loss = float("inf")
        gan_tolerance = 0
        gan_max_tolerance = 20
        best_gan_state = None

        for epoch in range(gan_epochs):
            if epoch % 10 == 0:
                print(f"confounder wgan training epoch {epoch}")

            dloss_list = []
            genloss_list = []

            for i, ((sdata, slabels), (tdata, tlabels)) in enumerate(zip(sourcetrainloader, targettrainloader)):
                sdata, tdata = sdata.to(device), tdata.to(device)
                slabels, tlabels = slabels.to(device), tlabels.to(device)

                # Critic step
                dloss_log = train_vae_discriminator(
                    s_batch=sdata, t_batch=tdata,
                    shared_vae=shared_vae,
                    source_private_vae=source_private_vae,
                    target_private_vae=target_private_vae,
                    discrim=discrim,
                    optimizer=discrim_optimizer,
                    scheduler=discrim_scheduler,
                )
                dloss_list.append(dloss_log)

                # Generator step every 5 iters
                if (i + 1) % 5 == 0:
                    genloss_log = train_vae_with_latent_diffusion(
                        s_batch=sdata, t_batch=tdata,
                        s_labels=slabels, t_labels=tlabels,
                        shared_vae=shared_vae,
                        source_private_vae=source_private_vae,
                        target_private_vae=target_private_vae,
                        latent_diffusion=latent_diffusion,
                        diffusion_process=diffusion_process,
                        discrim=discrim,
                        optimizer=vae_diffusion_optimizer,
                        scheduler=vae_diffusion_scheduler,
                        unique_labels=unique_labels,
                        samples_per_cls=samples_per_cls,
                        device=device,
                    )
                    genloss_list.append(genloss_log)

            # epoch mean
            dloss_mean = defaultdict(float)
            if dloss_list:
                for k in dloss_list[0]:
                    dloss_mean[k] = sum(d[k] for d in dloss_list) / len(dloss_list)

            genloss_mean = defaultdict(float)
            if genloss_list:
                for k in genloss_list[0]:
                    genloss_mean[k] = sum(d[k] for d in genloss_list) / len(genloss_list)

            append_file(dloss_logfile, dloss_mean)
            append_file(genloss_logfile, genloss_mean)

            current_loss = sum(dloss_mean.values()) + sum(genloss_mean.values())

            if current_loss < gan_best_loss:
                gan_best_loss = current_loss
                gan_tolerance = 0
                best_gan_state = {
                    "shared_vae": copy.deepcopy(shared_vae.state_dict()),
                    "source_private_vae": copy.deepcopy(source_private_vae.state_dict()),
                    "target_private_vae": copy.deepcopy(target_private_vae.state_dict()),
                    "latent_diffusion": copy.deepcopy(latent_diffusion.state_dict()),
                }
            else:
                gan_tolerance += 1

            if gan_tolerance >= gan_max_tolerance:
                print(f"GAN train early stop in epoch: {epoch}")
                break

        if best_gan_state is not None:
            shared_vae.load_state_dict(best_gan_state["shared_vae"])
            source_private_vae.load_state_dict(best_gan_state["source_private_vae"])
            target_private_vae.load_state_dict(best_gan_state["target_private_vae"])
            latent_diffusion.load_state_dict(best_gan_state["latent_diffusion"])

        torch.save(shared_vae.state_dict(), os.path.join(pretrain_dir, "after_traingan_shared_vae.pth"))
        torch.save(source_private_vae.state_dict(), os.path.join(pretrain_dir, "after_traingan_source_private_vae.pth"))
        torch.save(target_private_vae.state_dict(), os.path.join(pretrain_dir, "after_traingan_target_private_vae.pth"))
        torch.save(latent_diffusion.state_dict(), os.path.join(pretrain_dir, "after_traingan_latent_diffusion.pth"))

def main_pretrain(i):
    params_grid = {
        "pretrain_num_epochs": [0, 100, 300],
        'pretrain_learning_rate': [0.001],
        'gan_learning_rate': [0.001],
        "train_num_epochs": [100, 200, 300, 500, 750, 1000, 1500, 2000, 2500, 3000]
    }
    keys, values = zip(*params_grid.items())
    update_params_dict_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
    sourcepretrain, targetpretrain = pretrain_data()
    pretrain_path = os.path.join('./result/pretrain_vae_latent_diffusion','pretrain'+str(i))
    safemakedirs(pretrain_path)
    for param_dict in update_params_dict_list:
        pretrain_vae_latent_diffusion(
                sourcepretrain,
                targetpretrain,
                param=param_dict,
                parent_folder=args.outfolder,
                batch_size=batch_size,
            )

if __name__ == "__main__":
    for i in range(10):
        parser = argparse.ArgumentParser("VAE + Latent Diffusion Pretraining - Ablation: No Prototype Loss (Clean)")
        parser.add_argument(
            "--outfolder",
            dest="outfolder",
            default=f"./result/pretrain_vae_latent_diffusion/pretrain{i}",
            type=str,
            help="choose the output folder",
        )
        parser.add_argument("--source", dest="source", default=None, type=str, help=".csv file address for the source")
        parser.add_argument("--target", dest="target", default=None, type=str, help=".csv file address for the target")
        args = parser.parse_args()

        params_grid = {
            "pretrain_num_epochs": [0, 100, 300],
            "pretrain_learning_rate": [0.001],
            "gan_learning_rate": [0.001],
            "train_num_epochs": [100, 200, 300, 500, 750, 1000, 1500, 2000, 2500, 3000],
        }
        keys, values = zip(*params_grid.items())
        update_params_dict_list = [dict(zip(keys, v)) for v in itertools.product(*values)]

        safemakedirs(args.outfolder)

        sourcepretrain, targetpretrain, unique_labels, batch_size = pretrain_data()
        for param_dict in update_params_dict_list:
            param_dict["unique_labels"] = unique_labels
            pretrain_vae_latent_diffusion(
                sourcepretrain,
                targetpretrain,
                param=param_dict,
                parent_folder=args.outfolder,
                batch_size=batch_size,
            )
