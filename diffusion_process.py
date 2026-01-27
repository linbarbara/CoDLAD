import torch
import numpy as np
import math
from enum import Enum

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ModelMeanType(Enum):
    """What the model is trained to predict."""
    PREVIOUS_X = "previous_x"  # Predict x_{t-1}
    START_X = "start_x"        # Predict x_0
    EPSILON = "epsilon"        # Predict noise (epsilon)


class ModelVarType(Enum):
    """How the model variance is handled."""
    LEARNED = "learned"
    FIXED_SMALL = "fixed_small"
    FIXED_LARGE = "fixed_large"


class LossType(Enum):
    """Loss type."""
    MSE = "mse"
    KL = "kl"


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """Get a named beta schedule."""
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    
    elif schedule_name == "sqrt":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: 1 - np.sqrt(t + 0.0001),
        )
    
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """Create a beta schedule from an alpha_bar function."""
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class GaussianDiffusion:
    """Full Gaussian diffusion process.

    Implements DDPM and DDIM sampling.
    """
    def __init__(
        self,
        betas,
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
        rescale_timesteps=False,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps
        
        # Use float64 for better numerical precision
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()
        
        self.num_timesteps = int(betas.shape[0])
        
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        
        # Compute diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # Compute posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )
    
    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        """Extract values from a 1-D numpy array into a broadcastable tensor."""
        res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)
    
    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion: sample x_t from x_0.
        q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        
        return (
            self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
    
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """Compute posterior q(x_{t-1} | x_t, x_0)."""
        posterior_mean = (
            self._extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def p_mean_variance(self, model, x_t, t, model_kwargs=None):
        """Use the model to predict the mean and variance of p(x_{t-1} | x_t)."""
        if model_kwargs is None:
            model_kwargs = {}
        
        B = x_t.shape[0]
        assert t.shape == (B,)
        
        # Model prediction
        model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)
        
        # Parse model output to obtain the predicted x_0
        if self.model_mean_type == ModelMeanType.EPSILON:
            # The model predicts noise
            pred_xstart = self._predict_xstart_from_eps(x_t=x_t, t=t, eps=model_output)
        elif self.model_mean_type == ModelMeanType.START_X:
            # The model directly predicts x_0
            pred_xstart = model_output
        else:
            raise NotImplementedError(self.model_mean_type)
        
        # Compute mean and variance
        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x_t, t=t
        )
        
        if self.model_var_type == ModelVarType.FIXED_SMALL:
            model_variance = self._extract_into_tensor(self.posterior_variance, t, x_t.shape)
            model_log_variance = self._extract_into_tensor(
                self.posterior_log_variance_clipped, t, x_t.shape
            )
        elif self.model_var_type == ModelVarType.FIXED_LARGE:
            model_variance = self._extract_into_tensor(self.betas, t, x_t.shape)
            model_log_variance = torch.log(model_variance)
        else:
            raise NotImplementedError(self.model_var_type)
        
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }
    
    def _predict_xstart_from_eps(self, x_t, t, eps):
        """Compute x_0 from the predicted noise."""
        return (
            self._extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )
    
    def _scale_timesteps(self, t):
        """Scale timesteps (optional rescaling to 1000-step convention)."""
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t
    
    def p_sample(self, model, x_t, t, model_kwargs=None):
        """
        One-step denoising sample: p(x_{t-1} | x_t).
        """
        out = self.p_mean_variance(model, x_t, t, model_kwargs=model_kwargs)
        noise = torch.randn_like(x_t)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        )
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}
    
    def p_sample_loop(self, model, shape, model_kwargs=None, noise=None):
        device = next(model.parameters()).device
        if noise is None:
            img = torch.randn(*shape, device=device)
        else:
            img = noise
        
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            with torch.no_grad():
                out = self.p_sample(model, img, t, model_kwargs=model_kwargs)
                img = out["sample"]
        
        return img
    
    def ddim_sample(self, model, x_t, t, t_next, model_kwargs=None, eta=0.0):
        out = self.p_mean_variance(model, x_t, t, model_kwargs=model_kwargs)
        pred_xstart = out["pred_xstart"]

        alpha_bar = self._extract_into_tensor(self.alphas_cumprod, t, x_t.shape)
        alpha_bar_prev = self._extract_into_tensor(self.alphas_cumprod, t_next, x_t.shape)
        
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        
        mean_pred = (
            pred_xstart * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma ** 2)
            * (x_t - pred_xstart * torch.sqrt(alpha_bar))
            / torch.sqrt(1 - alpha_bar)
        )
        
        noise = torch.randn_like(x_t)
        nonzero_mask = (
            (t_next != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        )
        
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": pred_xstart}
    
    def ddim_sample_loop(self, model, shape, model_kwargs=None, noise=None, ddim_timesteps=50, eta=0.0):
        device = next(model.parameters()).device
        if noise is None:
            img = torch.randn(*shape, device=device)
        else:
            img = noise
        
        timesteps = np.linspace(0, self.num_timesteps - 1, ddim_timesteps, dtype=np.int64)
        timesteps = torch.from_numpy(timesteps).to(device)
        
        for i in reversed(range(len(timesteps))):
            t = torch.full((shape[0],), timesteps[i], device=device, dtype=torch.long)
            t_next = torch.full((shape[0],), timesteps[i-1] if i > 0 else 0, device=device, dtype=torch.long)
            
            with torch.no_grad():
                out = self.ddim_sample(model, img, t, t_next, model_kwargs=model_kwargs, eta=eta)
                img = out["sample"]
        
        return img
    
    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        if model_kwargs is None:
            model_kwargs = {}
        
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_t = self.q_sample(x_start, t, noise=noise)
        
        model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)
        
        if self.model_mean_type == ModelMeanType.EPSILON:
            target = noise
        elif self.model_mean_type == ModelMeanType.START_X:
            target = x_start
        else:
            raise NotImplementedError(self.model_mean_type)
        
        # MSE损失
        if self.loss_type == LossType.MSE:
            loss = (target - model_output) ** 2
            loss = loss.mean()
        else:
            raise NotImplementedError(self.loss_type)
        
        return loss


def create_diffusion(
    num_diffusion_timesteps=1000,            
    noise_schedule="linear",
    model_mean_type="epsilon",
    model_var_type="fixed_small",
    loss_type="mse",
    rescale_timesteps=False,
):
    betas = get_named_beta_schedule(noise_schedule, num_diffusion_timesteps)
    
    mean_type = {
        "epsilon": ModelMeanType.EPSILON,
        "start_x": ModelMeanType.START_X,
    }[model_mean_type]
    
    var_type = {
        "fixed_small": ModelVarType.FIXED_SMALL,
        "fixed_large": ModelVarType.FIXED_LARGE,
    }[model_var_type]
    
    loss_t = {
        "mse": LossType.MSE,
        "kl": LossType.KL,
    }[loss_type]
    
    return GaussianDiffusion(
        betas=betas,
        model_mean_type=mean_type,
        model_var_type=var_type,
        loss_type=loss_t,
        rescale_timesteps=rescale_timesteps,
    )
