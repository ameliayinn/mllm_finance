import torch
import torch.optim as optim
import numpy as np
from model import TextTimeSeriesDiffusion
from utils import q_sample_target


def train_forecast_diffusion(model, contexts, targets, text_list, tokenizer, sqrt_alphas_cumprod,
                             sqrt_one_minus_alphas_cumprod, num_epochs=5, batch_size=8, lr=1e-4, device='cpu',
                             save_checkpoint=True, ckpt_dir="checkpoints"):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse = torch.nn.MSELoss()

    N = len(contexts)
    indices = np.arange(N)

    if save_checkpoint:
        os.makedirs(ckpt_dir, exist_ok=True)

    def collate_fn(batch_indices):
        c_batch = contexts[batch_indices]
        t_batch = targets[batch_indices]
        text_batch = [text_list[i] for i in batch_indices]

        c_batch_tensor = torch.from_numpy(c_batch).float()
        t_batch_tensor = torch.from_numpy(t_batch).float()

        encoding = tokenizer(text_batch, padding=True, truncation=True, return_tensors="pt")
        return c_batch_tensor, t_batch_tensor, encoding

    model.train()
    loss_list = []

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_samples = 0
        for batch_idx in np.random.permutation(len(indices)):
            c_batch_tensor, t_batch_tensor, encoding = collate_fn(batch_idx)
            c_batch_tensor = c_batch_tensor.to(device)
            t_batch_tensor = t_batch_tensor.to(device)
            encoding = {k: v.to(device) for k, v in encoding.items()}

            t_rand = torch.randint(0, T, (c_batch_tensor.size(0),), device=device).long()
            noise = torch.randn_like(t_batch_tensor)

            x_t = q_sample_target(x0=t_batch_tensor, t=t_rand, noise=noise, sqrt_alphas_cumprod=sqrt_alphas_cumprod,
                                  sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod)
            noise_pred = model(c_batch_tensor, x_t, t_rand.unsqueeze(-1), encoding)
            loss = mse(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(indices)}")
        if save_checkpoint:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"model_epoch_{epoch + 1}.pt"))
