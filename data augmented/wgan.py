import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class GeneDataset(Dataset):
    """Custom Dataset for gene expression data"""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class Generator(nn.Module):
    """Generator network for WGAN"""

    def __init__(self, latent_dim, output_dim, num_classes):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )

    def forward(self, z, labels):
        # Concatenate noise and label embedding
        label_input = self.label_emb(labels)
        gen_input = torch.cat((z, label_input), -1)
        return self.model(gen_input)


class Critic(nn.Module):
    """Critic network for WGAN (instead of Discriminator)"""

    def __init__(self, input_dim, num_classes):
        super(Critic, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.Linear(input_dim + num_classes, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            # No sigmoid activation for WGAN
        )

    def forward(self, x, labels):
        # Concatenate input and label embedding
        label_input = self.label_emb(labels)
        critic_input = torch.cat((x, label_input), -1)
        return self.model(critic_input)


def compute_gradient_penalty(critic, real_samples, fake_samples, labels, device):
    """Calculates the gradient penalty for WGAN-GP"""
    # Random weight term for interpolation
    alpha = torch.rand((real_samples.size(0), 1)).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = critic(interpolates, labels)
    fake = torch.ones(real_samples.size(0), 1).to(device)

    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def load_and_process_data(fold_dir, subset):
    """
    Load and process data for a single fold
    Returns features and labels
    """
    X = []
    y = []
    for class_label in ['0', '1', '2']:
        class_dir = os.path.join(fold_dir, subset, class_label)
        for filename in os.listdir(class_dir):
            if filename.endswith('.tsv'):
                filepath = os.path.join(class_dir, filename)
                data = pd.read_csv(filepath, sep='\t', header=None, usecols=[2])
                features = np.round(data.values.astype(float), 5).flatten()
                X.append(features)
                y.append(int(class_label))
    return np.array(X), np.array(y)


def train_wgan(X_train, y_train, target_samples_per_class, n_epochs=50, batch_size=32):
    """
    Train WGAN to generate synthetic samples for minority classes
    Returns augmented dataset
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Calculate how many samples to generate per class
    unique, counts = np.unique(y_train, return_counts=True)
    class_counts = dict(zip(unique, counts))
    samples_to_generate = {cls: max(0, target_samples_per_class - count)
                           for cls, count in class_counts.items()}

    # Standardize data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Create dataset and dataloader
    dataset = GeneDataset(X_train_scaled, y_train)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=True)  # 丢弃最后不完整的batch
    # Initialize models
    latent_dim = 100
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))

    generator = Generator(latent_dim, input_dim, num_classes).to(device)
    critic = Critic(input_dim, num_classes).to(device)

    # Optimizers
    lr = 0.0001
    beta1, beta2 = 0.5, 0.9
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_C = optim.Adam(critic.parameters(), lr=lr, betas=(beta1, beta2))

    # Training loop
    n_critic = 5  # Number of critic iterations per generator iteration

    for epoch in tqdm(range(n_epochs), desc="Training WGAN"):
        for i, (real_data, labels) in enumerate(dataloader):
            real_data = real_data.to(device)
            labels = labels.to(device)
            batch_size = real_data.size(0)

            # Configure input
            real_data = real_data.view(batch_size, -1)

            # ---------------------
            #  Train Critic
            # ---------------------
            optimizer_C.zero_grad()

            # Sample noise and labels as generator input
            z = torch.randn(batch_size, latent_dim).to(device)
            gen_labels = torch.randint(0, num_classes, (batch_size,)).to(device)

            # Generate a batch of fake samples
            fake_data = generator(z, gen_labels)

            # Real samples
            real_validity = critic(real_data, labels)
            # Fake samples
            fake_validity = critic(fake_data.detach(), gen_labels)

            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(critic, real_data.data, fake_data.data,
                                                        labels.data, device)

            # WGAN critic loss
            c_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + 10 * gradient_penalty

            c_loss.backward()
            optimizer_C.step()

            # Train generator every n_critic iterations
            if i % n_critic == 0:
                # -----------------
                #  Train Generator
                # -----------------
                optimizer_G.zero_grad()

                # Generate a batch of fake samples
                gen_labels = torch.randint(0, num_classes, (batch_size,)).to(device)
                fake_data = generator(z, gen_labels)

                # Generator loss
                g_loss = -torch.mean(critic(fake_data, gen_labels))

                g_loss.backward()
                optimizer_G.step()

    # Generate synthetic samples
    synthetic_samples = []
    synthetic_labels = []

    for class_label, n_samples in samples_to_generate.items():
        if n_samples <= 0:
            continue

        # Generate noise and labels
        z = torch.randn(n_samples, latent_dim).to(device)
        labels = torch.full((n_samples,), class_label, dtype=torch.long).to(device)

        # Generate samples
        with torch.no_grad():
            gen_samples = generator(z, labels).cpu().numpy()

        # Inverse transform to original scale
        gen_samples = scaler.inverse_transform(gen_samples)

        synthetic_samples.append(gen_samples)
        synthetic_labels.append(labels.cpu().numpy())

    # Combine real and synthetic data
    if len(synthetic_samples) > 0:
        X_augmented = np.concatenate([X_train] + synthetic_samples)
        y_augmented = np.concatenate([y_train] + synthetic_labels)
    else:
        X_augmented = X_train
        y_augmented = y_train

    return X_augmented, y_augmented


def save_augmented_data(fold_dir, subset, X_aug, y_aug, output_dir):
    """
    Save augmented data maintaining original TSV file structure
    """
    for class_label in ['0', '1', '2']:
        class_mask = (y_aug == int(class_label))
        X_class = X_aug[class_mask]

        output_class_dir = os.path.join(output_dir, f'fold_{os.path.basename(fold_dir)}', subset, class_label)
        os.makedirs(output_class_dir, exist_ok=True)

        # Clear directory if it already exists
        for f in os.listdir(output_class_dir):
            os.remove(os.path.join(output_class_dir, f))

        # Save as individual TSV files
        for i in range(len(X_class)):
            output_path = os.path.join(output_class_dir, f'sample_{i}.tsv')
            # Create TSV with same format as original (assuming 3 columns)
            with open(output_path, 'w') as f:
                for val in X_class[i]:
                    f.write(f"gene\tinfo\t{val:.5f}\n")


def augment_and_save_with_wgan(data_dir, n_splits=5, target_samples_per_class=1000):
    """
    Perform data augmentation using WGAN and save results
    Maintains the cross-validation structure
    """
    # Create output directory
    output_dir = os.path.join(data_dir, 'augmented_wgan_data')
    os.makedirs(output_dir, exist_ok=True)

    for fold_idx in range(n_splits):
        print(f"\n=== Processing Fold {fold_idx} ===")
        fold_dir = os.path.join(data_dir, f'fold_{fold_idx}')

        # Load original training data
        X_train, y_train = load_and_process_data(fold_dir, 'train')

        # Print original data distribution
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"Fold {fold_idx} Original training data distribution: {dict(zip(unique, counts))}")

        # Data augmentation with WGAN
        X_aug, y_aug = train_wgan(X_train, y_train, target_samples_per_class)

        # Print augmented distribution
        unique_aug, counts_aug = np.unique(y_aug, return_counts=True)
        print(f"Fold {fold_idx} Augmented training data distribution: {dict(zip(unique_aug, counts_aug))}")

        # Save augmented training set
        save_augmented_data(fold_dir, 'train', X_aug, y_aug, output_dir)

        # Copy original validation set (no augmentation)
        X_val, y_val = load_and_process_data(fold_dir, 'val')
        save_augmented_data(fold_dir, 'val', X_val, y_val, output_dir)


# Usage example
data_directory = r'D:\deeplearning\Gene_fanxiu\不同数据扩充方法\cross'
augment_and_save_with_wgan(data_directory,
                           n_splits=5,
                           target_samples_per_class=1000)