import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import torchvision.transforms as transforms
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ProvGigaPathEmbedder:
    def __init__(self, model, transforms, paths):
        """Initializes the embedder with a model, transform, and path config."""
        self.model = model
        self.transform = transforms
        self.paths = paths

        self.tile_path = paths["PROCESSED_DIR"] / 'tiles'

        self.tiles_df = pd.read_csv(paths["METRICS_DIR"] / 'tiles_per_particle.csv')

    def load_image(self, image_path):
        """Loads an image from disk as an RGB PIL image."""
        img = Image.open(image_path).convert("RGB")
        return img

    def compute_embedding(self, image_path):
        """Computes the model embedding for a single image."""
        img = self.load_image(image_path)
        x = self.transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            feats = self.model.forward_intermediates(
                x, intermediates_only=True
            )
            embedding = feats[-1].mean(dim=(2, 3))

        return embedding.squeeze(0).cpu().numpy()


    def precompute_embeddings(self, data_csv: pd.DataFrame, output_dir: Path):
        """Computes and saves embeddings for all tiles listed in data_csv."""
        output_dir.mkdir(parents=True, exist_ok=True)

        for idx, row in tqdm(data_csv.iterrows(), total=len(data_csv)):
            slide_name = row['slide_name']
            temp_df = self.tiles_df[self.tiles_df["slide_name"] == slide_name]
            for _, tile_row in temp_df.iterrows():
                tile_pattern = f"{slide_name}*particle_{tile_row['particle_id']}*.png"
                matching_files = list(self.tile_path.glob(tile_pattern))


                for image_path in matching_files:
                    embedding = self.compute_embedding(image_path)
                    output_path = output_dir / f"{Path(image_path).stem}_embedding.npy"
                    np.save(output_path, embedding)