# src/transformer_model.py

import torch
import timm
from PIL import Image
from torchvision import transforms
import logging
from typing import List, Union

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VisionTransformerExtractor:
    """
    A feature extractor using a pretrained Vision Transformer (ViT) model.

    This class loads a powerful, pretrained model from the 'timm' library,
    preprocesses images to match the model's expected input format, and extracts
    rich feature embeddings. These embeddings represent the global context of an
    image, as learned from the massive ImageNet dataset.
    """

    def __init__(self, model_name: str = 'vit_base_patch16_224', device: str = None):
        """
        Initializes the feature extractor.

        Args:
            model_name (str): The name of the pretrained model from timm's library.
                              'vit_base_patch16_224' is a standard, effective choice.
            device (str, optional): The device ('cuda' or 'cpu') to run the model on.
                                    Automatically detects GPU if available.
        """
        # 1. Set the processing device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"VisionTransformerExtractor initialized on device: {self.device}")

        # 2. Load the pretrained Vision Transformer model
        # `pretrained=True` loads weights trained on ImageNet.
        # `num_classes=0` removes the final classification layer, so the model
        # outputs the feature embedding instead of a prediction.
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0).to(self.device)
        self.model.eval()  # Set the model to evaluation mode (disables dropout, etc.)

        # 3. Create the specific image transformation pipeline for this model
        # `timm` provides a convenient way to get the exact preprocessing steps
        # that the model was trained with.
        data_config = timm.data.resolve_data_config(self.model.pretrained_cfg)
        self.transform = timm.data.create_transform(**data_config)
        
        logging.info(f"Model '{model_name}' loaded successfully.")
        logging.info(f"Image transforms configured for input size: {data_config['input_size']}")

    def _preprocess_image(self, image_path: str) -> torch.Tensor:
        """Loads and preprocesses a single image file."""
        try:
            image = Image.open(image_path).convert("RGB")
            # Apply transforms (resize, crop, normalize) and add a batch dimension
            return self.transform(image).unsqueeze(0)
        except Exception as e:
            logging.error(f"Error processing image {image_path}: {e}")
            return None

    @torch.no_grad()  # Disables gradient calculation for inference, saving memory and time
    def get_embedding(self, image_path: str):
        """
        Extracts the feature embedding for a single image.

        Args:
            image_path (str): The file path to the image.

        Returns:
            numpy.ndarray: A 1D NumPy array representing the image embedding, or None.
        """
        image_tensor = self._preprocess_image(image_path)
        if image_tensor is None:
            return None
            
        image_tensor = image_tensor.to(self.device)
        embedding = self.model(image_tensor)
        
        # Move embedding to CPU and convert to a flat NumPy array
        return embedding.cpu().numpy().flatten()

    @torch.no_grad()
    def get_embeddings_batch(self, image_paths: List[str]):
        """
        Extracts feature embeddings for a batch of images for improved efficiency.

        Args:
            image_paths (List[str]): A list of file paths to the images.

        Returns:
            numpy.ndarray: A 2D array where each row is an image's feature embedding.
        """
        batch_tensors = []
        for path in image_paths:
            tensor = self._preprocess_image(path)
            if tensor is not None:
                batch_tensors.append(tensor)

        if not batch_tensors:
            return None

        # Stack individual tensors into a single batch tensor
        batch = torch.cat(batch_tensors).to(self.device)
        embeddings = self.model(batch)
        
        return embeddings.cpu().numpy()

# --- Main execution block for standalone testing ---
def main_test():
    """
    Demonstrates how to use the VisionTransformerExtractor and verifies it's working.
    This function will only run when the script is executed directly.
    """
    logging.info("--- Running standalone test for VisionTransformerExtractor ---")
    
    # 1. Initialize the extractor
    extractor = VisionTransformerExtractor()

    # 2. Create a dummy image for testing purposes
    try:
        dummy_image = Image.new('RGB', (250, 250), color='red')
        dummy_path = "test_image.png"
        dummy_image.save(dummy_path)
        logging.info(f"Created a dummy test image: '{dummy_path}'")

        # 3. Test single image embedding
        logging.info("Testing single image extraction...")
        embedding = extractor.get_embedding(dummy_path)
        if embedding is not None:
            logging.info(f"Success! Extracted single embedding of shape: {embedding.shape}")

        # 4. Test batch image embedding
        logging.info("\nTesting batch image extraction...")
        batch_embeddings = extractor.get_embeddings_batch([dummy_path, dummy_path])
        if batch_embeddings is not None:
            logging.info(f"Success! Extracted batch embeddings of shape: {batch_embeddings.shape}")

    except Exception as e:
        logging.error(f"An error occurred during the test: {e}")
    finally:
        # 5. Clean up the dummy image
        import os
        if os.path.exists(dummy_path):
            os.remove(dummy_path)
            logging.info(f"\nCleaned up dummy image.")

if __name__ == '__main__':
    main_test()