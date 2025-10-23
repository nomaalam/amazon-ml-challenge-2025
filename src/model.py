# src/model.py
import torch
import torch.nn as nn
from transformers import AutoModel

class MultiModalPricer(nn.Module):
    def __init__(self, text_model_name, image_model_name, num_extra_features=1):
        """
        Initializes the Multi-Modal Pricing Model.

        Args:
            text_model_name (str): Name of the pre-trained text model from Hugging Face.
            image_model_name (str): Name of the pre-trained image model from Hugging Face.
            num_extra_features (int): Number of additional numerical features (e.g., IPQ).
        """
        super(MultiModalPricer, self).__init__()
        
        # Load pre-trained models
        self.text_model = AutoModel.from_pretrained(text_model_name)
        self.image_model = AutoModel.from_pretrained(image_model_name)
        
        # Get the embedding dimensions from the models
        text_embed_dim = self.text_model.config.hidden_size
        image_embed_dim = self.image_model.config.hidden_size
        
        # Total dimension for the concatenated feature vector
        combined_dim = text_embed_dim + image_embed_dim + num_extra_features
        
        # Regression head to predict the final price
        self.regression_head = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, text_input_ids, text_attention_mask, pixel_values, extra_features):
        """
        Forward pass of the model.
        """
        # Get text features (using the [CLS] token's embedding)
        text_outputs = self.text_model(input_ids=text_input_ids, attention_mask=text_attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]
        
        # Get image features (using the pooled output)
        image_outputs = self.image_model(pixel_values=pixel_values)
        image_features = image_outputs.pooler_output
        
        # Reshape extra features to match the batch size
        extra_features = extra_features.view(-1, 1)

        # Concatenate all features
        combined_features = torch.cat([text_features, image_features, extra_features], dim=1)
        
        # Pass through the regression head to get the final prediction
        price_prediction = self.regression_head(combined_features)
        
        return price_prediction