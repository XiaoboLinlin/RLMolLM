import torch

class MoleculeValueNetwork(torch.nn.Module):
    """Value network that processes the entire molecule representation."""
    
    def __init__(self, hidden_size, device, dropout_rate=0.1):
        super(MoleculeValueNetwork, self).__init__()
        
        # Self-attention layer to process the sequence
        self.attention = torch.nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        ).to(device)
        
        # Layers for value prediction after attention
        self.fc1 = torch.nn.Linear(hidden_size, hidden_size).to(device)
        self.dropout = torch.nn.Dropout(dropout_rate).to(device)
        self.fc2 = torch.nn.Linear(hidden_size, 1).to(device)
        
    def forward(self, hidden_states, attention_mask=None):
        # Apply self-attention to capture relationships between tokens
        # Convert attention_mask to proper format for MultiheadAttention
        if attention_mask is not None:
            # Convert 1s to False and 0s to True (MultiheadAttention uses key_padding_mask)
            key_padding_mask = (1 - attention_mask).bool()
        else:
            key_padding_mask = None
            
        # Self-attention over sequence
        attended_states, _ = self.attention(
            hidden_states, 
            hidden_states, 
            hidden_states,
            key_padding_mask=key_padding_mask
        )
        
        # Global max pooling to get the most important features
        pooled_output, _ = torch.max(attended_states, dim=1)
        
        # Final value prediction
        x = self.fc1(pooled_output)
        x = torch.nn.functional.relu(x)
        x = self.dropout(x)
        value = self.fc2(x)
        
        return value.squeeze(-1)