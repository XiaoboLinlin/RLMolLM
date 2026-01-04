import torch
import transformers

class Discriminator(torch.nn.Module):
    """Class for a Discriminator model based off of a transformer model."""

    def __init__(self, model_directory, random_init=False, dropout_fraction=0.0):
        """Constructor for Discriminator class.

        Args:
            model_directory (str): Directory to be used to initialize model using hugging face
            dropout_fraction (float): Dropout used after embedding layer
        """
        super(Discriminator, self).__init__()

        # transformer model generates embeddings
        self.embedding = None
        if random_init:
            config = transformers.AutoConfig.from_pretrained(model_directory)
            self.embedding = transformers.AutoModel.from_config(config)
        else:
            self.embedding = transformers.AutoModel.from_pretrained(model_directory)
        self._embedding_dim = self.embedding.config.hidden_size

        self.dropout = torch.nn.Dropout(p=dropout_fraction)

        # single linear layer to generate score
        self.fc0 = torch.nn.Linear(self._embedding_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        """Setup initial parameters for final linear layer."""
        self.fc0.weight.data.normal_(mean=0.0, std=0.02)
        self.fc0.bias.data.zero_()

    def forward(self, input_ids, one_hot_tokens, attention_mask):
        """Forward pass for model.
        
        Args:
            input_ids (tensor): Contains token ids for input text
            one_hot_tokens (tensor): Contains one-hot version of input_ids, used if input_ids is None
            attention_mask (tensor): Contains attention mask for input text

        Returns:
            tensor with logits for discriminator task
        """
        inputs_embeds = None
        if input_ids is None:
            inputs_embeds = torch.matmul(one_hot_tokens, self.embedding.embeddings.word_embeddings.weight)
        x = self.embedding(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask)[0][:,0]
        x = self.dropout(x)
        x = self.fc0(x).flatten()
        
        return x