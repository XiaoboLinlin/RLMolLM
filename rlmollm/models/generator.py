import torch
import transformers

class Generator(torch.nn.Module):
    """Class for a Generator model based off of a masked language model."""

    def __init__(self, model_directory, tokenizer, random_init=False):
        """Constructor for Generator class.

        Args:
            model_directory (str): Directory to be used to initialize model using hugging face
            tokenizer (hugging face tokenizer): Tokenizer determines conversion of text to token ids 
        """
        super(Generator, self).__init__()
        self.model_directory = model_directory
        # language model is used to generate embeddings - each embedding ranks all tokens in vocab
        self.embedding = None
        if random_init:
            config = transformers.AutoConfig.from_pretrained(model_directory)
            self.embedding = transformers.AutoModelForMaskedLM.from_config(config)
        else:
            self.embedding = transformers.AutoModelForMaskedLM.from_pretrained(model_directory, use_auth_token=True)
        self._embedding_dim = self.embedding.config.hidden_size

        self._tokenizer = tokenizer

        # separator token is used to mark the end of molecule sequences
        self._sep_token_tensor = torch.nn.parameter.Parameter(torch.zeros(self.embedding.config.vocab_size, dtype=torch.float), requires_grad=False)
        self._sep_token_tensor[tokenizer.sep_token_id] = 1.0

    def forward(self, input_ids, attention_mask, hard=True, raw=False):
        """Forward pass for model.

        Args:
            input_ids (tensor): Contains token ids for input text
            attention_mask (tensor): Contains attention mask for input text
            hard (bool): Option to use hard (i.e. one-hot) version of gumbel_softmax

        Returns:
            tensor with one-hot representation of token ids
        """

        # generate language model outputs and perform categorical sampling
        x = self.embedding(input_ids=input_ids, attention_mask=attention_mask)[0]

        if not raw:
            x = torch.nn.functional.gumbel_softmax(x, tau=1.0, hard=hard)

            # handle sep and class token
            sep_token_mask = 1.0*(input_ids == self._tokenizer.sep_token_id)
            cls_token_mask = 1.0*(input_ids == self._tokenizer.cls_token_id)
            neg_special_token_mask = attention_mask - sep_token_mask - cls_token_mask
            x = neg_special_token_mask.unsqueeze(-1) * x
            x[:,0,self._tokenizer.cls_token_id] = 1.0
            x = x + (sep_token_mask.unsqueeze(-1)*self._sep_token_tensor.expand(x.shape[0],x.shape[1],self.embedding.config.vocab_size))

        return x
