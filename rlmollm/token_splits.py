# pretokenizer
from tokenizers.pre_tokenizers import Split
from tokenizers import Regex
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.pre_tokenizers import Digits
from tokenizers.pre_tokenizers import Sequence
from tokenizers.pre_tokenizers import WhitespaceSplit

pretokenizer_dict = {
    'regex': Sequence([WhitespaceSplit(),Split(Regex(r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""), behavior='isolated')]),
    'bert': BertPreTokenizer(),
    'bert_digits': Sequence([BertPreTokenizer(), Digits(individual_digits=True)]),
    'digits': Sequence([WhitespaceSplit(), Digits(individual_digits=True)])
}