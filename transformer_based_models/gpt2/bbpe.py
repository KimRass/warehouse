# Refrerences
    # https://medium.com/@pierre_guillou/byte-level-bpe-an-universal-tokenizer-but-aff932332ffe
    # https://velog.io/@goggling/%EC%9C%A0%EB%8B%88%EC%BD%94%EB%93%9C%EC%99%80-UTF-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0
    # https://velog.io/@zionhann/%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EC%9C%A0%EB%8B%88%EC%BD%94%EB%93%9C-%EB%AC%B8%EC%9E%90-%EB%B3%80%ED%99%98%ED%95%98%EA%B8%B0
    # https://www.compart.com/en/unicode/U+7FD2
    # https://konghana01.tistory.com/65

from transformers import GPT2TokenizerFast
import tokenizers
# from tokenizers import ByteLevelBPETokenizer
from pathlib import Path
import ssl


def tokenize(char):
    bytes = char.encode("utf-8")
    hexes = bytes.hex()
    tokenized = [chr(int(f"""0x{hexes[i: i + 2]}""", base=16)) for i in range(len(hexes))[:: 2]]
    return tokenized


ssl._create_default_https_context = ssl._create_unverified_context

gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
gpt2_tokenizer._tokenizer.get_vocab()

char="ㄱ"
bytes = char.encode("utf-8")
hexes = bytes.hex()
hexes
gpt2_tokenizer.encode("곰")
tokenize("곰")
gpt2_tokenizer.id_to_token([166])
# gpt2_tokenizer.decode(gpt2_tokenizer.encode("곰"))


gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token


bbpe_tokenizer_vocab_size = gpt2_tokenizer.vocab_size
bbpe_tokenizer_vocab_size





bbpe_tokenizer = ByteLevelBPETokenizer()
corpus_dir = ["/Users/jongbeomkim/Desktop/workspace/transformer_based_models/gpt2/corpus.txt"]
# Customize training with <|endoftext|> special GPT2 token
bbpe_tokenizer.train(
    files=corpus_dir, 
    # vocab_size=bbpe_tokenizer_vocab_size,
    vocab_size=2 ** 8,
    min_frequency=2, 
    special_tokens=["<|endoftext|>"]
)
# Get sequence length max of 1024
bbpe_tokenizer.enable_truncation(max_length=1024)

# save tokenizer
save_dir = Path("/Users/jongbeomkim/Desktop/workspace/transformer_based_models/gpt2/bbpe_test")
save_dir.mkdir(exist_ok=True, parents=True)
bbpe_tokenizer.save_model(str(save_dir))

# 3. Import the tokenizer config files in Portuguese into the pre-trained GPT2 Tokenizer

# Get the path to bbpe_tokenizer config files
bbpe_tokenizer_rep = "bbpe_tokenizer"
save_dir = path_data/bbpe_tokenizer_rep

# import the pre-trained GPT2TokenizerFast tokenizer with the tokenizer_pt config files
tokenizer_pt = GPT2TokenizerFast.from_pretrained(
    str(save_dir), 
    pad_token="<|endoftext|>"
)

# Get sequence length max of 1024
tokenizer_pt.model_max_length = 1024



from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

# Initialize a tokenizer
tokenizer = Tokenizer(models.BPE())

# Customize pre-tokenization and decoding
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
tokenizer.decoder = decoders.ByteLevel()
tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

# And then train
trainer = trainers.BpeTrainer(
    vocab_size=20000,
    min_frequency=2,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
)
tokenizer.train(["/Users/jongbeomkim/Desktop/workspace/transformer_based_models/gpt2/empty.txt"], trainer=trainer)

# And Save it
tokenizer.save("/Users/jongbeomkim/Desktop/workspace/transformer_based_models/gpt2/vocab.json", pretty=True)

tokenizer.encode("안녕").ids
tokenizer.
tokenizer.decode([168, 243, 230])
tokenizer.decode([168, 243, 100])
tokenizer.decode([168, 243])
tokenizer.id_to_token(220)


def tokenize(char):
    bytes = char.encode("utf-8")
    hexes = bytes.hex()
    tokenized = [chr(int(f"""0x{hexes[i: i + 2]}""", base=16)) + 1 for i in range(len(hexes))[:: 2]]
    return tokenized