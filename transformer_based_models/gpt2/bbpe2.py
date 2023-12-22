import sentencepiece as spm
import re

sp = spm.SentencePieceProcessor()
sp.EncodeAsIds(text)
sp.encode_


WHITESPACE_NORMALIZER = re.compile(r"\s+")
SPACE = chr(32)
SPACE_ESCAPE = chr(9601)
# excluding non-breaking space (160) here
PRINTABLE_LATIN = set(
    list(range(32, 126 + 1)) + list(range(161, 172 + 1)) + list(range(174, 255 + 1))
)
BYTE_TO_BCHAR = {
    b: chr(b) if b in PRINTABLE_LATIN else chr(256 + b) for b in range(256)
}
BCHAR_TO_BYTE = {bc: b for b, bc in BYTE_TO_BCHAR.items()}


# re.sub(pattern=r"\s+", repl=chr(32), string=text)
def byte_encode(x: str) -> str:
    # x = text
    normalized = WHITESPACE_NORMALIZER.sub(SPACE, x)
    [BYTE_TO_BCHAR[b] for b in normalized.encode("utf-8")]
    return "".join([BYTE_TO_BCHAR[b] for b in normalized.encode("utf-8")])
byte_encode(text)


class ByteBPE(object):
    def __init__(self, cfg):
        vocab = file_utils.cached_path(cfg.sentencepiece_model_path)
        try:
            import sentencepiece as spm

            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(vocab)
        except ImportError:
            raise ImportError(
                "Please install sentencepiece with: pip install sentencepiece"
            )

    def encode(self, x: str) -> str:
        byte_encoded = byte_encode(x)
        return SPACE.join(self.sp.EncodeAsPieces(byte_encoded))

    @staticmethod
    def decode(x: str) -> str:
        unescaped = x.replace(SPACE, "").replace(SPACE_ESCAPE, SPACE)
        return smart_byte_decode(unescaped)