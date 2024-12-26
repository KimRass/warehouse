# Refrerences
    # https://medium.com/@pierre_guillou/byte-level-bpe-an-universal-tokenizer-but-aff932332ffe
    # https://velog.io/@goggling/%EC%9C%A0%EB%8B%88%EC%BD%94%EB%93%9C%EC%99%80-UTF-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0
    # https://velog.io/@zionhann/%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EC%9C%A0%EB%8B%88%EC%BD%94%EB%93%9C-%EB%AC%B8%EC%9E%90-%EB%B3%80%ED%99%98%ED%95%98%EA%B8%B0
    # https://www.compart.com/en/unicode/U+7FD2
    # https://konghana01.tistory.com/65
    # https://medium.com/@hugmanskj/tokenization-%EB%B0%A9%EB%B2%95%EB%A1%A0%EB%93%A4%EC%97%90-%EB%8C%80%ED%95%9C-%EC%89%BD%EA%B3%A0-%EC%A7%81%EA%B4%80%EC%A0%81%EC%9D%B8-%EC%9D%B4%ED%95%B4-2fce5089758e

def bbpe_tokenize(string):
    ret = []
    for char in string:
        bytes = char.encode("utf-8")  # 입력 데이터를 raw bytes로 변환.
        hexes = bytes.hex()
        ret.append(
            [int(f"""0x{hexes[i: i + 2]}""", base=16) for i in range(len(hexes))[:: 2]]
        )
    return ret


def reverse_bbpe_tokenize(tokenized):
    result = ""
    for token in tokenized:
        byte_sequence = bytes(token)  # Convert the token back to bytes
        result += byte_sequence.decode("utf-8")  # Decode back to string
    return result


if __name__ == "__main__":
    string = "안녕!"
    tokenized = bbpe_tokenize(string)
    print(reverse_bbpe_tokenize(tokenized))
