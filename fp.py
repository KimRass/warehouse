import math


class FP16(object):
    def __init__(self, exp_bits=5, frac_bits=10):
        self.exp_bits = exp_bits
        self.frac_bits = frac_bits
        self.exp_bias = 2 ** (exp_bits - 1) - 1
        self.exp_end = self.exp_bits + 1

    @staticmethod
    def get_s(num):
        return int(num[:1])

    def get_E(self, num):
        E = int(num[1: self.exp_end], 2)
        return E

    def get_M(self, num):
        m = num[self.exp_end:]
        M = 0
        for i, j in enumerate(m, start=1):
            M += int(j) * (2 ** (-i))
        return M

    def __call__(self, num):
        assert len(num) == 1 + self.exp_bits + self.frac_bits

        E = self.get_E(num)
        M = self.get_M(num)
        if E == 0:
            if M == 0:
                return 0
            else:
                return ((-1) ** self.get_s(num)) * M * (
                    2 ** -(2 ** (self.exp_bits - 1) - 2)
                )
        elif E == 2 ** self.exp_bits - 1:
            if M == 0:
                return ((-1) ** self.get_s(num)) * math.inf
            else:
                return math.nan
        return ((-1) ** self.get_s(num)) * (1 + M) * (
            2 **  (E - self.exp_bias)
        )


class FP32(FP16):
    def __init__(self, exp_bits=8, frac_bits=23):
        super().__init__(exp_bits=exp_bits, frac_bits=frac_bits)


class FP64(FP16):
    def __init__(self, exp_bits=11, frac_bits=52):
        super().__init__(exp_bits=exp_bits, frac_bits=frac_bits)



if __name__ == "__main__":
    fp16 = FP16()
    fp16("0000000000000000") # 0
    # The minimum positive subnormal values; 5.960464477539063e-08
    fp16("0000000000000001")
    # The minimum positive normal values; 6.109476089477539e-05
    fp16("0000010000000001")
    fp16("0111101111111111") # The maximum representable value; 65504.0
    fp16("0111111000000000") # NaN
    fp16("0111110000000000") # Infinity

    fp32 = FP32()
    fp32("00000000000000000000000000000000") # 0
    # The minimum positive subnormal values; 1.401298464324817e-45
    fp32("00000000000000000000000000000001")
    # The minimum positive normal values; 1.175494490952134e-38
    fp32("00000000100000000000000000000001")
    # The maximum representable value; 3.4028234663852886e+38
    fp32("01111111011111111111111111111111")
    fp32("01111111100000010000000000000000") # NaN
    fp32("01111111100000000000000000000000") # Infinity

    fp64 = FP64()
    # The minimum positive subnormal values; 5e-324
    fp64("0000000000000000000000000000000000000000000000000000000000000001")
    # The minimum positive normal values; 4.450147717014404e-308
    fp64("0000000000100000000000000000000000000000000000000000000000000001")
    # The maximum representable value; 1.7976931348623157e+308
    fp64("0111111111101111111111111111111111111111111111111111111111111111")
