# CutMix
- Reference: https://hongl.tistory.com/223
- Paper: "CutMix: Regularization Strategy to Train Strong Classifiers
with Localizable Features" (https://arxiv.org/pdf/1905.04899.pdf)
- CNN으로 하여금 이미지의 덜 중요한 부분까지 포커싱하게 만드는 Regional dropout 전략입니다.
- ![comparison](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fs9x0c%2Fbtq6VqMQBAE%2FMOL14qR9itAs2UHmYwyStK%2Fimg.png)
- ![comparison2](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F8R1sq%2Fbtq6ZUF1oXo%2F3dwTG1nxOpJGP7eXkQOLF1%2Fimg.png)

# Algorithm
- ![formula](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F1Yv3f%2Fbtq6ZQDEnDb%2FUII5WjOOh3Hk2KKnAq3n21%2Fimg.png)
- ![algorithm](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdtFiKV%2Fbtq6Vin5EhK%2FGXymes556bXRpkLTR7Mr40%2Fimg.png)
- 'x': Image
- 'y': Label
- 'M': 이미지의 어느 부분을 'x_A' 대신 'x_B'로 채울 지를 정하는 마스크
- 'A'와 'B'는 Mini batch 내에서 랜덤하게 선택합니다.
- 'x_A'에서 잘라낼 영역은 Rectangle로 다음과 같이 나타낼 수 있습니다; '(r_x, r_y, r_w, r_h)'
- 'r_x', 'r_y'는 'x_A' 내에서 Uniform distribution에 따라 선택됩니다. ('r_x ~ U(0, W)', 'r_y ~ U(0, H)')
- 'lambda': A와 B 간의 결합 비율이며 최초에 'Beta(alpha, alpha)'로부터 추출됩니다.
```python
def cutmix(input, input_s, beta)
    rx = np.random.randint(W)
    ry = np.random.randint(H)

    lamb = np.random.beta(beta, beta)
    ratio = np.sqrt(1 - lamb)
    rw = np.int(W * ratio)
    rh = np.int(H * ratio)

    x1 = np.clip(a=cx - cut_w // 2, amin=0, amax=W)
    x2 = np.clip(a=cx + cut_w // 2, amin=0, amax=W)
    y1 = np.clip(a=cy - cut_h // 2, amin=0, amax=H)
    y2 = np.clip(a=cy + cut_h // 2, amin=0, amax=H)

    input[y1: y2, x1: x2, :] = input_s[y1: y2, x1: x2, :]
```