```python
import os
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "파일이름.json"
```
```python
def transcribe_gcs(gcs_uri):
    client = speech.SpeechClient()
    audio = types.RecognitionAudio(uri=gcs_uri)
    config = types.RecognitionConfig(encoding=enums.RecognitionConfig.AudioEncoding.FLAC, sample_rate_hertz=16000, language_code='ko-KR')
    operation = client.long_running_recognize(config,audio)
    response = operation.result()
    return response
response = transcribe_gcs("gs://s2t_test_3552/s2t test.flac")
with open("s2t test script", "w") as script:
    for result in response.results:
        script.write(u"{}".format(result.alternatives[0].transcript) + "\n")
print("completed")
```
