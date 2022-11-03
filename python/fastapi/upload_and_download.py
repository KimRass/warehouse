- Reference: https://lucky516.tistory.com/119
# 백엔드 서버에 이미지 파일을 저장해야 하는 일이 있다.
# 가령 게시판에 사진을 올린다던지.
# 헌데 이미지파일을 DB에 저장하면 DB 쿼리 성능을 저하시킨다.
# 이 때문에 서버에 이미지를 저장할 때에는
# 서버의 파일 시스템에 이미지를 저장
# DB에는 이 파일의 URL만 저장해야한다

import os
from fastapi import (
    FastAPI,
    File,
    UploadFile
)

app = FastAPI()


@app.post("/photo")
async def upload_photo(file: UploadFile):
    UPLOAD_DIR = "./photo"  # 이미지를 저장할 서버 경로
    
    content = await file.read()
    filename = f"{str(uuid.uuid4())}.jpg"  # uuid로 유니크한 파일명으로 변경
    with open(os.path.join(UPLOAD_DIR, filename), "wb") as fp:
        fp.write(content)  # 서버 로컬 스토리지에 이미지 저장 (쓰기)

    return {"filename": filename}