```xml
<!DOCTYPE html>
<html lang="ko">
  <head>
    <meta charset="utf-8">
    <title>Google</title>
    <style>
      h1{text-align:center; margin-top:166px; margin-bottom:37px; font-size:90px;}
      h1>span:nth-child(1){color:#4285f4;font-size:95px;}
      h1>span:nth-child(2){color:#ea4335;font-size:80px;}
      h1>span:nth-child(3){color:#fbbc05;font-size:80px;}
      h1>span:nth-child(4){color:#4285f4;font-size:80px;}
      h1>span:nth-child(5)<span style="color: #34a853">
      h1>span:nth-child(6){color:#ea4335;font-size:80px;}
      h1>span{letter-spacing:-3px;}
      .input-group{width:582px; height:44px; margin:-13px auto; border:1px solid rgba(150, 150, 150, 0.3); border-radius:24px;}
      .input-group:hover{box-shadow:0 1px 6px rgba(32, 33, 36, 0.28); border-color:rgba(223, 225, 229, 0);}
      input{width:440px; height:34px; background-color:transparent; border:none; margin-left:25px; margin-top:3px;}
      input:focus{outline:none;}
    </style>
  </head>
  <body>
    <h1 class="title">
      <span>G</span><span>o</span><span>o</span><span>g</span><span>l</span><span>e</span>
    </h1>
    <form action="https://www.google.com/search" method="get">
        <div class="input-group" >
          <input type="text" placeholder="Google 검색 또는 URL 입력" name="q">
        </div>
    </form>
  </body>
</html>

```
