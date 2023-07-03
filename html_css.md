# HTML
## HTML Escape
- `"<"`: `"&lt;"`
- `">"`: `"&gt;"`
- `" "`: `&nbsp;`
- `"&"`: `"&amp;"`
- `"""`: `"&quot;"`
- `"'"`: `"&apos;"`
- `"©"`: `"&copy;"`
## 기본 구조
```html
<!doctype html>
<html lang="ko">
    <head>
        <meta charset="UTF-8">
        <title>HTML</title>
    </head>
    <body>
    </body>
</html>
```
* element : defined by a start tag, some content, and an end tag
* ancestor elements <-> descendant elements
* parent elements<-> child elements
* sibling elements
## comment tags
```html
<!-- 여기에 작성되는 내용들은 모두 주석 처리가 됩니다. -->
```
## hypertext reference
```html
<h2 id="top">개발</h2>
<a href="#top">개발 페이지로 이동</a>
```
```html
<a href="http://www.naver.com", target="_blank">네이버</a>
```
## container

* \<div> : division. block level
* \<span> : inline level

## list

* \<ul> : unordered list. 순서가 없는 리스트
* \<ol> : ordered list. 순서가 있는 리스트
* \<li>만 자식 태그로 가질 수 있음
```html
<ul> 
    <li>콩나물</li> 
    <li>파</li> 
    <li>국간장</li>
</ul>
```
* \<dl> : definition list
* \<dt> : definition term.
* \<dd> : definition description. 용어의 정의
```html
<dl>
    <dt>리플리 증후군</dt>
    <dd>허구의 세계를 진실이라 믿고 거짓된 말과 행동을 상습적으로 반복하는 반사회적 성격장애를 뜻하는 용어</dd>
    <dt>피그말리온 효과</dt>
    <dd>타인의 기대나 관심으로 인하여 능률이 오르거나 결과가 좋아지는 현상</dd>
    <dt>언더독 효과</dt>
    <dd>사람들이 약자라고 믿는 주체를 응원하게 되는 현상</dd>
</dl>
```
## image
```html
<img src="./images/pizza.png" alt="피자" width="400" height="200">
```
* \./ : 현재 폴더
* \../ : 상위 폴더
## table
```html
<table>
    <tr>
        <td>1</td>
        <td>2</td>
        <td>3</td>
        <td>4</td>
    </tr>
    <tr>
        <td>5</td>
        <td>6</td>
        <td>7</td>
        <td>8</td>
    </tr>
    <tr>
        <td>9</td>
        <td>10</td>
        <td>11</td>
        <td>12</td>
    </tr>
    <tr>
        <td>13</td>
        <td>14</td>
        <td>15</td>
        <td>16</td>
    </tr>
</table>
```
* \<tr> : table row
* \<th> : table head
* \<td> : table data
```html
<table>
    <caption>table</caption>
    <thead>
        <tr>
          <th>Month</th>
          <th>Savings</th>
        </tr>
    </thead>
    <tbody>
        <tr>
          <td>January</td>
          <td rowspan="2">$100</td>
        </tr>
        <tr>
          <td>February</td>
        </tr>
    </tbody>
    <tfoot>
        <tr>
          <td colspan="2">Sum</td>
        </tr>
    </tfoot>
</table>
```
## form
```html
<form action="http://www.naver.com" method="get">
<label for="userid">아이디 : </label><input type="text" placeholder="id" id="userid" name="id"><br>
<label for="userpw">비밀번호 : </abel><input type="password" id="userpw" name="pw">
</form>
```
- method=" : action 값의 페이지로 이동
- method="get" : action?name1=value1&name1=value1 형태의 페이지로 이동
## fieldset, legend
```html
<fieldset>
    <legend>부가 입력 사항</legend>
</fieldset>
```
## input
```html
성별 : <label for="male">남자<input type="radio" name="gender" checked id="male"><label for="female"> 여자</label><input type="radio" name="gender" id="female">
```
```html
취미 : 등산<input type="checkbox" name="hobby">독서<input type="checkbox" name="hobby">운동<input type="checkbox" name="hobby"><br>
```
<input type="submit" value="제출"><br>
```html
<input type="reset" value="초기화"><br>
```
<input type="button" value="버튼"><br>
```html
<input type="image" src="http://placehold.it/50x50?text=click" alt="클릭" width="50" height="50">
```
## button
```html
<button type="submit">제출</button><br>
<button type="reset">초기화</button>
```
* \<br> : linebreak
## select
```html
지역 : <select>
    <option>서울</option>
    <option>경기</option>
    <option selected>강원</option>
</select>
```
## textarea
```html
경력 : <textarea cols="30" rows="10" placeholder="자유형식으로 작성 부탁드립니다."></textarea>
```
## semantic markup
```html
<b>굵은</b> vs <strong>중요한</strong>
<i>기울어진</i> vs <em>강조하는</em>
<u>밑줄친</u> vs <ins>새롭게 추가된</ins>
<s>중간선이 있는</s> vs <del>삭제된</del>
```
* \<article>
* \<aside>
* \<figcaption>
* \<figure>
* \<footer>
* \<header>
* \<main>
* \<mark>
* \<nav>
* \<section>
* \<time>
# CSS
## 기본 구조(.css)
```html
<head>
    h1{color:yellow;font-size:2em;}
</head>
```
* rule set - h1 { color: yellow; font-size:2em;}
* selector - h1
* declaration block - {color: yellow; font-size:2em;}
* declaration - color:yellow, font-size:2em
* property - color
* value - yellow
## application
```html
<link rel="stylesheet" href="./style.css">
```
## selector 
### type selector
### class selector
### id selector
### type selector
* 문서 내 유일한 element에 사용

### attribute selector
```html
p[class]{color:silver;}
p[class][id]{text-decoration:underline;}
```
```html
p[class="foo"]{color:silver;}
p[id="title"]{text-decoration:underline;}
```
```html
p[class~="color"]{font-style:italic;} /*공백으로 구분된 "color"를 포함*/
p[class*="color"]{font-style:italic;} /*"color"를 포함*/
p[class^="color"]{font-style:italic;} /*"color"로 시작*/
p[class$="color"]{font-style:italic;} /*"color"로 끝*/
```
### descendant selector
```html
div span{color:red;}
```
### child selector
```html
div>h1{color:red;}
```
### general sibling selector
```html
div~p{color:red;}
```
### adjacent sibling selector
```html
div+p{color:red;}
```
## pseudo class
```html
li:first-child{color:red;} /*첫 번째 child element 선택*/
li:last-child{color:blue;} /*마지막 child element 선택*/
```
```html
a:link{color:black;} /*아직 방문하지 않은 anchor*/
a:visited{color:blue;} /*이미 방문한 anchor*/
```
```html
a:focus{background-color:yellow;}
a:hover{font-weight:bold;}
a:active{color:red;}
```
- :focus: 현재 입력 초점을 가진 element에 적용
- :hover: 마우스 포인터가 있는 element에 적용
- :active: 사용자 입력으로 활성화된 element에 적용
## pseudo element
```html
p::before{content:"###"}
p::after{content:"!!!"}
p::first-line{color:blue;}
p::first-letter{color:blue;}
```
- :before : 가장 앞에 element 삽입
- :after : 가장 뒤에 element를 삽입
- :first-line : element의 첫 번째 줄에 있는 텍스트
- :first-letter : block level element의 첫 번째 문자
## cascading order
### specificity
```html
<p id="page" style="color:blue">Lorem impusm</p> /*inline style : 1, 0, 0, 0*/
div#page{...} /*id selector : 0, 1, 0, 1*/
p.bright em.dark{...} /*class selector : 0, 0, 2, 2*/
body h1{...} /*0, 0, 0, 2*/
```
### !important
```html
p#page{color:red!important;}
```
### inheritance
- margin, padding, background, border 등 박스 모델 property들은 상속되지 않음.
- 상속된 property는 아무런 구체성을 가지지 못함.
- !important > inline style > id selector > class selector > type selector > universal selector > inheritance

## attribute
### length
```html
.div{font-size:16px;}
```
- px : 1px=1/96inch
- pt : 1pt=1/72inch
- % : 부모의 값에 대해서 백분율로 환산한 크기
- em : **parent element의 font-size 기준 환산.** 소수점 3자리까지 표현 가능.
- rem : 최상위 element(보통은 html 태그)의 font-size 기준 환산.
- vw : viewport의 width값을 기준으로 1%의 값으로 계산.
```html
<h1 style="color:red">heading</h1>
<h1 style="color:#ff0000">heading</h1>
<h1 style="color:#f00">heading</h1>
<h1 style="color:rgb(255, 0, 0)">heading</h1>    
<h1 style="color:rgba(255, 0, 0, 0.5)">heading</h1>
```
- 6자리의 16진수에서 각각의 두 자리가 같은 값을 가지면 3자리로 축약하여 사용할 수 있습니다. 예를 들어, #aa11cc 는 #a1c로 축약하여 사용할 수 있습니다.
### background
```html
div{
  height:1500px;
  background-color:yellow;
  background-image:url(https://www.w3schools.com/CSSref/img_tree.gif);
  background-repeat:no-repeat; /*repeat, repeat-x, repeat-y, no-repeat
  background-position:center top; /*x% y%, xpx ypx*/
  background-attachment:scroll; /*scroll, local, fixed*/
}
```
```html
div{
  height:1500px;
  background:yellow url(https://www.w3schools.com/CSSref/img_tree.gif) no-repeat center top scroll;
}
```
## box model
- content - padding - border - margin
- width or height of box model = width or height of content + padding + border
### padding
- percent : specifies the padding in percent of the **width** of the containing element.
### border
```html
div{
  border-top-width:10px; /*border-top, border-bottom, border-right, border-left*/
  border-right-style:double; /*none, soild, double, dotted...*/
  border-left-color:#000;
}
```
- 참고자료 : https://www.w3schools.com/cssref/pr_border-bottom_style.asp
### margin, margin-top, margin-bottom, margin-right, margin-left
- length, percent, auto
-margin-right:auto; margin-left:auto;일 경우 자신의 width를 제외한 나머지 영역에 대해 균등 분할(수평 중앙 정렬. 수직 중앙 정렬은 불가).
- percent : specifies the margin in percent of the **width** of the containing element.
- can have negative values.
### margin collapse
- Top and bottom margins of elements are sometimes collapsed into a single margin that is equal to the largest of the two margins.
This does not happen on horizontal (left and right) margins! Only vertical (top and bottom) margins!
## width, height
- defines the width(height) in percent of the containing block.
- length, auto
## typography
- ![typography.png](/wikis/2670857615939396646/files/2823392681883991974)
- baseline : 소문자 x 기준 하단 라인.
### font-family
```html
font-family: Helvetica, Dotum, '돋움', Apple SD Gothic Neo, sans-serif;
```
- 가장 먼저 Helvetica를 사용하고, 이를 사용할 수 없을 때 Dotum을 사용하는 방식으로 우선순위에 따라 차례대로 적용 됩니다.
- 한글을 지원하지 않는 디바이스일 경우 해당 한글 폰트를 불러올 수 없으므로 영문명으로도 선언해 주어야합니다.
- family-name에 공백이 있으면 따옴표로 묶어서 선언한다.
- 마지막에는 반드시 generic-family를 선언 해주어야 합니다. 그 이유는 선언된 모든 서체를 사용할 수 있다는 보장이 없기 때문입니다. 이때 generic-family를 선언해주면, 시스템 폰트 내에서 사용자가 의도한 스타일과 유사한 서체로 적용되기 때문입니다. **자식 요소에서 font-family를 재선언하면 부모에 generic-family가 선언되어있어도 다시 선언해주어야 합니다.**
Generic-Family에는 대표적인 서체로 serif, sans-serif가 있습니다. serif는 글자 획에 삐침이 있는 폰트로 대표적으로 명조체가 있으며, sans-serif는 획에 삐침이 없는 폰트로 대표적으로 돋움체가 있습니다.

### line-height
- line-height로 제어되는 부분을 line-box라고도 하며 typography에서의 em+상하단의 여백을 의미
- normal, number, length, percent.
- number : **font-size 기준. 값이 그대로 상속. 즉 child element의 font-size를 기준으로 다시 계산.**
- percent : **font-size 기준. %에 의해 계산된 px값이 상속.**

### font-size
- keyword, length
### font-weight

- normal :기본값(=400)
- bold : 굵게 표현(=700)
- bolder : 부모 요소 보다 두껍게 표현
- lighter : 부모 요소 보다 얇게 표현
- number : 100, 200, 300, 400, 500, 600, 700, 800, 900(클수록 더 두껍게 표현)
- normal과 bold만 지원하는 폰트일 경우에는 100~500까지는 normal로, 600~900까지는 bold로 표현
### font-style
normal, italic, oblique
### font-variant
- normal, small-caps
### web fonts
```html
@font-face{
    font-family:webNanumGothic; /*이름*/
    src:url(NanumGothic.eot); /*경로*/
    font-weight:bold; /*옵션*/
    font-style:italic; /*옵션*/
}

h1{font-family:webNanumGothic;}
```
### vertical-align
- inline level 또는 inline-block level에서만 사용 가능.
- keyword : baseline(기본값), sub, super, top, text-top, middle, bottom, text-bottom
- length : baseline 기준.
- percent : line-height 기준.
### text-align
- left, right, center, justify
- inline level 또는 inline-block level에서만 사용 가능
### text-indent
- length, percent
- percent : **parent element의 width 기준.**
### text-decoration
```html
text-decoration:text-decoration-line text-decoration-color text-decoration-style;
```
- text-decoration-line : none(기본값), underline,  overline, line-through
- text-decoration-color : currentColor(기본값)
- text-decoration-style : solid(기본값), double, dashed, dashed, wavy
### white-space
- 공백의 처리에 관한 property
- normal : 공백과 개행 무시. 필요한 경우에 자동 줄바꿈 발생. 기본값.
- nowrap : 공백과 개행 무시. 자동 줄바꿈이 일어나지 않음.
- pre : 공백과 개행 표현. 자동 줄바꿈이 일어나지 않음.
- pre-line : 공백만 무시. 필요한 경우에 자동 줄바꿈 발생.
- pre-wrap : 개행만 무시. 필요한 경우 자동 줄바꿈 발생.
### letter-spacing
- normal, length
### word-spacing
- normal. length
### word-break
- normal :기본값. 중단점 - 음절(CJK), 공백 또는 하이픈(CJK 외)
- break-all : 중단점은 - 음절(CJK 및 그 외)
- keep-all : 중단점 - 공백 또는 하이픈(CJK 및 그 외)
### word-wrap
- normal : 기본값. 중단점에서 개행.
- break-word : 모든 글자가 요소를 벗어나지 않고 강제로 개행

## display
- none : 요소가 렌더링 되지 않음
- inline : inline level 요소처럼 렌더링
- block : block level 요소처럼 렌더링
- inline-block : inline level 요소처럼 렌더링되지만 block level의 성질을 가져 height나 width 등 box model property를 적용 가능.
- inline과 inline-block은 tag 사이의 공백 또는 개행이 있을 경우 4px의 여백을 가지게 됨.

|display|width|height|padding|border|margin|
| --- | --- | --- | --- | --- | --- |
|block|O|O|O|O|O|
|inline|X|X|좌/우|좌/우|좌/우|
|inline-block|O|O|O|O|O|

## visibility
- visible ; 보임. 기본값
- hidden ; 렌더링되며 화면에 margin을 포함한 box model은 차지하고 있지만 보이지 않도록 숨김
- collapse ; 셀간의 경계를 무시하고 숨김(박스영역 없음, 테이블의 행과 열 요소에만 지정 가능, 그 외 요소 지정은 hidden과 같음)

## float
- none, left, right
- 주변 텍스트나 인라인 요소가 주위를 감쌈.
- 대부분의 element에서 display:block;으로 변경.(예외 : inline-table, flex 등)
## clear
- none, left, right, both
- block-level element에만 적용 가능

## position
- static : normal-flow에 따라 배치. offset 미적용. 기본값.
- relative : 원래 있어야 할 위치를 기준으로 offset에 따라 배치. parent element의 position에 영향을 받지 않는다. normal-flow에 따라 배치. 주변 요소에 영향을 주지 않으면서 offset 값으로 이동.
- absolute : **position:static;가 아닌 ancestor element의 위치를 기준**으로 offset에 따라 배치(content+padding). display:block;으로 변경. **normal-flow에 따라 미배치.**
- fixed : 뷰포트를 기준으로 offset에 따라 배치.
즉, 화면 스크롤에 관계없이 항상 화면의 정해진 위치에 정보가 나타난다.
부모의 위치에 영향을 받지 않는다.
## offset - top, bottom, right, left
- length, percent, auto
- top, bottom은 기준이 되는 요소의 height를, left, right는 width를 기준으로 계산.

## z-index
**- only works on elements with position:relative;, absolute;, fixed;**.
- 따로 지정하지 않은 경우 코드 순서에 따라 쌓임.
- **parent element가 z-index property를 가지고 있을 경우 child element의 z-index property는 해당 parent element의 하위에서만 의미를 가짐.**
- auto, number
- auto : parent element와 동일한 stack order 설정.

## @media
```html
@media mediaqueries {/* style rules */}
```
- media types : all, print, screen
- media features : width, orientation
```html
media_query_list /* 여러개의 미디어 쿼리로 이루어진 리스트로 작성 가능하며, 쉼표를 이용해서 구분 */
 : S* [media_query [ ',' S* media_query ]* ]?
 ;
media_query
 : [ONLY | NOT]? S* media_type S* [ AND S* expression ]* /* 미디어 타입에 and 키워드를 이용해서 미디어 표현식을 붙일 수 있습니다. 미디어 타잎 앞에는 only 또는 not 키워드가 올 수 있습니다. 미디어 표현식은 생략 가능하기 때문에 미디어 타입 단독으로 사용될 수 있습니다. */
 | expression [ AND S* expression ]* /* 미디어 타입 없이 미디어 표현식이 바로 나올 수 있습니다.(미디어 타입이 명시되지 않으면 all로 간주합니다.) 미디어 표현식은 and 키워드를 이용해서 계속해서 나올 수 있습니다. */
 ;
expression
 : '(' S* media_feature S* [ ':' S* expr ]? ')' S* /* 디어 표현식은 괄호로 감싸야 하며, 특성 이름과 해당하는 값으로 이루어집니다. 이름과 값은 : 기호로 연결합니다. 또, 값이 없이 특성 이름만으로도 작성할 수 있다 */
 ;
```
- [a] : a가 나올 수도 있고 나오지 않을 수도 있습니다.
- a | b : a, b 중 하나만 선택.
- a? :  a가 0번 또는 1번만 나올 수 있음.
- a* : a가 0번 또는 그 이상 계속 나올 수 있음.

## viewport
```html
<meta name="viewport" content="width=device-width, initial-scale=1.0">
```
- width(height) : px 단위의 수치가 들어갈 수 있지만, 대부분 특수 키워드인 "device-width(height)"를 사용.
- initial-scale : 페이지가 처음 나타날 때 초기 줌 레벨 값 설정.
- user-scalable : 사용자의 확대/축소 기능 설정.
