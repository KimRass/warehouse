# pygame
```python
import pygame
```
## gamepad
### gamepad.blit()
```python
def bg(bg, x, y):
    global gamepad, background
    gamepad.blit(bg, (x, y))
```
```python
def plane(x, y):
    global gamepad, aircraft
    gamepad.blit(aircraft, (x, y)) #"aircraft"를 "(x, y)"에 배치
```
## pygame
### pygame.event.get(), event.type, event.key
```python
while not crashed:
    for event in pygame.event.get():
        if event.type==pygame.QUIT: #마우스로 창을 닫으면
            crashed=True #게임 종료

        if event.type==pygame.KEYDOWN: #키보드를 누를 때
            if event.key==pygame.K_RIGHT:
                x_change=15
```
### pygame.KEYUP, pygame.KEYDOWN
```python
if event.type==pygame.KEYUP
```
- pygame.KEYUP : 키보드를 누른 후 뗄 때
- pygame.KEYDOWN : 키보드를 누를 때
### pygame.init()
### pygame.quit()
### pygame.display.set_model()
```python
gamepad = pygame.display.set_mode((pad_width, pad_height))
```
```python
import pygame

WHITE=(255, 255, 255)
pad_width=1536
pad_height=960
bg_width=2560

def bg(bg, x, y):
    global gamepad, background
    gamepad.blit(bg, (x, y))

def plane(x, y):
    global gamepad, aircraft
    gamepad.blit(aircraft, (x, y)) #"aircraft"를 "(x, y)"에 배치

def runGame():
    global gamepad, aircraft, clock, bg1, bg2
    
    x=pad_width*0.01
    y=pad_height*0.5
    x_change=0
    y_change=0
    
    bg1_x=0
    bg2_x=bg_width
    
    crashed=False #"True" : 게임 종료,  "False" : 안 종료
    while not crashed:
        for event in pygame.event.get():
            if event.type==pygame.QUIT: #마우스로 창을 닫으면
                crashed=True #게임 종료
                
            if event.type==pygame.KEYDOWN: #키보드를 누를 때
                if event.key==pygame.K_RIGHT:
                    x_change=15
                if event.key==pygame.K_LEFT:
                    x_change=-15
                elif event.key==pygame.K_UP:
                    y_change=-15
                elif event.key==pygame.K_DOWN:
                    y_change=15
            if event.type==pygame.KEYUP: #키보드를 누른 후 뗄 때
                if event.key==pygame.K_RIGHT or event.key==pygame.K_LEFT:
                    x_change=0
                if event.key==pygame.K_UP or event.key==pygame.K_DOWN:
                    y_change=0
        x+=x_change
        y+=y_change
                    
        gamepad.fill(WHITE) #"gamepad"를 "WHITE"로 채우고
        bg1_x-=5
        bg2_x-=5
        if bg1_x==-bg_width:
            bg1_x=bg_width
        if bg2_x==-bg_width:
            bg2_x=bg_width
            
        bg(bg1, bg1_x, 0)
        bg(bg2, bg2_x, 0)
        
        plane(x, y) #"plane(x, y)" 함수를 실행한 뒤
        pygame.display.update() #화면 갱신
        clock.tick(60) #"tick=60"으로 FPS=60 지정
        
    pygame.quit() #종료
    
def initGame():
    global gamepad, aircraft, clock, bg1, bg2
    
    pygame.init()
    gamepad=pygame.display.set_mode((pad_width, pad_height)) #화면 크기 지정
    pygame.display.set_caption("PyFlying") #타이틀 지정
    aircraft=pygame.image.load("pngguru.com (4).png")
    bg1=pygame.image.load("background.jpg")
    bg2=bg1.copy()
    
    clock=pygame.time.Clock() #FPS를 지정하기 위한 변수 "clock" 선언
    runGame()
    
initGame()
```
