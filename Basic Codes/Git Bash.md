# git
## git init
## git config
```bash
git config --global user.email "rmx1000@yonsei.ac.kr"
git config --global user.name "KimRass"
```
```bash
git config --global http.postBuffer 524288000
```
### git config user.email
### git config user.name
## git remote add origin
```bash
git remote add origin "https://github.com/KimRass/Programming.git"
```
## git remote remove origin
## git remote -v
## git status
## git pull
```bash
git pull origin master
```
## git remote set-url origin
```bash
git remote set-url origin "git@github.com:KimRass/Work.git"
```
- SSH Public Key 등록한 다음에도 Git clone 시 ID/Password 인증을 요구하는 경우에는 git이 SSH로 생성된 것이 아니고, http로 생성된  경우입니다.  이 경우 현재 git의 URL을 $ git remote -v로  확인하고 SSH URL로 전환시켜줘야 합니다.
## git add
```bash
git add .
```
## git commit
```bash
git commit -m "commit"
```
## git push
```bash
git push origin master
```
```bash
git push -f origin master
```
## git stash
## git stash pop
## git clone
```bash
git clone "git@github.com:KimRass/Work.git"
```
## git reflog
## git reset
```bash
git reset HEAD~1
```
# ssh-keygen
# cat ~/.ssh/id_rsa.pub
# ssh -T git@github.com