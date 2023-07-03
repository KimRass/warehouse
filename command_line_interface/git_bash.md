# Initialize
```bash
# `git init` is a one-time command you use during the initial setup of a new repository.
# Executing `git init` creates a ".git" subdirectory in the current working directory, which contains all of the necessary Git metadata for the new repository.
git init
```

# Clone Repository
```bash
# For public repository
git clone "https://github.com/....git"
# For private repository
git clone "https://<user_name>@github.com/....git"
```
## Clone Specific Branch
```bash
git clone -b <branch_name> [--single-branch] "https://...github.com/....git"
```

# Settings
## Case Sensitive
```bash
git config core.ignorecase false
...
git rm -r --cached .
```

# User Information
## List User Information
```bash
git config --list
```
## Configure User Information
```bash
# Options: `--global`, `--local`
git config user.email "<user_email>"
git config user.name "<user_name>"
```
## Delete User Information
```bash
git config --unset user.email "<user_email>"
git config --unset user.name "<user_name>"
```
## Configure Pull Options
```bash
# Turn off fast-forward only option
git config --unset pull.ff
```

# Remote
- The `git remote` command lets you create, view, and delete connections to other repositories. Remote connections are more like bookmarks rather than direct links into other repositories. Instead of providing real-time access to another repository, they serve as convenient names that can be used to reference a not-so-convenient URL.
```bash
# List the remote connections you have to other repositories. With `-v`, it includes the URL of each connection.
git remote -v
```
## Add Remote
```bash
# Create a new connection to a remote repository. After adding a remote, you’ll be able to use `＜remote_name＞` as a convenient shortcut for `＜repository_url＞` in other Git commands.
git remote add <remote_name> <repository_url>
```
## Remove Remote
```bash
git remote rm <remote_name>
```
## Modify Remote Name
```bash
git remote rename <old-name> <new-name>
```
## Modify Repository URL
```bash
git remote set-url <remote_name> "<repository_url>"
```

# Add Changes to the Staging Area
```bash
# The `git add` command adds a change in the working directory to the staging area. It tells Git that you want to include updates to a particular file in the next commit. However, `git add` doesn't really affect the repository in any significant way—changes are not actually recorded until you run `git commit`.
git add <file1> <file2> ...
git add .
```
## Undo `git add`
```bash
git reset HEAD [<file1> <file2> ...]
# All files become unstaged
git add *
```

# Remove File from the Staging Area
```bash
# Options
# `-r`: Recursively
# `--cached`: Keeps file locally.
# `-f`: Force removal
git rm <file1> <file2> ...
```

# Remove Untracked Files
```bash
git clean -f
# 디렉토리까지 지웁니다.
git clean -fd
```

# Commit Changes
```bash
# The `git commit` command is used to commit a snapshot of the staging directory to the repositories commit history.
git commit -m "<message>"
```
## Amend Most Recent Commit Message
```bash
git commit --amend -m "<message>"
```
## Undo Commit Changes
```bash
# commit을 취소하고 해당 파일들은 unstaged 상태로 워킹 디렉터리에 보존
git reset --mixed HEAD^
# commit을 취소하고 해당 파일들은 staged 상태로 워킹 디렉터리에 보존
git reset --soft HEAD^
```

# Push
- The `git push` command is used to upload local repository content to a remote repository.
- Pushing has the potential to overwrite changes, caution should be taken when pushing.
- To prevent you from overwriting commits, Git won’t let you push when it results in a non-fast-forward merge in the destination repository.
```bash
git push [<remote_name> <branch_name>]
# Force `git push`
git push -f [<remote_name> <branch_name>]
```
## Cancel Push
```bash
git log --oneline
git reset --hard <commit id>
git push -f
```

# Pull Repository
```bash
git pull [<remote_name> <branch_name>]
```
## Force `git pull`
```bash
git fetch --all
git reset --hard [<remote_name>/<branch_name>]
git pull [<remote_name> <branch_name>]
```

# `git clean`
```bash
# Delete untracked files
git clean -f
# Delete untacked files and directories
git clean -fd
```

# Stash
```bash
# The `git stash` command takes your uncommitted changes (both staged and unstaged), saves them away for later use, and then reverts them from your working copy.
# `git stash` temporarily shelves (or stashes) changes you've made to your working copy so you can work on something else, and then come back and re-apply them later on.
git stash
...
# Reapply previously stashed changes.
git stash apply
```

# Fetch
```bash
# 원격저장소 변화정보 가져오기.
git fetch
```

# Reset
```bash
# Move the current branch backward by `1` commits, effectively removing the two snapshots we just created from the project history. Remember that this kind of reset should only be used on unpublished commits. Never perform the above operation if you’ve already pushed your commits to a shared repository.
git reset HEAD~1
```

# Branch
## Show Branch List
```bash
# 로컬저장소 브랜치 목록확인
git branch
# 원격저장소 브랜치 목록확인
git branch -r
# 모든 브랜치 목록확인
git branch -a
```
## Create Branch
```bash
git checkout -b <branch_name>
```
## Get Remote Branch
```bash
git checkout -t <branch_name>
```
### Delete Branch
```bash
git branch -d <branch_name>
```
### Create Branch from Another branch
```bash
git pull origin <branch_name1>
git checkout <branch_name1>
git checkout -b <branch_name2> <branch_name2>
```
## Switch to Branch
```bash
git checkout <branch_name>
# Force switch
# `-f` is short for `--force`, which is an alias for `--discard-changes`)
git switch -f <branch_name>
```
## Merge
```bash
# `<branch_name>`에서 작업 내용을 가져옵니다.
git merge <branch_name>
```
## Abort Merge
```bash
git merge --abort
```

# ".DS_Store" etc
## Delete ALL
```bash
find . -name .DS_Store -print0 | xargs -0 git rm -r --ignore-unmatch -f
find . -name *.pyc -print0 | xargs -0 git rm -r --ignore-unmatch -f
find . -name .ipynb_checkpoints -print0 | xargs -0 git rm -r --ignore-unmatch -f
```
## Stop from Generating
```bash
defaults write com.apple.desktopservices DSDontWriteNetworkStores true
```
## ".gitignore"
```bash
echo .DS_Store >> .gitignore
echo *.pyc* >> .gitignore
echo __pycache__/ >> .gitignore
echo .ipynb_checkpoints/ >> .gitignore

# Apply to already existing files
git rm -r --cached .
```

# Large Files
```bash
# 50 * 1024 * 1024 = 52428800
git config --global http.postBuffer 52428800
git config --global http.postBuffer 103809024
```

# Convert from CRLF to LF
```bash
git config --global core.autocrlf true
```

# Resolve Conflicts
- `git checkout origin main` -> `git pull origin main` -> `git checkout <branch_name>` -> `git merge origin/main` -> Add changes to the staging area -> Commit Changes -> Push
  
# "fatal: in unpopulated submodule '...'" 해결 방법
```bash
rm -rf .git
# Submodule로 이동한 뒤
git rm --cached . -rf
```

# Personal Access Token (PAT)
- Reference: https://git-scm.com/book/ko/v2/Git-%EC%84%9C%EB%B2%84-SSH-%EA%B3%B5%EA%B0%9C%ED%82%A4-%EB%A7%8C%EB%93%A4%EA%B8%B0
```bash
cd ~/.ssh
ls
# id_dsa나 id_rsa라는 파일 이름이 보일 것이고 이에 같은 파일명의 .pub 라는 확장자가 붙은 파일이 하나 더 있을 것이다. 그중 .pub 파일이 공개키이고 다른 파일은 개인키다.
pbcopy < ~/.ssh/id_rsa.pub # Copy key
vim config
# 아래 내용 추가
Host github.com
  IdentityFile ~/.ssh/id_rsa
  User git
ssh -T git@github.com # 적용
git remote remove origin
git remote add origin git@github.com:.../....git
git push --set-upstream origin main
```

# Roll Back Specific File
```bash
git checkout <commit_id> <path>
```

# Split Repository
```bash
# Reference: https://ashortday.tistory.com/58
# Example
# /Users/jongbeomkim/Desktop/workspace/machine_learning
git subtree split -P <folder_name_to_split> -b <branch_name>
cd <new_directory>
git init
git pull /Users/jongbeomkim/Desktop/workspace/machine_learning <branch_name>
git remote add origin <remote_address>
git push --set-upstream origin main
cd /Users/jongbeomkim/Desktop/workspace/machine_learning
git rm -rf <splitted_folder_name>
git add .
git commit -m "<commit_message>"
git push
```
