# git init
- `git init` is a one-time command you use during the initial setup of a new repository.
- Executing git init creates a .git subdirectory in the current working directory, which contains all of the necessary Git metadata for the new repository.

# git status

# git config
- The `git config` command is a convenience function that is used to set Git configuration values on a global or local project level.
```bash
git config --global user.email "rmx1000@yonsei.ac.kr"
git config --global user.name "KimRass"
```
- Global level configuration(`--global`) is user-specific, meaning it is applied to an operating system user. Global configuration values are stored in a file that is located in a user's home directory.

# git remote
- The `git remote` command lets you create, view, and delete connections to other repositories. Remote connections are more like bookmarks rather than direct links into other repositories. Instead of providing real-time access to another repository, they serve as convenient names that can be used to reference a not-so-convenient URL.
```bash
git remote -v
```
- List the remote connections you have to other repositories. With `-v`, it includes the URL of each connection.
## git remote add
```bash
git remote add <name> <url>
```
```bash
git remote add origin "https://github.com/KimRass/Programming.git"
```
- Create a new connection to a remote repository. After adding a remote, you’ll be able to use `＜name＞` as a convenient shortcut for `＜url＞` in other Git commands.
## git remote remove
```bash
git remote rm <name>
```
- Remove the connection to the remote repository called `＜name＞`.
## git remote rename
```bash
git remote rename <old-name> <new-name>
```
- Rename a remote connection from `＜old-name＞` to `＜new-name＞`.
## git remote set-url
```bash
git remote set-url origin "git@github.com:KimRass/Work.git"
```
- https에서 ssh URL로 전환.
- SSH Public Key 등록한 다음에도 Git clone 시 ID/Password 인증을 요구하는 경우에는 git이 SSH로 생성된 것이 아니고, http로 생성된  경우입니다. 이 경우 현재 git의 URL을 `git remote -v`로  확인하고 SSH URL로 전환시켜줘야 합니다.

# git add
```bash
git add .
```
- The `git add` command adds a change in the working directory to the staging area. It tells Git that you want to include updates to a particular file in the next commit. However, git add doesn't really affect the repository in any significant way—changes are not actually recorded until you run `git commit`.

# git commit
```bash
git commit -m "commit"
```
- The `git commit` command is used to commit a snapshot of the staging directory to the repositories commit history.
```bash
git commit --amend "commit"
```
- 스테이징에 추가된 내용을 반영해주는 동시에 커밋 메시지도 변경해줍니다. 따라서 변경할 내용이 없을 때도 커밋메시지를 변경하고 싶을 때 자주 사용합니다.

# git push
- The `git push` command is used to upload local repository content to a remote repository.
- Pushing has the potential to overwrite changes, caution should be taken when pushing.
- To prevent you from overwriting commits, Git won’t let you push when it results in a non-fast-forward merge in the destination repository.
```bash
git bash
```
```bash
git push <remote-name> <branch-name>
```
```bash
git push origin master
```
```bash
git push -f origin master
```

# git fetch
- 원격저장소 변화정보 가져오기.

# git pull
```bash
git pull origin master
```

# git stash
- The `git stash` command takes your uncommitted changes (both staged and unstaged), saves them away for later use, and then reverts them from your working copy.
- `git stash` temporarily shelves (or stashes) changes you've made to your working copy so you can work on something else, and then come back and re-apply them later on.
## git stash pop
- Reapply previously stashed changes.
## git stash apply

# git clone
```bash
git clone "git@github.com:KimRass/Work.git"
```
- When you clone a repository with git clone, it automatically creates a remote connection called origin pointing back to the cloned repository. This is useful for developers creating a local copy of a central repository, since it provides an easy way to pull upstream changes or publish local commits. This behavior is also why most Git-based projects call their central repository origin.

# git reflog

# git reset
```bash
git reset HEAD~1
```
- Move the current branch backward by `1` commits, effectively removing the two snapshots we just created from the project history. Remember that this kind of reset should only be used on unpublished commits. Never perform the above operation if you’ve already pushed your commits to a shared repository.

# git revert

# ETC
## ssh-keygen
- ssh public key 생성.
## cat ~/.ssh/id_rsa.pub
## ssh -T git@github.com