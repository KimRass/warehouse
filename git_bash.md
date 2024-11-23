# Branch

## Create New Local Branch
```bash
git branch <BRANCH>
```

# Remote

## Change Remote
```bash
git remote remove origin
git remote add origin https://github.com/OWNER/REPOSITORY
```

- Pull:
    ```bash
    # "fatal: refusing to merge unrelated histories"
    git pull origin main --allow-unrelated-histories
    ```
- Fetch:
    ```bash
    # 원격 저장소의 최신 커밋을 다운로드하여 '로컬 저장소의 원격 브랜치' (`origin/REMOTE_BRANCH`)를 업데이트합니다. '(로컬 저장소의) 로컬 브랜치'는 변경하지 않습니다.
    git fetch origin
    ```
- Rebase:
    ```bash
    git checkout LOCAL_BRANCH
    git fetch origin
    # LOCAL_BRANCH 브랜치의 커밋들이 `origin/REMOTE_BRANCH` 브랜치의 커밋 위로 이동하게 됩니다.
    # `origin/REMOTE_BRANCH` 브랜치의 최신 커밋 위에 LOCAL_BRANCH 브랜치의 커밋이 순차적으로 적용됩니다.
    # LOCAL_BRANCH 브랜치에서 작업한 커밋들이 `origin/REMOTE_BRANCH`의 최신 커밋을 기반으로 다시 "적용"되기 때문에, LOCAL_BRANCH 브랜치의 커밋들이 최신 상태로 갱신됩니다.
    # Ensures the current branch is based on the latest commits in the `origin/REMOTE_BRANCH`.
    # Creates new commit objects, rewriting the history of the branch.
    # REMOTE_BRANCH:    C1 - C2 - C3
    # LOCAL_BRANCH: C1 - C2 - F1 - F2 -> C1 - C2 - C3 - F1' - F2'
    git rebase origin/REMOTE_BRANCH
    ```
- See log:
    ```bash
    # It displays each commit with a short commit hash (abbreviated to 7 characters by default) and the commit message.
    git log --oneline

    # File modification (`M`): This shows that the contents of the file were updated.
    # File deletion (`D`): This indicates that the file was removed from the repository.
    # File addition (`A`): This indicates that a new file was introduced to the repository.
    # The `100644` represents a regular file (non-executable).
    # The `000000` indicates a deleted file.
    ```
- Delete file from history:
    ```bash
    # pip install git-filter-repo
    git filter-repo --invert-paths --path FILE_PATH
    git remote add origin https://github.com/OWNER/REPOSITORY
    # `--all`: 모든 브랜치에 적용.
    git push origin --force --all
    ```
    ```bash
    # 차이점은??
    git filter-branch --force --index-filter 'git rm --cached --ignore-unmatch FILE_PATH' --prune-empty --tag-name-filter cat -- --all
    git push origin --force [--all]
    ```

# Errors

## "send-pack: unexpected disconnect while reading sideband packet\nfatal: the remote end hung up unexpectedly"
- Reference: https://happy-jjang-a.tistory.com/222
- Git에서 기본적으로 한개 파일의 최대 용량이 1MB로 설정되어 있고, 그것을 초과한 파일을 push 할 때 발생하는 오류였다.
```bash
git config --global http.postBuffer 524288000
git config --global ssh.postBuffer 524288000
```
<!-- 
## "this is larger than GitHub's recommended maximum file size of 50.00 MB"
```bash
# Locate files exceeding a certain size.
find . -type f -size +50M
``` -->