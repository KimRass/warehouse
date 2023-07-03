# Python Environment Path
```sh
cd
ls -a
# .bash_profile 없다면
touch ~/.bash_profile
open ~/.bash_profile
# 내용 추가
export PYTHONPATH="/Users/jongbeom.kim/Desktop/workspace/flitto/data_mgmt"
export GOOGLE_APPLICATION_CREDENTIALS="/Users/jongbeom.kim/Desktop/workspace/Github/Work/flitto-351906-36a591c7de9c.json"
# 수정된 경로 적용
source ~/.bash_profile
# 확인
echo $PATH
```

# "command not found"
```sh
export PATH=%PATH:/bin:/usr/local/bin:/usr/bin
```

# Basic Commands
## `pwd`: Print working directory
## `cd`: Change directory
## `mkdir`: Create a directory
## `rm`: Removes a file.
## `rm -d`: Removes an empty directory.
## `rm -r`: Removes directory which is not empty.
## `ls`
- Source: https://linuxize.com/post/how-to-list-files-in-linux-using-the-ls-command/
```sh
# When used with no options and arguments, `ls` displays a list of the names of all files in the current working directory.
# `option`
	# `-l`: The default output of the `ls` command shows only the names of the files and directories, which is not very informative. ***The `-l` option tells ls to print files in a long listing format. When the long listing format is used, you can see the following file information: The file type, The file permissions, Number of hard links to the file, File owner, File group, File size, Date and Time, File name.***
	# `-a`: To display all files including the hidden files use the `-a` option.
	# `-R`: The `-R` option tells the `ls` command to display the contents of the subdirectories recursively.
ls [<option>] [<file>]
```
## `open`: Open files
## `cp`:  Copy a file to another directory
## `mv`: Move a file
## `sudo`: Execute commands with superuser privileges
## `exit`
- This command will close out the current session in the Terminal.

- Reference: https://engineer-mole.tistory.com/200
- 변수에 액세스할 때 변수명의 앞에 $를 넣는다. 혹은 $넣어서 변수를 {}로 감싼다.
- 변수의 값이 덮어 쓰기되는 것을 방지하기 위해서는 readonly를 사용한다. (`readonly var`)
- `$0`: 스크립트명
- `&&`: And
- `||`: Or
- `-iname`: Turns off case sensitivity.
- `-name`: Retain case sensitivity.

# Change Mode
```sh
# `+`: Grant
# `-`: 박탈
# `x`: Excute
# `r`: Read
# `w`: Write
chmod +x <file>
```

# Operators
## Relational Operators
- `-eq`: Equal to
- `-ne`: Not equal to
- `-gt`: Greater than
- `-lt`: Less than
- `-ge`: Greater than or equal to
- `-le`: Less than or equal to
## Arithmetic Operators
```sh
# 아래 세 줄은 전부 동일한 코드입니다.
((number=number + 1))
number=$((number+1))
number=$(($number+1))
```
## Boolean Operators
- `-not`, `!`: Logical negation
- `-and`, `-a`: Logical AND
- `-or`, `-o`: Logical OR

# Read Value from Input
```sh
read <variable1> <variable2>

# `"<message>"`와 같은 줄에서 변수를 입력하도록 합니다.
read -p "<message>" <variable1> <variable2> ...

# 입력 받은 값을 표시하지 않습니다.
read -s ...
```

# Define and Run Function
```sh
# Define function
<function>() {
	$1...
	$2...
	...
}
...

# Run function
<function> $<argument1> $<argument2> ...
```

# Import Function
```sh
# `<file>`: "....sh"
source <file>
```

# For Statement
```sh
for ... in ...; do
	...
done
```

# IF Statement
```sh
if ...; then
	...
elif ...; then
	...
else
	...
fi
```

# Read File Line by Line
```sh
while read line; do
	...
done < <file>
```

# Find File by Pattern
```sh
# `-type f`: Regular file
# `-type d`: Directory

find <path> -name "<pattern1>"

# Example
find . -name "*.heic" -or -name "*.HEIC"
```

# `wc` (word count)
```sh
# `-w`: Count words.
# `-c`: Count bytes.
# `-m`: Count characters.
# `-l`: Count lines.
wc -l <file>

# To omit the <file>
wc -l < <file>
```
## 현재 디렉토리의 하위 파일 개수
```sh
find . -type f | wc -l
```
## csv 파일의 행 수
```sh
sed -e ":top;s/,//$(head -n1 <path/to/csv> | grep -o "," | wc -l);t end;N; b top;:end;s/\n//g" <path/to/csv> | wc -l
```

# Check `charset`
```sh
# For a file
file -bi <file>
# For a server
locale
```

# Convert File Encoding
```sh
iconv -c -f <encoding1> -t <encoding2> <file1> > <file2>

# Example
iconv -c -f euc-kr -t utf-8 stock_flitto.html > stock_flitto_enc.html
```

# `echo`
```sh
# 개행하지 않고 같은 줄에 출력합니다.
echo -n ...
```

# `cat` (concatenate)
```sh
# 파일의 내용을 순서대로 출력
cat <file1> <file2> ...

# 입력한 내용으로 새로운 파일을 만듭니다.
# 내용을 입력하고 "ctrl + d"를 눌러 저장한다.)
# 기존 파일 내용을 지우고 처음부터 새로 입력합니다. (Write or overwrite)
cat > <file>
# 기존 파일 내용에 연속해서 기록합니다. (Append)
cat >> <file>

# Merge files into the one.
cat <file1> <file2> > <file3>
# Copy
cat <file1> > <file2>
```
## Create Empty File
```sh
# 파일이 없을 경우 새로 빈 파일을 생성하고 파일이 있을 경우 파일은 그대로 둔 채 그 내용만 삭제합니다.
cat /dev/null > <file>
```

# `curl` (Client URL)
```sh
# `<url>`의 html 파일을 `<file>`에 저장합니다.
curl -s <url> > <file>
```
## Send JSON
```sh
# `--request`와 `-X`는 서로 동일합니다.
# `--header`와 `-H`는 서로 동일합니다.
# Example
curl --request POST \
    --header "Content-Type: application/json" \
    --data "$json" \
    $slack_webhook_url
```
## Download File
```sh
curl -L -O <url>
# Example curl -L -O https://github.com/kthworks/KoreanSTT-DeepSpeech2/raw/main/aihub_character_vocabs.csv
```

# `grep`
```sh
# `<option>`
	# `-o`: 매치되는 문자열만 표시합니다.
	# `-E`: Extended grep includes meta characters that were added later.
grep [<option>] <pattern> <file>
```
## Text between two texts
```sh
echo $<text> | grep -P -o "(?<=(<text1> )).*(?= <text2>)"
```
## Extract Text from File
```sh
grep "<regex>" <file>

# Example
text=$(grep ".*현재가.*</dd>$" stock_flitto_enc.html)
```
## Extract Text from Variable
```sh
<variable1>=$(echo $<variable2> | grep -o -E "<regex>")
```
## Length of Text
```sh
# Example
s3_dir_without_slash=${s3_dir:0:${#s3_dir} - 1}
```

# `sed`
## Replace Text
```sh
# `-e`, `--expression`: `sed -e` 이후에 여러 개의 expression을 사용할 때 필요합니다.
sed 's/<text1>/<text2/g'
# Example
... | sed "s|$sub_dir||g"
```

# `cut`
## Split Text by Character
```sh
# `-c1-2`: The first two characters
echo $<text> | cut -d "<character>" -f1
```

# `sort`
```sh
# `<option>`
	# `-r`, `--reverse`: 역순으로 정렬합니다.
	# `-u`, `--unique`: 정렬 후 중복을 제거합니다.
sort [<option>] <file>
... | sort [<option>]
```

# `awk`
```sh
awk "<pattern>" <file>
awk "{<action>}" <file>
awk "<pattern> {<action>}" <file>

# `$0`: 전체 컬럼
# `$n`: `n`번째 컬럼
```
```sh
... | awk '{print $4}'
```

# `file`
```sh
# 파일의 유형을 출력합니다.
file <file>
```

# Move or Copy File
```sh
# Move
# 이름을 변경하여 옮기는 것도 가능합니다.
mv <file> <dir>

# Copy
cp <file> <dir>
```

# Regular Expression
```sh
# Directory or directory name
...
# File name or basename
${<text>##*/}
```