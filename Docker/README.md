# Install
- 명령 프롬프트에서 `docker images` 실행 시 정상적으로 메시지가 출력되면 설치가 완료된 것입니다.

# Docker Hub
- 레지스트리 같은 것. 이곳에서 다운받은 것 = Image. 이 행위: Pull
## Image
- image -> container: Run
- https://hub.docker.com/ -> `Explore` -> `Containers`
- https://docs.docker.com/ -> `Reference` -> `Command-line reference` -> `Docker CLI (docker)`

# `docker`
## `docker pull`
```
docker pull IMAGE
```
## `docker run`, `docker stop`
```
docker run [OPTIONS] IMAGE [COMMAND]
```
- Container를 만듭니다.
- 하나의 Image로 여러 개의 Container를 만들 수 있습니다.
- `[OPTIONS]`
	- `--name NAME`
	- `-p` (= `--publish`): Publish a container's port(s) to the host.
		```
		docker run --name ws3 -p 8080:80 httpd
		```
## `docker start`
- `docker stop` 후 재실행합니다.
## `docker rm`
```
docker rm [OPTIONS] CONTAINER
```
- 실행 중인 Containers는 삭제할 수 없습니다.
## `docker rmi`
```
docker rmi [OPTIONS] IMAGE
```
### `docker rm --force`
## `docker images`
## `docker ps`
```
docker ps [OPTIONS]
```
- List containers.
- `[OPTIONS]
	- `-a` (= `--all`): Show all containers (default shows just running).
## `docker logs`
```
docker logs [OPTIONS] CONTAINER
```
- Fetch the logs of a container.
- `[OPTIONS]`
	- `-f` (= `--follow`): 	Follow log output.
## `docker exec`
```
docker exec [OPTIONS] CONTAINER COMMAND
```
- Run a command in a running container
```
docker exec ws pwd
```
- By default docker exec command runs in the same working directory set when container was created.
- `[OPTIONS]`
	- `-i` (= `--interactive`):	Keep STDIN open even if not attached.
	- `-t` (= `--tty`): Allocate a pseudo-TTY.
		```
		docker exec -it ws /bin/sh
		```
		```
		docker exec -it ws /bin/bash
		```
		
`pwd`, `ls -al`
`/usr/local/apache2/htdocs/`