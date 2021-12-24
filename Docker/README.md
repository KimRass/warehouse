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
- Containers를 확인합니다.
### `docker ps -a`
- All
## `docker logs`
```
docker logs IMAGE
```
### `docker -f logs`
- Log가 계속 뜹니다.