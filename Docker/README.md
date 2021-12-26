# Docker Hub
- https://hub.docker.com/
- 레지스트리 같은 것. 이곳에서 다운받은 것 = Image. 이 행위: Pull
## Image
- image -> container: Run
- `Explore` -> `Containers`
- https://docs.docker.com/ -> `Reference` -> `Command-line reference` -> `Docker CLI (docker)`

# Docker CLI (Command Line Interface)
- Source: https://docs.docker.com/reference/
## `docker pull`
```
docker pull IMAGE
```
- Pull an image or a repository (set of images) from a registry.
## `docker run`, `docker stop`
```
docker run [OPTIONS] IMAGE [COMMAND]
```
- Run a command in a new container.
- `[OPTIONS]`
	- `--name NAME`
	- `-p` (= `--publish`): Publish a container's port(s) to the host.
		```
		docker run -p LOCAL HOST PORT:CONTAINER PORT IMAGE
		```
	- `-v` (= `volume`): The `-v` flag mounts the current working directory into the container. When the host directory of a bind-mounted volume doesn’t exist, Docker will automatically create this directory on the host for you.
		```
		docker run -v HOST DIRECTORY:CONTAINER DIRECTORY IMAGE
		```
## `docker exec`
```
docker exec [OPTIONS] CONTAINER COMMAND
```
- ***Run a command in a running container.***
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
## `docker start`
- `docker stop` 후 재실행합니다.
## `docker rm`
```
docker rm [OPTIONS] CONTAINER
```
- Remove one or more containers.
- `[OPTIONS]`
	- `-f` (= `--force`): Force the removal of a running container (uses SIGKILL).
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
	
# nano?
- `apt update` -> `apt install nano` -> `nano index.html`