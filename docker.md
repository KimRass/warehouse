# List

## List Volumes
```bash
docker volume ls
```

## List Images
```bash
docker images
```

## List Containers
```bash
# List running containers on your Docker host.
# `-a` or `--all`: Show all containers, including stopped ones.
docker ps  # `ps`: Process status.
# Output: e.g.,
# CONTAINER ID   IMAGE         COMMAND                  CREATED          STATUS          PORTS     NAMES
# 3f1b4d53f798   nginx         "nginx -g 'daemon of…"   10 minutes ago   Up 9 minutes    80/tcp    vibrant_morse
# 4c1a1f0d02bc   postgres      "docker-entrypoint.s…"   2 hours ago      Up 2 hours      5432/tcp  happy_morse
```

# Start Container

## 'Dockerfile'
```
WORKDIR <CONTAINER_ROOT_DIR>
# Set the working directory in the container.
# After setting `WORKDIR`, any subsequent instructions that involve file paths (like `COPY`, `RUN`, or `CMD`) are executed relative to that directory.
# e.g., `WORKDIR /app`

COPY <HOST_DIR> <CONTAINER_DIR>
# e.g., `COPY . .`: This copies files from the current directory on the host into `WORKDIR` in the container.

# 이미지를 만들 때 실행할 코드.
RUN pip install --no-cache-dir -r requirements.txt

# 컨테이너가 실행될 때 처음으로 실행되는 프로그램이나 명령어를 뜻합니다.
# Using `--entrypoint` with `docker run`, you can specify a different entry point for the container.
# e.g., `ENTRYPOINT ["/usr/bin/python3"]`
ENTRYPOINT

# 기본 인자나 명령을 정의. `docker run` 실행 시 덮어쓸 수 있음.
CMD
# e.g., `CMD ["app.py"]`
```

## 'docker-compose.yaml'
```yaml
version: <version>  # e.g., `version: "3"`.

services:
    <service_name>:
        image: <image_name>
        container_name: <container_name>
        [depends_on:]
            [<service_name>:]
                [condition: <condition>]
            ...
        [ports:]
            [- <host_port>:<container_port>]
            ...
        environment:
            <KEY>: <value>
        [command: <command>]
        [healthcheck:]
            test: ["CMD", "pg_isready", "-q", "-U", "mlflowuser", "-d", "mlflowdatabase"]
            interval: 10s
            timeout: 5s
            retries: 5
    ...
```

## Build
```bash
# `t`: Same as `--tag`.
# `DOCKERFILE_PATH`: This is the path to the directory containing the Dockerfile (often the current directory is used, represented by `.`).
docker build -t <IMAGE_NAME>[:<TAG>] <DOCKERFILE_PATH>
```

## Run
```bash
# `-i`: This stands for "interactive." It keeps the standard input (stdin) open, even if not attached. This is useful when you want to interact with a running container or run a command that requires user input.
# `-t`: This stands for "tty" (teletypewriter). It allocates a pseudo-TTY (terminal) for the container, which enables a terminal interface. This allows for features like line editing, text formatting, and more interactive capabilities.
# `-d`: When you run a Docker container in detached mode, it runs in the background instead of occupying the terminal session. This allows you to continue using your terminal for other tasks, while the container continues to run.
docker run [-it] [-p <HOST_PORT>:<CONTAINER_PORT>] [--entrypoint <ENTRYPOINT>] [-w <WORKING_DIR>] [-v <VOLUME_NAME>:/<CONTAINER_DIR] --name <CONTAINER_NAME> IMAGE_NAME
# e.g., docker run -it --name my-python-container -v "$(pwd):/workspace" -w /workspace my-python-image
# e.g., docker run --entrypoint "/bin/bash" IMAGE_NAME
```

# End Container

## Stop
```bash
# Sends a `SIGTERM` (signal to terminate) to the main process inside the container, allowing it to perform cleanup tasks like saving state, closing connections, or releasing resources.
# If the container does not stop within a default timeout period (10 seconds by default), Docker sends a `SIGKILL` to forcibly terminate the process.
docker stop <CONTAINER_NAME>  # Graceful shutdown.
```

## Kill
```bash
# Sends a `SIGKILL` directly to the container's main process, immediately terminating it without giving the application any chance to clean up.
# This is faster but can lead to data corruption, incomplete tasks, or resource leaks if the application was in the middle of a critical operation.
docker kill <CONTAINER_NAME>  # Immediate termination.
```

## Remove Container
```bash
docker stop <CONTAINER_NAME>
docker rm <CONTAINER_NAME>
```

## Remove Image
```bash
[docker images]
docker rmi <IMAGE_ID>
```

## Remove Volume
```bash
docker volume rm <VOLUME_NAME>
```

# Volumes

## Create
```bash
docker volume create <VOLUME_NAME>
```

# Mount
```bash
# This mounts the my_volume to the `/CONTAINER_DIR` directory inside the container. Any data written to `/CONTAINER_DIR` will be stored in the volume and persist beyond the life of the container.
docker run -v <VOLUME_NAME>:/<CONTAINER_DIR> <IMAGE_NAME>
```

# Python Virtual Environment
1. Docker Desktop 실행.
2. Set 'Dockerfile':
    ```
    FROM python:3.9

    RUN pip install --no-cache-dir pandas numpy
    ```
3. Build docker image: `docker build -t IMAGE_NAME DOCKERFILE_PATH`
4. Run docker container: `docker run -it --name CONTAINER_NAME IMAGE_NAME`
5. 'devcontainers.json':
    ```json
        {
        "name": "Python Dev Container",
        "build": {
            "dockerfile": "../Dockerfile"
        },
        "customizations": {
            "vscode": {
                "settings": {
                    "terminal.integrated.profiles.linux": {
                        "bash": {
                            "path": "/bin/bash"
                        }
                    },
                    "terminal.integrated.defaultProfile.linux": "bash"
                },
                "extensions": [
                    "ms-python.python",  // Python 확장 설치
                    "ms-toolsai.jupyter" // Jupyter 확장 설치 (필요시)
                ]
            }
        },
        "workspaceFolder": "/workspace",
        "workspaceMount": "source=${localWorkspaceFolder},target=/workspace,type=bind"
    }
    ```
6. Run 'Rebuild Container'

# Install GCC
```sh
sudo apt update
sudo apt install build-essential
sudo apt-get install manpages-dev
``` -->
