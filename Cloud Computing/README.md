# Cloud Computing
- Source: https://en.wikipedia.org/wiki/Cloud_computing
- Cloud computing is the on-demand availability of computer system resources, especially data storage (cloud storage) and computing power, without direct active management by the user. Large clouds often have functions distributed over multiple locations, each location being a data center. Cloud computing relies on sharing of resources to achieve coherence and economies of scale, typically using a "pay-as-you-go" model which can help in reducing capital expenses but may also lead to unexpected operating expenses for unaware users.
## Virtualization
- 다수의 VM으로 분할하는 기술.
- 리소스 효율적 관리
- 반대: Bare-Metal
- Source: https://azure.microsoft.com/en-us/overview/what-is-a-container/#overview
- When people think about virtualization, virtual machines (VMs) often come to mind. In fact, virtualization can take many forms, and containers are one of those.
### VMs (Virtual Machines)
- ***At a high level, VMs virtualize the underlying hardware so that multiple operating system (OS) instances can run on the hardware. Each VM runs an OS and has access to virtualized resources representing the underlying hardware.***
- VMs have many benefits. These include the ability to run different operating systems on the same server, more efficient and cost-effective utilization of physical resources, and faster server provisioning. On the flip side, each VM contains an OS image, libraries, applications, and more, and therefore can become quite large.
#### Container
- Just as shipping industries use physical containers to isolate different cargos—for example, to transport in ships and trains—software development technologies increasingly use an approach called containerization.
- A standard package of software—known as a container—bundles an application’s code together with the related configuration files and libraries, and with the dependencies required for the app to run. This allows developers and IT pros to deploy applications seamlessly across environments.
- The problem of an application failing to run correctly when moved from one environment to another is as old as software development itself. Such problems typically arise due to differences in configuration underlying library requirements and other dependencies.
- Containers address this problem by providing a lightweight, immutable infrastructure for application packaging and deployment. ***An application or service, its dependencies, and its configuration are packaged together as a container image. The containerized application can be tested as a unit and deployed as a container image instance to the host operating system.***
- ***This way, containers enable developers and IT professionals to deploy applications across environments with little or no modification.***
- ***A container virtualizes the underlying OS and causes the containerized app to perceive that it has the OS—including CPU, memory, file storage, and network connections—all to itself.*** Because the differences in underlying OS and infrastructure are abstracted, as long as the base image is consistent, ***the container can be deployed and run anywhere.*** For developers, this is incredibly attractive.
- ***Since containers share the host OS, they don’t need to boot an OS or load libraries. This enables containers to be much more efficient and lightweight.*** Containerized applications can start in seconds, and many more instances of the application can fit onto the machine as compared to a VM scenario. The shared OS approach has the added benefit of reduced overhead when it comes to maintenance, such as patching and updates.
- ***Though containers are portable, they’re constrained to the operating system they’re defined for. For example, a container for Linux can’t run on Windows, and vice versa.***
## OS (Operating System)
- Examples: Windows, Linux, MacOS, Android, iOS
- Privileged Instruction: 시스템 요소들과 소통하는 명령.
- 하나의 하드웨어 시스템당 OS는 1개만 돌아갈 수 있음.
- 일반 프로그램들은 특권 명령이 필요 없어 여러 개를 동시에 수행 가능
## IaaS
- Virtualization, Servers, Storage, Networking
- Examples: AWS EC2
## PaaS
- Source: https://www.redhat.com/en/topics/cloud-computing/iaas-vs-paas-vs-saas
- Virtualization, Servers, Storage, Networking
- OS, Middleware, Runtime
- You write the code, build, and manage your apps, but you do it without the headaches of software updates or hardware maintenance. The environment to build and deploy is provided for you.
- Examples: AWS Elastic Beanstalk, Heroku, and Red Hat OpenShift.
### Heroku
- Source: https://en.wikipedia.org/wiki/Heroku
- Heroku is a cloud platform as a service (PaaS) supporting several programming languages. One of the first cloud platforms, Heroku has been in development since June 2007, when it supported only the Ruby programming language, ***but now supports Java, Node.js, Scala, Clojure, Python, PHP, and Go.*** For this reason, Heroku is said to be a polyglot platform as it has features for a developer to build, run and scale applications in a similar manner across most languages. Heroku was acquired by Salesforce.com in 2010 for $212 million.
#### Heroku Dynos
- Source: https://www.heroku.com/dynos
- The Heroku Platform uses the container model to run and scale all Heroku apps. The containers used at Heroku are called “dynos.” Dynos are isolated, virtualized Linux containers that are designed to execute code based on a user-specified command. Your app can scale to any specified number of dynos based on its resource demands. Heroku’s container management capabilities provide you with an easy way to scale and manage the number, size, and type of dynos your app may need at any given time.
## SaaS
- Source: https://www.redhat.com/en/topics/cloud-computing/iaas-vs-paas-vs-saas
- Virtualization, Servers, Storage, Networking
- OS, Middleware, Runtime
- Applications, Data
- Software updates, bug fixes, and general software maintenance are handled by the provider and the user connects to the app via a dashboard or API. ***There’s no installation of the software on individual machines and group access to the program is smoother and more reliable.***
- You’re already familiar with a form of SaaS if you have an email account with a web-based service like Outlook or Gmail, for example, as you can log into your account and get your email from any computer, anywhere. 

# AWS (Amazon Web Services)
## 글로벌 서비스
### IAM (Identity and Access Management)
### Amazon S3 (Amazon Simple Storage Service)
- Source: https://en.wikipedia.org/wiki/Amazon_S3
- Amazon S3 or Amazon Simple Storage Service is a service offered by Amazon Web Services (AWS) that provides object storage through a web service interface. Amazon S3 can be employed to store any type of object, which allows for uses like storage for Internet applications, backup and recovery, disaster recovery, data archives, data lakes for analytics, and hybrid cloud storage.
### Amazon SageMaker
- Source: https://en.wikipedia.org/wiki/Amazon_SageMaker
- Amazon SageMaker is a cloud machine-learning platform that was launched in November 2017. SageMaker enables developers to create, train, and deploy machine-learning (ML) models in the cloud.
- SageMaker enables developers to operate at a number of levels of abstraction when training and deploying machine learning models. ***At its highest level of abstraction, SageMaker provides pre-trained ML models that can be deployed as-is. In addition, SageMaker provides a number of built-in ML algorithms that developers can train on their own data.*** Further, SageMaker provides managed instances of TensorFlow and Apache MXNet, where developers can create their own ML algorithms from scratch. Regardless of which level of abstraction is used, ***a developer can connect their SageMaker-enabled ML models to other AWS services, such as the Amazon DynamoDB database for structured data storage, AWS Batch for offline batch processing, or Amazon Kinesis for real-time processing.***
### Amazon DynamoDB
- Source: https://en.wikipedia.org/wiki/Amazon_DynamoDB
- Amazon DynamoDB is a fully managed proprietary NoSQL database service that supports key–value and document data structures.
### Amazon Batch
### Amazon Kinesis
### CloudFront
## 지역 서비스
- 특정 리전 기반.
## Region
- Each Amazon EC2 Region is designed to be isolated from the other Amazon EC2 Regions. This achieves the greatest possible fault tolerance and stability.
- When you view your resources, you see only the resources that are tied to the Region that you specified. This is because Regions are isolated from each other, and we don't automatically replicate resources across Regions.
- 서버의 물리적 위치
- 리전별로 제공 서비스 다름
### Availability Zone (AZ)
- Availability Zones are multiple, isolated locations within each Region.
## Edge Location
- 임시 데이터 저장소. 속도 빨라짐.
## Amazon Resource Name (ARN)
- 모든 리소스의 고유 아이디.
