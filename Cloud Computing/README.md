# Cloud Computing
- Source: https://en.wikipedia.org/wiki/Cloud_computing
- Cloud computing is the on-demand availability of computer system resources, especially data storage (cloud storage) and computing power, without direct active management by the user. Large clouds often have functions distributed over multiple locations, each location being a data center. Cloud computing relies on sharing of resources to achieve coherence and economies of scale, typically using a "pay-as-you-go" model which can help in reducing capital expenses but may also lead to unexpected operating expenses for unaware users.
## Virtualization
- 다수의 VM으로 분할하는 기술.
- 리소스 효율적 관리
- 반대: Bare-Metal
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