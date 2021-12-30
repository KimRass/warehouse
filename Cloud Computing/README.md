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
- ***A standard package of software—known as a container—bundles an application’s code together with the related configuration files and libraries, and with the dependencies required for the app to run. This allows developers and IT pros to deploy applications seamlessly across environments.***
- ***The problem of an application failing to run correctly when moved from one environment to another is as old as software development itself. Such problems typically arise due to differences in configuration underlying library requirements and other dependencies.***
- ***Containers address this problem by providing a lightweight, immutable infrastructure for application packaging and deployment. An application or service, its dependencies, and its configuration are packaged together as a container image. The containerized application can be tested as a unit and deployed as a container image instance to the host operating system.***
- ***This way, containers enable developers and IT professionals to deploy applications across environments with little or no modification.***
- ***Since containers share the host OS, they don’t need to boot an OS or load libraries. This enables containers to be much more efficient and lightweight.*** Containerized applications can start in seconds, and many more instances of the application can fit onto the machine as compared to a VM scenario. The shared OS approach has the added benefit of reduced overhead when it comes to maintenance, such as patching and updates.
- ***Though containers are portable, they’re constrained to the operating system they’re defined for. For example, a container for Linux can’t run on Windows, and vice versa.***
## IaaS
- Virtualization, Servers, Storage, Networking
- Examples: AWS EC2
## PaaS
- Source: https://www.redhat.com/en/topics/cloud-computing/iaas-vs-paas-vs-saas
- Virtualization, Servers, Storage, Networking
- OS, Middleware, Runtime
- You write the code, build, and manage your apps, but you do it without the headaches of software updates or hardware maintenance. The environment to build and deploy is provided for you.
- Examples: AWS Elastic Beanstalk, Heroku, and Red Hat OpenShift.
- Middleware
	- Source: https://en.wikipedia.org/wiki/Middleware
	- Middleware is a type of computer software that provides services to software applications beyond those available from the operating system. It can be described as "software glue".
	- Source: https://azure.microsoft.com/en-us/overview/what-is-middleware/
	- Middleware is software that lies between an operating system and the applications running on it. Essentially functioning as hidden translation layer, middleware enables communication and data management for distributed applications. It’s sometimes called plumbing, as it connects two applications together so data and databases can be easily passed between the "pipe".
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
- Application
	- Source: https://en.wikipedia.org/wiki/Application_software
	- An application program (application or app for short) is a computer program designed to carry out a specific task other than one relating to the operation of the computer itself, typically to be used by end-users.

# Clustering
- Source: https://medium.com/@mena.meseha/difference-between-distributed-and-cluster-aca9d50c2c44
- It means that multiple servers are grouped together to achieve the same business and can be regarded as one computer.

# AWS (Amazon Web Services)
## AWS Region
- ***Amazon cloud computing resources are housed in highly available data center facilities in different areas of the world (for example, North America, Europe, or Asia).*** Each data center location is called an AWS Region.
- ***Each Amazon EC2 Region is designed to be isolated from the other Amazon EC2 Regions. This achieves the greatest possible fault tolerance and stability.***
- ***When you view your resources, you see only the resources that are tied to the Region that you specified. This is because Regions are isolated from each other, and we don't automatically replicate resources across Regions.***
### Availability Zone (AZ)
- Each AWS Region contains multiple distinct locations called Availability Zones, or AZs. ***Each Availability Zone is engineered to be isolated from failures in other Availability Zones. Each is engineered to provide inexpensive, low-latency network connectivity to other Availability Zones in the same AWS Region.*** By launching instances in separate Availability Zones, you can protect your applications from the failure of a single location.
- ***Availability Zones are multiple, isolated locations within each Region.***
## Edge Location
- 임시 데이터 저장소. 속도 빨라짐.
## Amazon Resource Name (ARN)
- 모든 리소스의 고유 아이디.
## IAM (Identity and Access Management)
- MFA (Multi-Factor Authentication)
## Amazon EC2 (Elastic Compute Cloud)
### AMI (Amazon Machine Image)
- ***An Amazon Machine Image (AMI) is a template that contains a software configuration (for example, an operating system, an application server, and applications).***
### Instance
- ***An instance is a virtual server in the cloud. Its configuration at launch is a copy of the AMI that you specified when you launched the instance.***
- ***You can launch different types of instances from a single AMI.*** An instance type essentially determines the hardware of the host computer used for your instance. Each instance type offers different compute and memory capabilities. Select an instance type based on the amount of memory and computing power that you need for the application or software that you plan to run on the instance.
- ***Your instances keep running until you stop, hibernate, or terminate them, or until they fail.*** If an instance fails, you can launch a new one from the AMI.
- ***You are not charged for additional instance usage while the instance is in a stopped state.*** A minimum of one minute is charged for every transition from a stopped state to a running state. If the instance type was changed while the instance was stopped, you will be charged the rate for the new instance type after the instance is started.
- On-Demand Instance
	- There is no long-term commitment required when you purchase On-Demand Instances. ***You pay only for the seconds that your On-Demand Instances are in the running state. The price per second for a running On-Demand Instance is fixed.***
- Spot Instance
- `Delete on Termination`: *EBS volumes persist independently from the running life of an EC2 instance. However, you can choose to automatically delete an EBS volume when the associated instance is terminated.*
- Security Group
	- `Type`
		- The protocol to open to network traffic. *You can choose a common protocol, such as SSH (for a Linux instance), RDP (for a Windows instance), and HTTP and HTTPS to allow Internet traffic to reach your instance.* You can also manually enter a custom port or port ranges.
		- Instance에 원격으로 접속할 것이라면 `SSH`로 설정합니다.
	- `Source`: *Determines the traffic that can reach your instance. Specify a single IP address, or an IP address range in CIDR notation (for example, 203.0.113.5/32).*
#### Amazon Elastic Block Store (EBS)
- ***Amazon Elastic Block Store (Amazon EBS) provides block level storage volumes for use with EC2 instances.*** EBS volumes behave like raw, unformatted block devices. ***You can mount these volumes as devices on your instances. EBS volumes that are attached to an instance are exposed as storage volumes that persist independently from the life of the instance.***
- ***We recommend Amazon EBS for data that must be quickly accessible and requires long-term persistence.***
## Amazon S3 (Simple Storage Service)
- Source: https://en.wikipedia.org/wiki/Amazon_S3
- Amazon S3 or Amazon Simple Storage Service is a service offered by Amazon Web Services (AWS) that provides object storage through a web service interface. Amazon S3 can be employed to store any type of object, which allows for uses like storage for Internet applications, backup and recovery, disaster recovery, data archives, data lakes for analytics, and hybrid cloud storage.
## Amazon SageMaker
- Source: https://en.wikipedia.org/wiki/Amazon_SageMaker
- Amazon SageMaker is a cloud machine-learning platform that was launched in November 2017. SageMaker enables developers to create, train, and deploy machine-learning (ML) models in the cloud.
- SageMaker enables developers to operate at a number of levels of abstraction when training and deploying machine learning models. ***At its highest level of abstraction, SageMaker provides pre-trained ML models that can be deployed as-is. In addition, SageMaker provides a number of built-in ML algorithms that developers can train on their own data.*** Further, SageMaker provides managed instances of TensorFlow and Apache MXNet, where developers can create their own ML algorithms from scratch. Regardless of which level of abstraction is used, ***a developer can connect their SageMaker-enabled ML models to other AWS services, such as the Amazon DynamoDB database for structured data storage, AWS Batch for offline batch processing, or Amazon Kinesis for real-time processing.***
### Amazon DynamoDB
- Source: https://en.wikipedia.org/wiki/Amazon_DynamoDB
- Amazon DynamoDB is a fully managed proprietary NoSQL database service that supports key–value and document data structures.
- Partition key: The partition key is part of the table's primary key. *It is a hash value that is used to retrieve items from your table.*
- Sort key (optional): You can use a sort key as the second part of a table's primary key. *The sort key allows you to sort or search among all items sharing the same partition key.*
## Amazon Batch
## Amazon Kinesis
## Amazon EMR (Elastic MapReduce)
- Source: https://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-what-is-emr.html
- ***Amazon EMR (previously called Amazon Elastic MapReduce) is a managed cluster platform that simplifies running big data frameworks, such as Apache Hadoop and Apache Spark, on AWS to process and analyze vast amounts of data.*** Using these frameworks and related open-source projects, you can process data for analytics purposes and business intelligence workloads. Amazon EMR also lets you transform and move large amounts of data into and out of other AWS data stores and databases, such as Amazon Simple Storage Service (Amazon S3) and Amazon DynamoDB.
## Amazon RDS (Relational Database Service)
- Amazon Relational Database Service (Amazon RDS) is a web service that makes it easier to set up, operate, and scale a relational database in the AWS Cloud. It provides cost-efficient, resizable capacity for an industry-standard relational database and manages common database administration tasks.
- Engine types: Amazon Aurora, MySQL, MariaDB, PostgreSQL, Oracle, Microsoft SQL Server
- *Multi-AZ deployments: This option will has Amazon RDS maintain a synchronous standby replica in a different Availability Zone than the DB instance. Amazon RDS will automatically fail over to the standby in the case of a planned or unplanned outage of the primary.*
- Subnet group: Subnet group defines which subnets and IP ranges the DB instance can use in the Virtual Private Cloud (VPC) you chose.
- Public Access
	- Yes: *Amazon EC2 instances and devices outside the VPC can connect to your database. Choose one or more VPC security groups that specify which EC2 instances and devices inside the VPC can connect to the database.*
	- No: RDS will not assign a public IP address to the database. *Only Amazon EC2 instances and devices inside the VPC can connect to your database.*
## CloudFront
