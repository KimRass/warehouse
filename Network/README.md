# Protocol
- Source: https://www.cloudflare.com/ko-kr/learning/network-layer/what-is-a-protocol/
- ***In networking, a protocol is a set of rules for formatting and processing data. Network protocols are like a common language for computers. The computers within a network may use vastly different software and hardware; however, the use of protocols enables them to communicate with each other regardless.***
- Standardized protocols are like a common language that computers can use, similar to how two people from different parts of the world may not understand each other's native languages, but they can communicate using a shared third language. If one computer uses the Internet Protocol (IP) and a second computer does as well, they will be able to communicate — just as the United Nations relies on its 6 official languages to communicate amongst representatives from all over the globe. But if one computer uses IP and the other does not know this protocol, they will be unable to communicate.
## TCP (Transport Layer Protocol)
- TCP is a transport layer protocol that ensures reliable data delivery. ***TCP is meant to be used with IP, and the two protocols are often referenced together as TCP/IP.***
## UDP (User Datagram Protocol)
- UDP is a faster but less reliable alternative to TCP at the transport layer. It is often used in services like video streaming and gaming, where fast data delivery is paramount.
## HTTP (HyperText Transfer Protocol)
- HTTP is the foundation of the World Wide Web, the Internet that most users interact with.
## HTTPS (HTTP Secure)
- ***The problem with HTTP is that it is not encrypted — any attacker who intercepts an HTTP message can read it. HTTPS corrects this by encrypting HTTP messages.***
- Source: https://namu.wiki/w/TLS#s-1.2
- TLS를 사용해 암호화된 연결을 하는 HTTP를 HTTPS(HTTP Secure)라고 하며, 당연히 웹사이트 주소 역시 "http://"가 아닌 "https://"로 시작된다. 기본 포트는 80번이 아닌 443번을 쓴다.
- 흔히 TLS와 HTTPS를 혼동하는 경우가 많은데, 둘은 유사하긴 하지만 엄연히 다른 개념임을 알아두자. TLS는 다양한 종류의 보안 통신을 하기 위한 프로토콜이며, HTTPS는 TLS 위에 HTTP 프로토콜을 얹어 보안된 HTTP 통신을 하는 프로토콜이다. 다시 말해 TLS는 HTTP뿐만이 아니라 FTP, SMTP와 같은 여타 프로토콜에도 적용할 수 있으며, HTTPS는 TLS와 HTTP가 조합된 프로토콜만을 가리킨다.
- HTTP는 HTTPS와 달리 암호화되지 않았으며, 중간자 공격 또는 도청의 가능성이 높으므로 HTTPS만큼 안전하지 않다.
### TLS (Transport Layer Security)
#### SSL (Secure Sockets Layer)
- ***TLS is the protocol HTTPS uses for encryption.*** TLS used to be called SSL.
- Source: https://namu.wiki/w/TLS#s-1.2
- 인터넷에서의 정보를 암호화해서 송수신하는 프로토콜. 넷스케이프 커뮤니케이션스사가 개발한 SSL(Secure Sockets Layer)에 기반한 기술로, 국제 인터넷 표준화 기구에서 표준으로 인정받은 프로토콜이다. TCP 443 포트를 사용한다. 표준에 명시된 정식 명칭은 TLS지만 아직도 SSL이라는 용어가 많이 사용되고 있다.
- 인터넷을 사용한 통신에서 보안을 확보하려면 두 통신 당사자가 서로가 신뢰할 수 있는 자임을 확인할 수 있어야 하며, 서로간의 통신 내용이 제 3자에 의해 도청되는 것을 방지해야 한다. 따라서 서로 자신을 신뢰할 수 있음을 알리기 위해 전자 서명이 포함된 인증서를 사용하며, 도청을 방지하기 위해 통신 내용을 암호화한다. 이러한 통신 규약을 묶어 정리한 것이 바로 TLS. 주요 웹브라우저 주소창에 자물쇠 아이콘이 뜨는 것으로 TLS의 적용 여부를 확인할 수 있다.
- 예를 들어 인터넷 뱅킹을 하기 위해 은행의 사이트에 방문했을 때, 고객은 그 사이트가 정말 은행의 사이트가 맞는지 아니면 해커가 만든 가짜 피싱 사이트인지 확인할 수 있어야 하며, 은행 역시 자신의 서비스에 접속한자가 해당 고객이 맞는지 아니면 고객의 컴퓨터와 서버 사이에서 내용을 가로채고자 하는 해커인지 확인할 수 있어야 한다. 그리고 은행과 고객 간의 통신 내용이 다른 해커에게 도청되지 않도록 내용을 숨겨야 한다. 이럴 때 바로 은행과 고객 간에 TLS를 사용한 연결을 맺어 안전하게 통신을 할 수 있다.
- 쉽게 요약해서, 먼저 서로가 어떤 TLS 버전을 사용 가능한지를 확인하고, 인증서를 사용해 서로를 믿을 수 있는지 확인한 뒤, 서로간의 통신에 쓸 암호를 교환하는 것이다. 그 다음부터는 서로 교환한 암호를 사용해 제3자가 도청할 수 없는 암호화된 통신을 하면 된다.
## SSH (Secure SHell)
- Source: https://www.ucl.ac.uk/isd/what-ssh-and-how-do-i-use-it
- SSH or Secure Shell is a network communication protocol that enables two computers to communicate (c.f http or hypertext transfer protocol, which is the protocol used to transfer hypertext such as web pages) and share data. An inherent feature of ssh is that the communication between the two computers is encrypted meaning that it is suitable for use on insecure networks.
- Source: https://www.ssh.com/academy/ssh/protocol
- The protocol works in the client-server model, which means that the connection is established by the SSH client connecting to the SSH server. ***The SSH client drives the connection setup process and uses public key cryptography to verify the identity of the SSH server.*** After the setup phase the SSH protocol uses strong symmetric encryption and hashing algorithms to ensure the privacy and integrity of the data that is exchanged between the client and server.
- There are several options that can be used for user authentication. The most common ones are passwords and public key authentication.
- The public key authentication method is primarily used for automation and sometimes by system administrators for single sign-on. It has turned out to be much more widely used than we ever anticipated. ***The idea is to have a cryptographic key pair - public key and private key - and configure the public key on a server to authorize access and grant anyone who has a copy of the private key access to the server. The keys used for authentication are called SSH keys.***
## OAuth
- Source: https://www.varonis.com/blog/what-is-oauth/
- OAuth is an open-standard authorization protocol or framework that provides applications the ability for “secure designated access.” For example, you can tell Facebook that it’s OK for ESPN.com to access your profile or post updates to your timeline without having to give ESPN your Facebook password. This minimizes risk in a major way: In the event ESPN suffers a breach, your Facebook password remains safe.
- OAuth doesn’t share password data but instead uses authorization tokens to prove an identity between consumers and service providers. OAuth is an authentication protocol that allows you to approve one application interacting with another on your behalf without giving away your password.
- ***The simplest example of OAuth in action is one website saying “hey, do you want to log into our website with other website’s login?” In this scenario, the only thing the first website – let’s refer to that website as the consumer – wants to know is that the user is the same user on both websites and has logged in successfully to the service provider – which is the site the user initially logged into, not the consumer.***
- Your smart home devices – toaster, thermostat, security system, etc. – probably use some kind of login data to sync with each other and allow you to administer them from a browser or client device. These devices use what OAuth calls confidential authorization. That means they hold onto the secret key information, so you don’t have to log in over and over again.
- ***OAuth is about authorization and not authentication. Authorization is asking for permission to do stuff. Authentication is about proving you are the correct person because you know things. OAuth doesn’t pass authentication data between consumers and service providers – but instead acts as an authorization token of sorts.***
## SAML (Security Assertion Markup Language)
- Source: https://www.varonis.com/blog/what-is-oauth/
- SAML is an alternative federated authentication standard that many enterprises use for Single-Sign On (SSO). SAML enables enterprises to monitor who has access to corporate resources.
- There are many differences between SAML and OAuth. SAML uses XML to pass messages, and OAuth uses JSON. OAuth provides a simpler mobile experience, while SAML is geared towards enterprise security. That last point is a key differentiator: OAuth uses API calls extensively, which is why mobile applications, modern web applications, game consoles, and Internet of Things (IoT) devices find OAuth a better experience for the user. SAML, on the other hand, drops a session cookie in a browser that allows a user to access certain web pages – great for short-lived work days, but not so great when have to log into your thermostat every day.

# Network
- Source: https://fcit.usf.edu/network/chap1/chap1.htm
- A network consists of two or more computers that are linked in order to share resources (such as printers and CDs), exchange files, or allow electronic communications. The computers on a network may be linked through cables, telephone lines, radio waves, satellites, or infrared light beams.
## LAN (Local Area Network)
- Source: https://www.cisco.com/c/en/us/products/switches/what-is-a-lan-local-area-network.html
- A local area network (LAN) is a collection of devices connected together in one physical location, such as a building, office, or home. A LAN can be small or large, ranging from a home network with one user to an enterprise network with thousands of users and devices in an office or school.
- Regardless of size, a LAN's single defining characteristic is that it connects devices that are in a single, limited area. In contrast, a wide area network (WAN) or metropolitan area network (MAN) covers larger geographic areas. Some WANs and MANs connect many LANs together.
## VPN(Virtual Private Network)
- Source: https://en.wikipedia.org/wiki/Virtual_private_network
- A virtual private network (VPN) extends a private network across a public network and enables users to send and receive data across shared or public networks as if their computing devices were directly connected to the private network. The benefits of a VPN include increases in functionality, security, and management of the private network. It provides access to resources inaccessible on the public network and is typically used for telecommuting workers. Encryption is common, although not an inherent part of a VPN connection
- Source: https://www.vpn-mentors.com/popular/vpns-101-vpnmentors-vpn-guide-newbies/?keyword=what%20is%20vpn&geo=2410&device=&cq_src=google_ads&cq_cmp=990646304&cq_term=what%20is%20vpn&cq_plac=&cq_net=g&cq_plt=gp&gclid=Cj0KCQiA2NaNBhDvARIsAEw55hirE5SUS7P_mYPwC-PA6SS3f8A7p5TCRPX1xgYblYssT1Uscg5qF48aAtLTEALw_wcB
- VPNs encrypt all the data you send over the internet.
	- When you’re connected to a VPN server, all your internet traffic is encrypted. This means that nobody can see what you’re doing online.
	- Encryption stops hackers from seeing sensitive information that you enter into websites, like your passwords. This is especially important if you’re using public WiFi because it’s easy for cybercriminals to monitor your connection on public networks. But a VPN makes sure that even if someone stole your data, they wouldn’t be able to decrypt it or even understand it.
- Your VPN also protects your privacy
	- Websites and services use your IP to determine your location. When you connect to a VPN server, your IP address won’t be visible. Because they can no longer see your real IP, they can’t see where you’re located.

# Client-Server Model
## Host
- Source: https://4sight.mt/blog/whats-the-difference-between-host-and-server/
- Think of a host as a machine that can be connected to a device within the network. As mentioned above, this could include your personal computer, your work laptop or even your faithful iPhone.
- Source: https://learntomato.flashrouters.com/what-is-a-client-what-is-a-server-what-is-a-host/
- Suppose you want to download an image from another computer on your network. That computer is “hosting” the image and therefore, it is the host computer. On the other hand, if that same computer downloads an image from your computer, your computer becomes the host computer.
- Your computer can be a host to other computers. Likewise, your router can be a host to other routers. ***But a host must have an assigned IP address. Therefore, modems, hubs, and switches are not considered hosts because they do not have assigned IP addresses.***
## Client
- Source: https://learntomato.flashrouters.com/what-is-a-client-what-is-a-server-what-is-a-host/
- A client is a computer hardware device or software that accesses a service made available by a server. The server is often (but not always) located on a separate physical computer.
- 클라이언트는 일반적으로 웹 브라우저를 의미합니다.
## Server
- Source: https://learntomato.flashrouters.com/what-is-a-client-what-is-a-server-what-is-a-host/
- A server is a physical computer dedicated to run services to serve the needs of other computers. Depending on the service that is running, it could be a file server, database server, home media server, print server, or web server.
- Source: https://4sight.mt/blog/whats-the-difference-between-host-and-server/
- First off, a server can be both software and hardware. Its role is to provide a service to any device that is connected to the network. But not all connected devices are hosts.
- Devices using these type of services are called Clients and can also be both hardware or software. A server can serve multiple users at the same time from either the same device or different devices entirely.
## Web Server
- Source: https://www.wpbeginner.com/glossary/apache/
- Wondering what the heck is a web server? Well a web server is like a restaurant host. When you arrive in a restaurant, the host greets you, checks your booking information and takes you to your table. Similar to the restaurant host, the web server checks for the web page you have requested and fetches it for your viewing pleasure. However, A web server is not just your host but also your server. Once it has found the web page you requested, it also serves you the web page.
- So basically a web server is the software that receives your request to access a web page. It runs a few security checks on your HTTP request and takes you to the web page. Depending on the page you have requested, the page may ask the server to run a few extra modules while generating the document to serve you. It then serves you the document you requested.
- Source: https://www.infoworld.com/article/2077354/app-server-web-server-what-s-the-difference.html
- A Web server handles the HTTP protocol. When the Web server receives an HTTP request, it responds with an HTTP response, such as sending back an HTML page. To process a request, a Web server may respond with a static HTML page or image, send a redirect.
- 클라이언트의 요청(Request)을 WAS에 보내고, WAS가 처리한 결과를 클라이언트에게 전달(응답, Response)한다.
### Apache HTTP Server
- Source: https://www.wpbeginner.com/glossary/apache/
- Apache is the most widely used web server software. Developed and maintained by Apache Software Foundation, Apache is an open source software available for free. It runs on 67% of all webservers in the world. It is fast, reliable, and secure. It can be highly customized to meet the needs of many different environments by using extensions and modules.
## WAS (Web Application Server)
- Source: https://www.infoworld.com/article/2077354/app-server-web-server-what-s-the-difference.html
- ***While a Web server mainly deals with sending HTML for display in a Web browser, an application server provides access to business logic for use by client application programs.***
- Such application server clients can include GUIs (graphical user interface) running on a PC, a Web server, or even other application servers. The information traveling back and forth between an application server and its client is not restricted to simple display markup. Instead, the information is program logic. Since the logic takes the form of data and method calls and not static HTML, the client can employ the exposed business logic however it wants.
- Source: https://codechasseur.tistory.com/25
- WAS는 웹 서버와 웹 컨테이너가 합쳐진 형태로서, 웹 서버 단독으로는 처리할 수 없는 데이터베이스의 조회나 다양한 로직 처리가 필요한 동적 컨텐츠를 제공한다. 덕분에 사용자의 다양한 요구에 맞춰 웹 서비스를 제공할 수 있다. WAS는 JSP, Servlet 구동환경을 제공해주기 때문에 웹 컨테이너 혹은 서블릿 컨테이너라고도 불린다.
- ***WAS는 DB 조회 및 다양한 로직을 처리하는 데 집중해야 한다. 따라서 단순한 정적 컨텐츠는 웹 서버에게 맡기며 기능을 분리시켜 서버 부하를 방지한다.*** 만약 WAS가 정적 컨텐츠 요청까지 처리하면, 부하가 커지고 동적 컨텐츠 처리가 지연되면서 수행 속도가 느려지고 이로 인해 페이지 노출 시간이 늘어나는 문제가 발생하여 효율성이 크게 떨어진다.
- Source: https://gmlwjd9405.github.io/2018/10/27/webserver-vs-was.html
- Web Server만을 이용한다면 사용자가 원하는 요청에 대한 결과값을 모두 미리 만들어 놓고 서비스를 해야 한다. 하지만 이렇게 수행하기에는 자원이 절대적으로 부족하다.
### Web Container
- 웹 서버가 보낸 JSP, PHP 등의 파일을 수행한 결과를 다시 웹 서버로 보내주는 역할을 함.
### Apache Tomcat
- Source: https://en.wikipedia.org/wiki/Apache_Tomcat
- Apache Tomcat (called "Tomcat" for short) is a free and open-source implementation of the Jakarta Servlet, Jakarta Expression Language, and WebSocket technologies. Tomcat provides a "pure Java" HTTP web server environment in which Java code can run.

# Authentication vs. Authorization
- https://www.sailpoint.com/identity-library/difference-between-authentication-and-authorization/
- Simply put, authentication is the process of verifying who someone is, whereas authorization is the process of verifying what specific applications, files, and data a user has access to. The situation is like that of an airline that needs to determine which people can come on board. The first step is to confirm the identity of a passenger to make sure they are who they say they are. Once a passenger’s identity has been determined, the second step is verifying any special services the passenger has access to, whether it’s flying first-class or visiting the VIP lounge.
- In the digital world, authentication and authorization accomplish these same goals. Authentication is used to verify that users really are who they represent themselves to be. Once this has been confirmed, authorization is then used to grant the user permission to access different levels of information and perform specific functions, depending on the rules established for different types of users.
## Authentication
- ***Authentication verifies who the user is.***
- ***Authentication works through passwords, one-time pins, biometric information, and other information provided or entered by the user.***
- Authentication is visible to and partially changeable by the user.
- Authentication is the first step of a good identity and access management process.
## Authorization
- ***Authorization determines what resources a user can access.***
- Authorization works through settings that are implemented and maintained by the organization.
- ***Authorization always takes place after authentication.***
- Authorization isn’t visible to or changeable by the user.

# Apache Log4j
- Source: https://en.wikipedia.org/wiki/Log4j
- Apache Log4j is a Java-based logging utility. It is part of the Apache Logging Services, a project of the Apache Software Foundation. Log4j is one of several Java logging frameworks.
- Source: https://www.wsj.com/articles/what-is-the-log4j-vulnerability-11639446180
- Software developers use the Log4j framework to record user activity and the behavior of applications. Distributed free by the nonprofit Apache Software Foundation, Log4j has been downloaded millions of times and is among the most widely used tools to collect information across corporate computer networks, websites and applications.
## The Log4j Vulnerability
- The Log4j flaw allows attackers to execute code remotely on a target computer, which could let them steal data, install malware or take control. Exploits discovered recently include hacking systems to mine cryptocurrency. Other hackers have built malware to hijack computers for large-scale assaults on internet infrastructure, cyber researchers have found.