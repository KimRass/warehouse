# Protocol
- Source: https://www.cloudflare.com/ko-kr/learning/network-layer/what-is-a-protocol/
- ***In networking, a protocol is a set of rules for formatting and processing data. Network protocols are like a common language for computers. The computers within a network may use vastly different software and hardware; however, the use of protocols enables them to communicate with each other regardless.***
- Standardized protocols are like a common language that computers can use, similar to how two people from different parts of the world may not understand each other's native languages, but they can communicate using a shared third language. If one computer uses the Internet Protocol (IP) and a second computer does as well, they will be able to communicate — just as the United Nations relies on its 6 official languages to communicate amongst representatives from all over the globe. But if one computer uses IP and the other does not know this protocol, they will be unable to communicate.
## Internet Protocol Suite (= TCP/IP)
- ***The Internet protocol suite, commonly known as TCP/IP, is the set of communications protocols used in the Internet and similar computer networks. The current foundational protocols in the suite are the Transmission Control Protocol (TCP) and the Internet Protocol (IP).***
### IP (Internet Protocol)
- Source: https://phoenixnap.com/blog/ipv4-vs-ipv6?utm_term=&utm_campaign=&utm_source=adwords&utm_medium=ppc&hsa_ad=545481684602&hsa_kw=&hsa_cam=14630707084&hsa_grp=127504961735&hsa_net=adwords&hsa_mt=&hsa_ver=3&hsa_tgt=dsa-410675953091&hsa_src=g&hsa_acc=2931804872&gclid=Cj0KCQiAwqCOBhCdARIsAEPyW9kZfHueBP3LaktZigCwbk7CJJiOsS9WfYEGuXAZnRgNbfdvMXI9IK4aAh6oEALw_wcB
- IPv4 and IPv6
	- *Even with 4.3 billion possible addresses, that’s not nearly enough to accommodate all of the currently connected devices.* Device types are far more than just desktops. Now there are smartphones, hotspots, IoT, smart speakers, cameras, etc.
	- IPv4 addresses are set to finally run out, making IPv6 deployment the only viable solution left for the long-term growth of the Internet.
	- IPv6 is considered as an enhanced version of the older IPv4 protocol, as it supports a significantly larger number of nodes than the latter.
	- IPv4: 32-bit number (2^32 = 4,294,967,296), IPv6: 128-bit number
#### IP Address
- ***An IP address, or Internet Protocol address, is a complex string of numbers that acts as a binary identifier for devices across the Internet. In short, an IP address is the address that computers, servers and other devices use to identify one another online. The vast majority of IP addresses are arranged into four sets of digits – i.e., 12.34.56.78.***
- Source: https://www.techtarget.com/searchnetworking/definition/port-number
- Localhost is the default name used to establish a connection with a computer. The IP address is usually 127.0.0.1.
##### Public IP Address(= External IP Address)
- Source: https://www.avast.com/c-ip-address-public-vs-private#gref
- ***A public IP address is an IP address that can be accessed directly over the internet and is assigned to your network router by your internet service provider (ISP). Your personal device also has a private IP that remains hidden when you connect to the internet through your router’s public IP.***
- *Public IP addresses can be traced back to your ISP, which can potentially reveal your general geographical location.*
- *Websites also use IP tracking to analyze online behavior patterns, making it easier for them to determine if the same individual visits the site repeatedly. Websites can then use these patterns to predict your preferences.*
##### Private IP Address(= Internal IP Address, Local IP Address)
- Source: https://www.avast.com/c-ip-address-public-vs-private#gref
- ***A private IP address is the address your network router assigns to your device. Each device within the same network is assigned a unique private IP address — this is how devices on the same internal network talk to each other.***
- ***Private IP addresses let devices connected to the same network communicate with one another without connecting to the entire internet. By making it more difficult for an external host or user to establish a connection, private IPs help bolster security within a specific network, like in your home or office. This is why you can print documents via wireless connection to your printer at home, but your neighbor can’t send their files to your printer accidentally.***
- ***Your private IP address exists within specific private IP address ranges reserved by the Internet Assigned Numbers Authority (IANA) and should never appear on the internet.*** There are millions of private networks across the globe, all of which include devices assigned private IP addresses within these ranges:
	- Class A: 10.0.0.0 — 10.255.255.255
	- Class B: 172.16.0.0 — 172.31.255.255 
	- Class C: 192.168.0.0 — 192.168.255.255 
- ***These might not seem like wide ranges, but they don’t really need to be. Because these IP addresses are reserved for private network use only, they can be reused on different private networks all over the world — without consequence or confusion.***
### TCP (Transmission Control Protocol)
- The Transmission Control Protocol (TCP) is one of the main protocols of the Internet protocol suite
### UDP (User Datagram Protocol)
- UDP is a faster but less reliable alternative to TCP at the transport layer. It is often used in services like video streaming and gaming, where fast data delivery is paramount.
### HTTP (HyperText Transfer Protocol)
- ***HTTP is the foundation of the World Wide Web, the Internet that most users interact with.***
#### HTTP Response Status Codes
-  404 Not Found
	- Source: https://en.wikipedia.org/wiki/HTTP_404
	- This error message is to indicate that *the browser was able to communicate with a given server, but the server could not find what was requested.*
#### HTTPS (HTTP Secure)
- ***The problem with HTTP is that it is not encrypted — any attacker who intercepts an HTTP message can read it. HTTPS corrects this by encrypting HTTP messages.***
- HTTPS is the secure and encrypted version of HTTP. 
- Source: https://namu.wiki/w/TLS#s-1.2
- TLS를 사용해 암호화된 연결을 하는 HTTP를 HTTPS(HTTP Secure)라고 하며, 당연히 웹사이트 주소 역시 "http://"가 아닌 "https://"로 시작된다. 기본 포트는 80번이 아닌 443번을 쓴다.
- 흔히 TLS와 HTTPS를 혼동하는 경우가 많은데, 둘은 유사하긴 하지만 엄연히 다른 개념임을 알아두자. TLS는 다양한 종류의 보안 통신을 하기 위한 프로토콜이며, HTTPS는 TLS 위에 HTTP 프로토콜을 얹어 보안된 HTTP 통신을 하는 프로토콜이다. 다시 말해 TLS는 HTTP뿐만이 아니라 FTP, SMTP와 같은 여타 프로토콜에도 적용할 수 있으며, HTTPS는 TLS와 HTTP가 조합된 프로토콜만을 가리킨다.
- HTTP는 HTTPS와 달리 암호화되지 않았으며, 중간자 공격 또는 도청의 가능성이 높으므로 HTTPS만큼 안전하지 않다.
### TLS (Transport Layer Security)/SSL (Secure Sockets Layer)
- ***TLS is the protocol HTTPS uses for encryption.*** TLS used to be called SSL.
- Source: https://namu.wiki/w/TLS#s-1.2
- 인터넷에서의 정보를 암호화해서 송수신하는 프로토콜. 넷스케이프 커뮤니케이션스사가 개발한 SSL(Secure Sockets Layer)에 기반한 기술로, 국제 인터넷 표준화 기구에서 표준으로 인정받은 프로토콜이다. TCP 443 포트를 사용한다. 표준에 명시된 정식 명칭은 TLS지만 아직도 SSL이라는 용어가 많이 사용되고 있다.
- 인터넷을 사용한 통신에서 보안을 확보하려면 두 통신 당사자가 서로가 신뢰할 수 있는 자임을 확인할 수 있어야 하며, 서로간의 통신 내용이 제 3자에 의해 도청되는 것을 방지해야 한다. 따라서 서로 자신을 신뢰할 수 있음을 알리기 위해 전자 서명이 포함된 인증서를 사용하며, 도청을 방지하기 위해 통신 내용을 암호화한다. 이러한 통신 규약을 묶어 정리한 것이 바로 TLS. 주요 웹브라우저 주소창에 자물쇠 아이콘이 뜨는 것으로 TLS의 적용 여부를 확인할 수 있다.
- 예를 들어 인터넷 뱅킹을 하기 위해 은행의 사이트에 방문했을 때, 고객은 그 사이트가 정말 은행의 사이트가 맞는지 아니면 해커가 만든 가짜 피싱 사이트인지 확인할 수 있어야 하며, 은행 역시 자신의 서비스에 접속한자가 해당 고객이 맞는지 아니면 고객의 컴퓨터와 서버 사이에서 내용을 가로채고자 하는 해커인지 확인할 수 있어야 한다. 그리고 은행과 고객 간의 통신 내용이 다른 해커에게 도청되지 않도록 내용을 숨겨야 한다. 이럴 때 바로 은행과 고객 간에 TLS를 사용한 연결을 맺어 안전하게 통신을 할 수 있다.
- 쉽게 요약해서, 먼저 서로가 어떤 TLS 버전을 사용 가능한지를 확인하고, 인증서를 사용해 서로를 믿을 수 있는지 확인한 뒤, 서로간의 통신에 쓸 암호를 교환하는 것이다. 그 다음부터는 서로 교환한 암호를 사용해 제3자가 도청할 수 없는 암호화된 통신을 하면 된다.
### SSH (Secure SHell)
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
- ***A network consists of two or more computers that are linked in order to share resources (such as printers and CDs), exchange files, or allow electronic communications.*** The computers on a network may be linked through cables, telephone lines, radio waves, satellites, or infrared light beams.
## Network Traffic
- Source: https://www.fortinet.com/resources/cyberglossary/network-traffic
- *Network traffic is the amount of data moving across a computer network at any given time. Network traffic, also called data traffic, is broken down into data packets and sent over a network before being reassembled by the receiving device or computer.*
- *Network traffic has two directional flows, north-south and east-west. Traffic affects network quality because an unusually high amount of traffic can mean slow download speeds or spotty Voice over Internet Protocol (VoIP) connections. Traffic is also related to security because an unusually high amount of traffic could be the sign of an attack.*
### North-South Traffic
- *North-south traffic refers to client-to-server traffic that moves between the data center and the rest of the network (i.e., a location outside of the data center).* 
### East-West Traffic
- *East-west traffic refers to traffic within a data center, also known as server-to-server traffic.*
### Data Packets
- *When data travels over a network or over the internet, it must first be broken down into smaller batches so that larger files can be transmitted efficiently. The network breaks down, organizes, and bundles the data into data packets so that they can be sent reliably through the network and then opened and read by another user in the network.* Each packet takes the best route possible to spread network traffic evenly. 

## LAN (Local Area Network)
- Source: https://www.cisco.com/c/en/us/products/switches/what-is-a-lan-local-area-network.html
- ***A local area network (LAN) is a collection of devices connected together in one physical location, such as a building, office, or home.*** A LAN can be small or large, ranging from a home network with one user to an enterprise network with thousands of users and devices in an office or school.
- Regardless of size, a LAN's single defining characteristic is that it connects devices that are in a single, limited area. In contrast, a wide area network (WAN) or metropolitan area network (MAN) covers larger geographic areas. Some WANs and MANs connect many LANs together.
- Source: https://www.cloudflare.com/ko-kr/learning/network-layer/what-is-a-router/
- There are several types of routers, but most routers pass data between LANs (local area networks) and WANs (wide area networks). A LAN is a group of connected devices restricted to a specific geographic area. A LAN usually requires a single router.

## VPN(Virtual Private Network)
- Source: https://en.wikipedia.org/wiki/Virtual_private_network
- A virtual private network (VPN) extends a private network across a public network and enables users to send and receive data across shared or public networks as if their computing devices were directly connected to the private network. The benefits of a VPN include increases in functionality, security, and management of the private network. It provides access to resources inaccessible on the public network and is typically used for telecommuting workers. Encryption is common, although not an inherent part of a VPN connection
- Source: https://www.vpn-mentors.com/popular/vpns-101-vpnmentors-vpn-guide-newbies/?keyword=what%20is%20vpn&geo=2410&device=&cq_src=google_ads&cq_cmp=990646304&cq_term=what%20is%20vpn&cq_plac=&cq_net=g&cq_plt=gp&gclid=Cj0KCQiA2NaNBhDvARIsAEw55hirE5SUS7P_mYPwC-PA6SS3f8A7p5TCRPX1xgYblYssT1Uscg5qF48aAtLTEALw_wcB
- VPNs encrypt all the data you send over the internet.
	- When you’re connected to a VPN server, all your internet traffic is encrypted. This means that nobody can see what you’re doing online.
	- Encryption stops hackers from seeing sensitive information that you enter into websites, like your passwords. This is especially important if you’re using public WiFi because it’s easy for cybercriminals to monitor your connection on public networks. But a VPN makes sure that even if someone stole your data, they wouldn’t be able to decrypt it or even understand it.
- Your VPN also protects your privacy
	- Websites and services use your IP to determine your location. When you connect to a VPN server, your IP address won’t be visible. Because they can no longer see your real IP, they can’t see where you’re located.
- Source: https://www.avast.com/c-ip-address-public-vs-private#gref
- To browse the internet more anonymously, you can hide your IP address by connecting through a security protocol: a proxy server, a VPN, or the Tor browser.
	
# DNS (Domain Name System)
- Source: https://whatismyipaddress.com/dns
- Every time you visit a website, you are interacting with the largest ***distributed database*** in the world. This massive database is collectively known as the DNS, or the Domain Name System.
## Domain Name
- When you input a URL like www.example.com/index into a web browser, its domain name is example.com. Basically, a domain name is the human-friendly version of an IP address.***
## TLD (Top Level Domain)
- ***A Top Level Domain refers to the part of a domain name that comes after the period. For instance, the TLD of example.com is COM.*** While there’s an ever-expanding number of domain names, there’s a relatively static number of Top Level Domains; .com, .edu and .org are just a few key examples.
## Root Server
- ***Specialized computers called root servers store the IP addresses of each Top Level Domain’s registries. Therefore, the first stop that the DNS makes when it resolves, or translates, a domain name is at its associated root server.*** From there, the requested domain name is sent along to a Domain Name Resolver, or DNR.
## DNR (Domain Name Resolver)
- ***Domain Name Resolvers, or resolvers, are located within individual Internet Service Providers and organizations. They respond to requests from root servers to find the necessary IP addresses.*** Since the root server already recognizes the .com, .edu or other part of the equation, it simply has to resolve the remainder of the request. It usually does this instantly, and the information is forwarded to the user’s PC.
## URL (Universal Resource Locator)
- ***The universal resource locator, or URL, is an entire set of directions, and it contains extremely detailed information. The domain name is one of the pieces inside of a URL***

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
### Nginx
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

# Port
- Source: https://www.cloudflare.com/ko-kr/learning/network-layer/what-is-a-computer-port/
- Vastly different types of data flow to and from a computer over the same network connection. ***The use of ports helps computers understand what to do with the data they receive.***
- Suppose Bob transfers an MP3 audio recording to Alice using the File Transfer Protocol (FTP). If Alice's computer passed the MP3 file data to Alice's email application, the email application would not know how to interpret it. ***But because Bob's file transfer uses the port designated for FTP (port 21), Alice's computer is able to receive and store the file. Meanwhile, Alice's computer can simultaneously load HTTP webpages using port 80, even though both the webpage files and the MP3 sound file flow to Alice's computer over the same WiFi connection.***
- In total , the system has 65535 ports to be used for providing services.
- Some of the most commonly used ports, along with their associated networking protocol, are:
	- Port 22: SSH
	- ***Port 80: The common standard port for HTTP.
	- ***Port 443: HTTPS. All HTTPS web traffic goes to port 443. Network services that use HTTPS for encryption, such as DNS over HTTPS, also connect at this port.***
	- Source: https://www.techtarget.com/searchnetworking/definition/port-number
	- ***Port 8080: Port number 8080 is usually used for web servers.*** When a port number is added to the end of the domain name, it drives traffic to the web server. However, users can not reserve port 8080 for secondary web servers.
## Port Forwarding
- Source: https://stevessmarthomeguide.com/understanding-port-forwarding/
- ***Port forwarding is a technique that is used to allow external devices access to computers services on private networks. It does this by mapping an external port to an internal IP address and port.***

# Cookie (= HTTP Cookie, Web Cookie, Internet Cookie, Browser Cookie)
- Source: https://en.wikipedia.org/wiki/HTTP_cookie
- ***HTTP cookies are small blocks of data created by a web server while a user is browsing a website and placed on the user's computer or other device by the user’s web browser. Cookies are placed on the device used to access a website, and more than one cookie may be placed on a user’s device during a session.
- ***Cookies serve useful and sometimes essential functions on the web. They enable web servers to store stateful information (such as items added in the shopping cart in an online store) on the user’s device or to track the user's browsing activity (including clicking particular buttons, logging in, or recording which pages were visited in the past). They can also be used to save for subsequent use information that the user previously entered into form fields, such as names, addresses, passwords, and payment card numbers.***
	- *In information technology and computer science, a system is described as stateful if it is designed to remember preceding events or user interactions; the remembered information is called the state of the system.*
- Authentication cookies are commonly used by web servers to authenticate that a user is logged in, and with which account they are logged in. *Without the cookie, users would need to authenticate themselves by logging in on each page containing sensitive information that they wish to access.* The security of an authentication cookie generally depends on the security of the issuing website and the user's web browser, and on whether the cookie data is encrypted. Security vulnerabilities may allow a cookie's data to be read by an attacker, used to gain access to user data, or used to gain access (with the user's credentials) to the website to which the cookie belongs.
- Another popular use of cookies is for logging into websites. When the user visits a website's login page, the web server typically sends the client a cookie containing a unique session identifier. When the user successfully logs in, the server remembers that that particular session identifier has been authenticated and grants the user access to its services.
## Session cookie
- A session cookie (also known as an in-memory cookie, transient cookie or non-persistent cookie) exists only in temporary memory while the user navigates a website. *Session cookies expire or are deleted when the user closes the web browser.* Session cookies are identified by the browser by the absence of an expiration date assigned to them.
- ***Cookies can be used to remember information about the user in order to show relevant content to that user over time.*** For example, a web server might send a cookie containing the username that was last used to log into a website, so that it may be filled in automatically the next time the user logs in.
- ***Many websites use cookies for personalization based on the user's preferences.*** *Users select their preferences by entering them in a web form and submitting the form to the server. The server encodes the preferences in a cookie and sends the cookie back to the browser. This way, every time the user accesses a page on the website, the server can personalize the page according to the user's preferences.* For example, the Google search engine once used cookies to allow users (even non-registered ones) to decide how many search results per page they wanted to see. Also, DuckDuckGo uses cookies to allow users to set the viewing preferences like colors of the web page.

# Credential
- Source: https://en.wikipedia.org/wiki/Digital_credential
- Digital credentials are the digital equivalent of paper-based credentials. *Just as a paper-based credential could be a passport, a driver's license, a membership certificate or some kind of ticket to obtain some service, such as a cinema ticket or a public transport ticket, a digital credential is a proof of qualification, competence, or clearance that is attached to a person.* Also, digital credentials prove something about their owner.
- Sometimes passwords or other means of authentication are referred to as credentials.

# `bs4`
## `BeautifulSoup()`
```python
from bs4 import BeautifulSoup as bs
```
```python
soup = bs(xml,"lxml")
```
### `soup.find_all()`
#### `soup.find_all().find()`
#### `soup.find_all().find().get_text()`
```python
features = ["bjdcode", "codeaptnm", "codehallnm", "codemgrnm", "codesalenm", "dorojuso", "hocnt", "kaptacompany", "kaptaddr", "kaptbcompany",  "kaptcode", "kaptdongcnt", "kaptfax", "kaptmarea", "kaptmarea",  "kaptmparea_136", "kaptmparea_135", "kaptmparea_85", "kaptmparea_60",  "kapttarea", "kapttel", "kapturl", "kaptusedate", "kaptdacnt", "privarea"]
for item in soup.find_all("item"):
    for feature in features:
        try:
            kapt_data.loc[index, feature] = item.find(feature).get_text()
        except:
            continue
```

# `selenium`
## `webdriver`
```python
from selenium import webdriver
```
```python
driver = webdriver.Chrome("chromedriver.exe")
```
### `driver.get()`
```python
driver.get("https://www.google.co.kr/maps/")
```
### `driver.find_element_by_css_selector()`, `driver.find_element_by_tag_name()`, `driver.find_element_by_class_name()`, `driver.find_element_by_id()`, `driver.find_element_by_xpath()`,
#### `driver.find_element_by_*().text`
```python
df.loc[index, "배정초"]=driver.find_element_by_xpath("//\*[@id='detailContents5']/div/div[1]/div[1]/h5").text
```
#### `driver.find_element_by_*().get_attribute()`
```python
driver.find_element_by_xpath("//*[@id='detailTab" +str(j) + "']").get_attribute("text")
```
#### `driver.find_element_by_*().click()`
#### `driver.find_element_by_*().clear()`
```python
driver.find_element_by_xpath('//*[@id="searchboxinput"]').clear()
```
#### `driver.find_element_by_*().send_keys()`
```python
driver.find_element_by_xpath('//*[@id="searchboxinput"]').send_keys(qeury)
```
```python
driver.find_element_by_name('username').send_keys(id)
driver.find_element_by_name('password').send_keys(pw)
```
```python
driver.find_element_by_xpath('//*[@id="wpPassword1"]').send_keys(Keys.ENTER)
```
### `driver.execute_script()`
```python
for j in [4,3,2]:
    button = driver.find_element_by_xpath("//\*[@id='detailTab"+str(j)+"']")
    driver.execute_script("arguments[0].click();", button)
```
### `driver.implicitly_wait()`
```python
driver.implicitly_wait(1)
```
### `driver.current_url`
### `driver.save_screenshot()`
```python
driver.save_screenshot(screenshot_title)
```
## `WebDriverWait()`
### `WebDriverWait().until()`
```python
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
```
```python
WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.XPATH, "//\*[@id='detailContents5']/div/div[1]/div[1]/h5")))
```
- `By.ID`, `By.XPATH`
## `ActionChains()`
```python
from selenium.webdriver import ActionChains
```
```python
module=["MDM","사업비","공사","외주","자재","노무","경비"]

for j in module:
    module_click=driver.find_element_by_xpath("//div[text()='"+str(j)+"']")
    actions=ActionChains(driver)
    actions.click(module_click)
    actions.perform()
```
### `actions.click()`, `actions.double_click()`

# `urllib`
```python
import urllib
```
## `urllib.request`
### `urllib.request.urlopen()`
```python
xml = urllib.request.urlopen(full_url).read().decode("utf-8")
```
### `urllib.request.urlretrieve()`
```python
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
```
- 해당 URL에 연결된 파일을 다운로드합니다.

# `urllib3`
```python
import urllib3
```
## `urllib3.PoolManager()`
### `urllib3.PoolManager().request()`
```python
urllib3.PoolManager().request("GET", url, preload_content=False)
```

# `pathlib`
```python
import pathlib
```
## `pathlib.Path()`
```python
data_dir = pathlib.Path(data_dir)
```

# `requests`
```python
import requests
```
## `requests.get()`
```python
req = requests.get("https://github.com/euphoris/datasets/raw/master/imdb.zip")
```
### `req.content`

# `wget`
````python
import wget
````
## `wget.download()`

# `requests`
```python
import requests
```
## `requests.get()`
```python
req = requests.get(url)
```