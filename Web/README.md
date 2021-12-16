# VPN(Virtual Private Network)
- Source: https://en.wikipedia.org/wiki/Virtual_private_network
- A virtual private network (VPN) extends a private network across a public network and enables users to send and receive data across shared or public networks as if their computing devices were directly connected to the private network. The benefits of a VPN include increases in functionality, security, and management of the private network. It provides access to resources inaccessible on the public network and is typically used for telecommuting workers. Encryption is common, although not an inherent part of a VPN connection
- Source: https://www.vpn-mentors.com/popular/vpns-101-vpnmentors-vpn-guide-newbies/?keyword=what%20is%20vpn&geo=2410&device=&cq_src=google_ads&cq_cmp=990646304&cq_term=what%20is%20vpn&cq_plac=&cq_net=g&cq_plt=gp&gclid=Cj0KCQiA2NaNBhDvARIsAEw55hirE5SUS7P_mYPwC-PA6SS3f8A7p5TCRPX1xgYblYssT1Uscg5qF48aAtLTEALw_wcB
- VPNs encrypt all the data you send over the internet.
	- When you’re connected to a VPN server, all your internet traffic is encrypted. This means that nobody can see what you’re doing online.
	- Encryption stops hackers from seeing sensitive information that you enter into websites, like your passwords. This is especially important if you’re using public WiFi because it’s easy for cybercriminals to monitor your connection on public networks. But a VPN makes sure that even if someone stole your data, they wouldn’t be able to decrypt it or even understand it.
- Your VPN also protects your privacy
	- Websites and services use your IP to determine your location. When you connect to a VPN server, your IP address won’t be visible. Because they can no longer see your real IP, they can’t see where you’re located.
	
# SSL (Secure Sockets Layer)
## TLS (Transport Layer Security)
- Source: https://namu.wiki/w/TLS#s-1.2
- 인터넷에서의 정보를 암호화해서 송수신하는 프로토콜. 넷스케이프 커뮤니케이션스사가 개발한 SSL(Secure Sockets Layer)에 기반한 기술로, 국제 인터넷 표준화 기구에서 표준으로 인정받은 프로토콜이다. TCP 443 포트를 사용한다. 표준에 명시된 정식 명칭은 TLS지만 아직도 SSL이라는 용어가 많이 사용되고 있다.
- 인터넷을 사용한 통신에서 보안을 확보하려면 두 통신 당사자가 서로가 신뢰할 수 있는 자임을 확인할 수 있어야 하며, 서로간의 통신 내용이 제 3자에 의해 도청되는 것을 방지해야 한다. 따라서 서로 자신을 신뢰할 수 있음을 알리기 위해 전자 서명이 포함된 인증서를 사용하며, 도청을 방지하기 위해 통신 내용을 암호화한다. 이러한 통신 규약을 묶어 정리한 것이 바로 TLS. 주요 웹브라우저 주소창에 자물쇠 아이콘이 뜨는 것으로 TLS의 적용 여부를 확인할 수 있다.
- 예를 들어 인터넷 뱅킹을 하기 위해 은행의 사이트에 방문했을 때, 고객은 그 사이트가 정말 은행의 사이트가 맞는지 아니면 해커가 만든 가짜 피싱 사이트인지 확인할 수 있어야 하며, 은행 역시 자신의 서비스에 접속한자가 해당 고객이 맞는지 아니면 고객의 컴퓨터와 서버 사이에서 내용을 가로채고자 하는 해커인지 확인할 수 있어야 한다. 그리고 은행과 고객 간의 통신 내용이 다른 해커에게 도청되지 않도록 내용을 숨겨야 한다. 이럴 때 바로 은행과 고객 간에 TLS를 사용한 연결을 맺어 안전하게 통신을 할 수 있다.
- 쉽게 요약해서, 먼저 서로가 어떤 TLS 버전을 사용 가능한지를 확인하고, 인증서를 사용해 서로를 믿을 수 있는지 확인한 뒤, 서로간의 통신에 쓸 암호를 교환하는 것이다. 그 다음부터는 서로 교환한 암호를 사용해 제3자가 도청할 수 없는 암호화된 통신을 하면 된다.
## HTTPS
- Source: https://namu.wiki/w/TLS#s-1.2
- TLS를 사용해 암호화된 연결을 하는 HTTP를 HTTPS(HTTP Secure)라고 하며, 당연히 웹사이트 주소 역시 "http://"가 아닌 "https://"로 시작된다. 기본 포트는 80번이 아닌 443번을 쓴다.
- 흔히 TLS와 HTTPS를 혼동하는 경우가 많은데, 둘은 유사하긴 하지만 엄연히 다른 개념임을 알아두자. TLS는 다양한 종류의 보안 통신을 하기 위한 프로토콜이며, HTTPS는 TLS 위에 HTTP 프로토콜을 얹어 보안된 HTTP 통신을 하는 프로토콜이다. 다시 말해 TLS는 HTTP뿐만이 아니라 FTP, SMTP와 같은 여타 프로토콜에도 적용할 수 있으며, HTTPS는 TLS와 HTTP가 조합된 프로토콜만을 가리킨다.
- HTTP는 HTTPS와 달리 암호화되지 않았으며, 중간자 공격 또는 도청의 가능성이 높으므로 HTTPS만큼 안전하지 않다.