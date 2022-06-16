import ftplib


def getFile(ftp, filename):
    try:
        ftp.retrbinary(
            "RETR " + filename, open(filename, mode="wb").write
        )
    except:
        print("Error")


ftp = ftplib.FTP("ftp.nluug.nl")
ftp.login("anonymous", "ftplib-example-1")

# data = list()
ftp.cwd("/pub/")
getFile(ftp, "README.nluug")
# ftp.dir(data.append)

ftp.quit()
 
for line in data:
    print(line)
    
data[0]