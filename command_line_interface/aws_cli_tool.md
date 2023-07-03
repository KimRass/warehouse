# Install
Command line installer - All users
```sh
curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
sudo installer -pkg AWSCLIV2.pkg -target /.
```
```sh
which aws

aws --version
```

# List
```sh
aws s3 ls s3://flitto-deliveries-9957/...
```

# Upload
```sh
# `<directory>`의 최하단 폴더 이름을 `s3://flitto-deliveries-9957/...`의 마지막에 넣어 줘야 합니다.
aws s3 cp <directory> s3://flitto-deliveries-9957/... [--recursive] --acl bucket-owner-full-control
```

# Download
```sh
aws s3 cp s3://flitto-deliveries-9957/... <path or directory> [--recursive]
```

# Remove
```sh
aws s3 rm s3://bucket/... [--recursive]
```