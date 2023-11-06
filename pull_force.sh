#!/bin/bash

# 定义代理和配置文件的路径
NETWORK_CONFIG_FILE="/etc/network_turbo"

# 检查网络配置文件是否存在
if [ -e "$NETWORK_CONFIG_FILE" ]; then
    echo "Network configuration file found at $NETWORK_CONFIG_FILE"
    # 在这里添加处理网络配置文件的命令
    source /etc/network_turbo
    echo $https_proxy
    echo $http_proxy
else
    echo "Setting proxy"
    export https_proxy="http://${hostip}:${http_hostport}"
    export http_proxy="http://${hostip}:${http_hostport}"
    export ALL_PROXY="socks5://${hostip}:${socks_hostport}"
    export all_proxy="socks5://${hostip}:${socks_hostport}"
    echo $ALL_PROXY
    echo $all_proxy
    echo $https_proxy
    echo $http_proxy
fi

git fetch --all
git reset --hard origin/master
