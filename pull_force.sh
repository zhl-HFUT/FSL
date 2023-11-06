#!/bin/bash

# 定义代理和配置文件的路径
NETWORK_CONFIG_FILE="/etc/network_turbo"

# 检查网络配置文件是否存在
if [ -e "$NETWORK_CONFIG_FILE" ]; then
    echo "Network configuration file found at $NETWORK_CONFIG_FILE"
    # 在这里添加处理网络配置文件的命令
    sh /etc/network_tuobo
else
    echo "Setting proxy"
fi

git fetch --all
git reset --hard origin/master
