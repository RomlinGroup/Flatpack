#!/bin/bash

if command -v iptables-legacy > /dev/null 2>&1; then
    IPTABLES_CMD=iptables-legacy
else
    IPTABLES_CMD=iptables
fi

$IPTABLES_CMD -P OUTPUT DROP
$IPTABLES_CMD -A OUTPUT -o lo -j ACCEPT
$IPTABLES_CMD -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
$IPTABLES_CMD -A OUTPUT -p udp --dport 53 -j ACCEPT
$IPTABLES_CMD -A OUTPUT -p tcp --dport 53 -j ACCEPT
$IPTABLES_CMD -A OUTPUT -p tcp --dport 443 -j ACCEPT
$IPTABLES_CMD -A OUTPUT -p tcp --dport 80 -j DROP

mkdir -p /etc/iptables

if command -v iptables-legacy-save > /dev/null 2>&1; then
    iptables-legacy-save > /etc/iptables/rules.v4
else
    iptables-save > /etc/iptables/rules.v4
fi

EXTERNAL_TEST=$(curl -s --connect-timeout 5 http://8.8.8.8)
if [ $? -ne 0 ]; then
    echo "✅ Outgoing traffic is blocked as expected."
else
    echo "❌ Outgoing traffic is NOT blocked."
fi

LOCALHOST_TEST=$(ping -c 1 127.0.0.1)
if [ $? -eq 0 ]; then
    echo "✅ Localhost traffic is allowed as expected."
else
    echo "❌ Localhost traffic is NOT allowed."
fi

echo "🌐 DNS queries are allowed on port 53 (both UDP and TCP)."
echo "🔒 All outgoing HTTPS traffic is allowed (only port 443)."
echo "🔥 Blocked all outgoing traffic except for DNS and HTTPS."
