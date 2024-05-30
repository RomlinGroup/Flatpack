#!/bin/bash

# Determine which iptables command to use
if command -v iptables-legacy > /dev/null 2>&1; then
    IPTABLES_CMD=iptables-legacy
else
    IPTABLES_CMD=iptables
fi

# Block all outgoing traffic by default
$IPTABLES_CMD -P OUTPUT DROP

# Allow loopback traffic
$IPTABLES_CMD -A OUTPUT -o lo -j ACCEPT

# Allow established and related connections
$IPTABLES_CMD -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

# Allow DNS queries
$IPTABLES_CMD -A OUTPUT -p udp --dport 53 -j ACCEPT
$IPTABLES_CMD -A OUTPUT -p tcp --dport 53 -j ACCEPT

# Allow all outgoing HTTPS traffic
$IPTABLES_CMD -A OUTPUT -p tcp --dport 443 -j ACCEPT

# Add a rule to drop all other outgoing traffic by default
$IPTABLES_CMD -A OUTPUT -p tcp --dport 80 -j DROP

# Save the iptables rules
mkdir -p /etc/iptables

if command -v iptables-legacy-save > /dev/null 2>&1; then
    iptables-legacy-save > /etc/iptables/rules.v4
else
    iptables-save > /etc/iptables/rules.v4
fi

# Test outgoing traffic block
EXTERNAL_TEST=$(curl -s --connect-timeout 5 http://8.8.8.8)
if [ $? -ne 0 ]; then
    echo "✅ Outgoing traffic is blocked as expected."
else
    echo "❌ Outgoing traffic is NOT blocked."
fi

# Test localhost traffic
LOCALHOST_TEST=$(ping -c 1 127.0.0.1)
if [ $? -eq 0 ]; then
    echo "✅ Localhost traffic is allowed as expected."
else
    echo "❌ Localhost traffic is NOT allowed."
fi

echo "🌐 DNS queries are allowed on port 53 (both UDP and TCP)."
echo "🔒 All outgoing HTTPS traffic is allowed (only port 443)."
echo "🔥 Blocked outgoing traffic except for DNS and HTTPS."
