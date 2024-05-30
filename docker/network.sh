#!/bin/bash

# Determine which iptables command to use
if command -v iptables-legacy > /dev/null 2>&1; then
    IPTABLES_CMD=iptables-legacy
else
    IPTABLES_CMD=iptables
fi

# List of allowed domains
ALLOWED_DOMAINS=("github.com", "raw.githubusercontent.com")

# Block all outgoing traffic by default
$IPTABLES_CMD -P OUTPUT DROP

# Allow loopback traffic
$IPTABLES_CMD -A OUTPUT -o lo -j ACCEPT

# Allow established and related connections
$IPTABLES_CMD -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

# Allow DNS queries
$IPTABLES_CMD -A OUTPUT -p udp --dport 53 -j ACCEPT
$IPTABLES_CMD -A OUTPUT -p tcp --dport 53 -j ACCEPT

# Clear existing custom rules
$IPTABLES_CMD -F CUSTOM-OUTGOING 2>/dev/null
$IPTABLES_CMD -X CUSTOM-OUTGOING 2>/dev/null

# Create a new chain for custom outgoing rules
$IPTABLES_CMD -N CUSTOM-OUTGOING

# Process each domain in the allowed domains list
for domain in "${ALLOWED_DOMAINS[@]}"; do
    # Resolve the domain to get its IP addresses
    for ip in $(dig +short "$domain"); do
        # Add a rule to allow HTTPS traffic to the resolved IP
        $IPTABLES_CMD -A CUSTOM-OUTGOING -p tcp -d $ip --dport 443 -j ACCEPT
    done
done

# Add a rule to drop all other outgoing traffic by default
$IPTABLES_CMD -A OUTPUT -j CUSTOM-OUTGOING
$IPTABLES_CMD -A OUTPUT -p tcp --dport 443 -j DROP
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

echo "🌐 DNS queries are allowed on port 53 (both UDP and TCP) to ensure proper name resolution."
echo "🔒 HTTPS traffic is allowed only to specified domains: ${ALLOWED_DOMAINS[*]}."
echo "🔥 Blocked all outgoing traffic except DNS and specified HTTPS destinations!"
