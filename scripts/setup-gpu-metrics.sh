#!/bin/bash
# Setup sudoers for powermetrics GPU monitoring

set -e

USERNAME=$(whoami)
SUDOERS_FILE="/etc/sudoers.d/qwen3-powermetrics"

echo "Setting up GPU metrics permissions..."
echo ""
echo "This script will configure sudo to allow powermetrics without password."
echo "The rule is restricted to safe intervals (1000-60000ms) to prevent DoS."
echo ""

# Create sudoers file with restricted interval pattern
echo "$USERNAME ALL=(ALL) NOPASSWD: /usr/bin/powermetrics --samplers gpu_power -i [1-6][0-9][0-9][0-9]" | sudo tee $SUDOERS_FILE
sudo chmod 440 $SUDOERS_FILE

echo ""
echo "✓ GPU metrics configured successfully."
echo ""
echo "You can now run the server and GPU metrics will be collected automatically."
echo "To verify, run: sudo powermetrics --samplers gpu_power -i 2000 -n 1"
