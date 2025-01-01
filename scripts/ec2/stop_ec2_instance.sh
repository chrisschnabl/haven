#!/bin/zsh

# is instance ID provided?
if [ -z "$1" ]; then
  echo "Error: No instance ID provided."
  echo "Usage: ./stop_ec2_instance.sh <INSTANCE_ID>"
  exit 1
fi

INSTANCE_ID="$1"

# Stop the instance
echo "Stopping instance: $INSTANCE_ID..."
aws ec2 stop-instances --instance-ids "$INSTANCE_ID"

# Check
if [ $? -eq 0 ]; then
  echo "Successfully sent stop command to instance: $INSTANCE_ID."
else
  echo "Failed to stop instance: $INSTANCE_ID."
  exit 1
fi

