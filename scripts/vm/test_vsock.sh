#!/bin/bash
# Run the srv and client as two seperate processes to check that local vsock config works as expected
if [ "$1" = "srv" ]; then
    socat -v -s VSOCK-LISTEN:5005 -
else
    echo "Hello, vsock!" | socat -v -s - VSOCK-CONNECT:2:5005
fi