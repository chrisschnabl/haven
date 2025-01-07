# Build

```make build``

# Run 

In two different terminals: 
```make run-server```
```make run-client```


We use this to transfer data between a Host and an Enclave. We do this to avoid baking data into the secure Nitro image. 

Client runs on Host 
Server runs on Enclave

Host
- stores model
- stores datasets

Enclave
- contains llama.cpp
- contains evaluation code 

Start-Up
- Host sends model to enclave
-> How big is it? How long does it take to run?
- Server receives model 
-> Uses llama crate to run model

Next steps:
-> Test transfer of model locally
-> Test hello world exchange on AWS Nitro
-> Test transfer of model on AWS nitro
-> Test inference of model locally
-> Test inference on AWS nitro

Next steps after that:
-> Do the same with datasets
*/
