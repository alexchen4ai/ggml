# Overview
In this file, we show the procedure how to implement the Jina V3 using the llama cpp. And we use a systematic way to achieve this goal. That is, we wish to implement the work similar to the `qwen.cpp` way, by divide and conquer.

## Understand the model arc
We don't even need to read the paper (better to have), but we must know the essential model architecture and inference process. The model arc is included in the `jina_v3.mermaid` file. And it can be visualized in website like: https://mermaid.live/edit. 


## From leaves to root
We consider to implement the individual components first, and finally combine them together. For our next milestone, we wish to implement the layer 0 first.