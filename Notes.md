# Overview
In this notes, I will include detailed introduction of my setup environment. Also, I will include some tricks during my development.

## Setup env
When we develop a specific project using GGML, we should set the workspace the same as the ggml folder so that we can better use the `.vscode` and copilot for the work.

## Debug setp
To enable the debug setup, we should build the project using the following:
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --config Debug -j8
```

To make everything easier, we can use the `.vscode/launch.json` to automate and use the debug tool inside VSCode better. Refer to the current setup for different project that we wish to debug.