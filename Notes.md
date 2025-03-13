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

To make everything easier, we can use the `.vscode/launch.json` to automate and use the debug tool inside VSCode better. Refer to the current setup for different project that we wish to debug. After this, we can freely setup break point, and there are four options:
- `Continue`: Continue the execution of the program until the next breakpoint is encountered.
- `Step Over`: Execute the current line of code and move to the next line. If the current line contains a function call, the function is executed and the result is returned.
- `Step Into`: Execute the current line of code. If the current line contains a function call, the debugger steps into the function and stops at the first line of the function.
- `Step Out`: Continue executing the program until the current function is completed and return to the calling function.

Try to use this command to debug the program. This is very important for our later model development.

Note that in the `.vscode/launch.json`, we can setup multiple testing project, and each project should have a different name, and we should add a config (a dict) in the list, and it will work.

Another useful setup is the `tasks.json`, this is to automate some build process. For example, we can define a task as build in the debug mode, then we can define it in the `"preLaunchTask": "Build Debug GGML"`, in this way the project will build before we do the debug, and we don't need to build it manually.