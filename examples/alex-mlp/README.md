# GGML for MLP

Create model and convert it to GGML format.

```
python model.py
python convert.py
```

Build and run the example. Don't forget to include the current folder in the `CMakeLists.txt` of the parent directory.

```
cd ~ # go to home directory
rm -r -f build # remove the build directory
cmake -B build # set the build directory
cmake --build build --config Release -j16 # build project
./build/bin/alex-mlp ./path-to-gguf/examples/alex-mlp/model/mlp.gguf # Run the example
```

This project can be done only dependent on the GGML library, or llamacpp, and we need to edit the CMakeLists.txt file to change the dependency.
