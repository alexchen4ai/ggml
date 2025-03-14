# main-ctx
set(TARGET alex-mlp)
add_executable(${TARGET} main-ctx.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# main-backend
# set(TARGET_BACKEND alex-mlp-backend)
# add_executable(${TARGET_BACKEND} main-backend.cpp)
# install(TARGETS ${TARGET_BACKEND} RUNTIME)
# target_link_libraries(${TARGET_BACKEND} PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})
# target_compile_features(${TARGET_BACKEND} PRIVATE cxx_std_17)

# If we have multiple cpp code, we can create multiple executable
# For GPU offloading
if (GGML_CUDA)
    add_compile_definitions(GGML_USE_CUDA)
endif()