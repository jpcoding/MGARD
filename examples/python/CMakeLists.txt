cmake_minimum_required(VERSION 3.9)
project(mgardpy)
find_package(MGARD REQUIRED HINTS "../../install")

add_executable(test_mgard cpp/test_mgard.cpp)
target_link_libraries(test_mgard mgard::mgard)
target_include_directories(
    test_mgard
    PUBLIC

)

add_library(
    mgardpy
    SHARED
    cpp/test_mgard.cpp
)

target_link_libraries(
    mgardpy
    PUBLIC
    mgard::mgard
)

install(TARGETS test_mgard mgardpy EXPORT mgardpy
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mgardpy)