
# try to figure out the CUDA version. See if nvcc is in the path, and 
# then call it to get the version. If not, use a default version.
# If $CUDA_VERSION is already set, dont touch it. 

if [ "${CUDA_VERSION}" = "" ];then
    if [[ $(type -P nvcc) ]]; then
        CUDA_VERSION=`nvcc --version | grep release | sed 's/.*release //' | sed 's/\,.*//'`
    else
        CUDA_VERSION="7.5"
    fi
fi
