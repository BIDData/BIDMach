export JAVA_OPTS="-Xmx4G -Xms128M -Dfile.encoding=UTF-8"
export KMP_DUPLICATE_LIB_OK=TRUE
printenv JAVA_OPTS
printenv KMP_DUPLICATE_LIB_OK
./sbt console
