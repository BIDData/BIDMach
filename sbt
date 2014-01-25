#!/bin/bash

java $SBT_OPTS -Dfile.encoding=UTF-8 -Xss4M -Xmx512M -XX:MaxPermSize=512M -XX:NewSize=128M -XX:NewRatio=3 -jar lib/sbt-launch.jar "$@"
