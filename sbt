#!/usr/bin/env bash


###  ------------------------------- ###
###  Helper methods for BASH scripts ###
###  ------------------------------- ###

realpath () {
(
  TARGET_FILE="$1"
  FIX_CYGPATH="$2"

  cd "$(dirname "$TARGET_FILE")"
  TARGET_FILE=$(basename "$TARGET_FILE")

  COUNT=0
  while [ -L "$TARGET_FILE" -a $COUNT -lt 100 ]
  do
      TARGET_FILE=$(readlink "$TARGET_FILE")
      cd "$(dirname "$TARGET_FILE")"
      TARGET_FILE=$(basename "$TARGET_FILE")
      COUNT=$(($COUNT + 1))
  done

  # make sure we grab the actual windows path, instead of cygwin's path.
  if [[ "x$FIX_CYGPATH" != "x" ]]; then
    echo "$(cygwinpath "$(pwd -P)/$TARGET_FILE")"
  else
    echo "$(pwd -P)/$TARGET_FILE"
  fi
)
}


# Uses uname to detect if we're in the odd cygwin environment.
is_cygwin() {
  local os=$(uname -s)
  case "$os" in
    CYGWIN*) return 0 ;;
    MINGW*) return 0 ;;
    *)  return 1 ;;
  esac
}

# TODO - Use nicer bash-isms here.
CYGWIN_FLAG=$(if is_cygwin; then echo true; else echo false; fi)


# This can fix cygwin style /cygdrive paths so we get the
# windows style paths.
cygwinpath() {
  local file="$1"
  if [[ "$CYGWIN_FLAG" == "true" ]]; then
    echo $(cygpath -w $file)
  else
    echo $file
  fi
}

. "$(dirname "$(realpath "$0")")/sbt-launch-lib.bash"


declare -r noshare_opts="-Dsbt.global.base=project/.sbtboot -Dsbt.boot.directory=project/.boot -Dsbt.ivy.home=project/.ivy"
declare -r sbt_opts_file=".sbtopts"
declare -r etc_sbt_opts_file="${sbt_home}/conf/sbtopts"
declare -r win_sbt_opts_file="${sbt_home}/conf/sbtconfig.txt"

usage() {
 cat <<EOM
Usage: $script_name [options]

  -h | -help         print this message
  -v | -verbose      this runner is chattier
  -d | -debug        set sbt log level to debug
  -no-colors         disable ANSI color codes
  -sbt-create        start sbt even if current directory contains no sbt project
  -sbt-dir   <path>  path to global settings/plugins directory (default: ~/.sbt)
  -sbt-boot  <path>  path to shared boot directory (default: ~/.sbt/boot in 0.11 series)
  -ivy       <path>  path to local Ivy repository (default: ~/.ivy2)
  -mem    <integer>  set memory options (default: $sbt_mem, which is $(get_mem_opts $sbt_mem))
  -no-share          use all local caches; no sharing
  -no-global         uses global caches, but does not use global ~/.sbt directory.
  -jvm-debug <port>  Turn on JVM debugging, open at the given port.
  -batch             Disable interactive mode

  # sbt version (default: from project/build.properties if present, else latest release)
  -sbt-version  <version>   use the specified version of sbt
  -sbt-jar      <path>      use the specified jar as the sbt launcher
  -sbt-rc                   use an RC version of sbt
  -sbt-snapshot             use a snapshot version of sbt

  # java version (default: java from PATH, currently $(java -version 2>&1 | grep version))
  -java-home <path>         alternate JAVA_HOME

  # jvm options and output control
  JAVA_OPTS          environment variable, if unset uses "$java_opts"
  SBT_OPTS           environment variable, if unset uses "$default_sbt_opts"
  .sbtopts           if this file exists in the current directory, it is
                     prepended to the runner args
  /etc/sbt/sbtopts   if this file exists, it is prepended to the runner args
  -Dkey=val          pass -Dkey=val directly to the java runtime
  -J-X               pass option -X directly to the java runtime 
                     (-J is stripped)
  -S-X               add -X to sbt's scalacOptions (-S is stripped)

In the case of duplicated or conflicting options, the order above
shows precedence: JAVA_OPTS lowest, command line options highest.
EOM
}



process_my_args () {
  while [[ $# -gt 0 ]]; do
    case "$1" in
     -no-colors) addJava "-Dsbt.log.noformat=true" && shift ;;
      -no-share) addJava "$noshare_opts" && shift ;;
     -no-global) addJava "-Dsbt.global.base=$(pwd)/project/.sbtboot" && shift ;;
      -sbt-boot) require_arg path "$1" "$2" && addJava "-Dsbt.boot.directory=$2" && shift 2 ;;
       -sbt-dir) require_arg path "$1" "$2" && addJava "-Dsbt.global.base=$2" && shift 2 ;;
     -debug-inc) addJava "-Dxsbt.inc.debug=true" && shift ;;
         -batch) exec </dev/null && shift ;;

    -sbt-create) sbt_create=true && shift ;;

              *) addResidual "$1" && shift ;;
    esac
  done
  
  # Now, ensure sbt version is used.
  [[ "${sbt_version}XXX" != "XXX" ]] && addJava "-Dsbt.version=$sbt_version" 
}

loadConfigFile() {
  cat "$1" | sed '/^\#/d' | while read line; do
    eval echo $line
  done
}

# TODO - Pull in config based on operating system... (MSYS + cygwin should pull in txt file).
# Here we pull in the global settings configuration.
[[ -f "$etc_sbt_opts_file" ]] && set -- $(loadConfigFile "$etc_sbt_opts_file") "$@"
# -- Windows behavior stub'd
# JAVA_OPTS=$(cat "$WDIR/sbtconfig.txt" | sed -e 's/\r//g' -e 's/^#.*$//g' | sed ':a;N;$!ba;s/\n/ /g')


#  Pull in the project-level config file, if it exists.
[[ -f "$sbt_opts_file" ]] && set -- $(loadConfigFile "$sbt_opts_file") "$@"


run "$@"

