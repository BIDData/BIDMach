#!/bin/bash
CMD_STR="cluster-mux -p su2 -c 'tail -f /tmp/bidmach_worker.log'"
if [[ $(whoami) != "aleks" ]]; then
  sudo su aleks -c "$CMD_CTR"
else
  eval $CMD_STR
fi
