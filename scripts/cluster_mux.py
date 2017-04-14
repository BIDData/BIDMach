#!/usr/bin/env python
import subprocess
import time

SPARK_SLAVE_PATH = '/opt/spark/conf/slaves'


def tmux_cmd(cmd, fail_ok=False):
    if type(cmd) is str:
        cmd = cmd.split(' ')
    try:
        return subprocess.check_output(['tmux'] + cmd).strip().split('\n')
    except subprocess.CalledProcessError as e:
        if not fail_ok:
            raise e


def send_cmd(pid, cmd):
    tmux_cmd(['send-keys', '-t', pid, cmd+'\n'])


def main():
    tmux_cmd('kill-window -t tail-workers', fail_ok=True)
    tmux_cmd('new-window -d -n tail-workers')

    pane_id = tmux_cmd('list-panes -t tail-workers -F #D')[0]
    tmux_cmd('split-window -d -h -t {}'.format(pane_id))
    pane_ids = tmux_cmd('list-panes -t tail-workers -F #D')
    for pid in pane_ids:
        tmux_cmd('split-window -d -v -t {}'.format(pid))
    pane_ids = tmux_cmd('list-panes -t tail-workers -F #D')

    with open(SPARK_SLAVE_PATH, 'r') as f:
        slave_addrs = list(f.readlines())

    for pid, saddr in zip(pane_ids, slave_addrs):
        send_cmd(pid, 'su2')
        time.sleep(0.05)
        send_cmd(pid, 'ssh {}'.format(saddr))
        time.sleep(0.1)
        send_cmd(pid, 'tail -f /tmp/bidmach_worker.log')


if __name__ == '__main__':
    main()
