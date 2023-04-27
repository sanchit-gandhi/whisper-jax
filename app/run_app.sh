#!/bin/bash

# Sets up the environment, suitable to be run from a cron job, i.e.:
# @reboot /home/sanchitgandhi/whisper-jax/run_app.sh
cd ~/whisper-jax/app
source ~/hf/bin/activate
chmod +x monitor.sh
./monitor.sh &
