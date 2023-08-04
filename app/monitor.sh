#!/bin/bash

waiting=0
check_server() {
	url=http://localhost:7860
	response_code=$(curl -o /dev/null -s -w "%{http_code}" --connect-timeout 2 $url)
	[[ $response_code -ne 200 ]] && {
		return 0
	}
	return 1
}

while [ 1 ]
do
  # periodically clear the /tmp directory for files created > 30 mins ago so it doesn't fill up
  sudo find /tmp -type f -amin +30 -delete
	check_server
	if [[ $? -ne 1 ]]
	then
		if [[ $waiting -eq 0 ]]
		then
			waiting=1
			echo "Restarting"
			pkill -9 python
			#sudo lsof -t /dev/accel0 | xargs kill -9
			sleep 5
			mv log.txt log_`date +%Y%m%d%H%M%S`
			TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=10000000000 python ./app.py &> log.txt &
		else
			echo "Waiting for restart"
		fi
	else
		if [[ $waiting -eq 1 ]]
		then
			waiting=0
			echo "Restarted"
		fi
	fi
	sleep 10
done
