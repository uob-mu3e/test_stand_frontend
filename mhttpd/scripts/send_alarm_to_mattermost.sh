#!/bin/bash
alarm_message=$1
generate_post_data()
{
  cat <<EOF
  {
  "text": "@all Midas alarm triggered with message = $alarm_message", 
  "icon_emoji":":rotating_light:", 
  "username":"Midas"}
EOF
}
curl -i -X POST -H 'Content-Type: application/jsoni' --data "$(generate_post_data)" https://mattermost.gitlab.rlp.net/hooks/8fm5nqcuzjf9xnjoaao7m6im9c
#echo $alarm_message > /home/mu3e/online/mhttpd/scripts/out.txt
