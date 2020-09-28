post_message "git rev [exec git rev-parse HEAD]"

# make 'cmp' package
exec "util/cmp_pkg.sh"

# do not track arriaV transceivers in git
exec "a5/cpIPs.sh"

source "a5/genIPs.tcl"
