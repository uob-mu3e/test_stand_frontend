post_message "git rev [exec git rev-parse HEAD]"

# make 'cmp' package
exec "util/cmp_pkg.sh"
