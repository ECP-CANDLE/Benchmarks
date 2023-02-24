
# GET NODE TXT AWK
# See get-node-txt.sh

$2 == node && $1 == "node:" {
  print $0;
  while (1) {
    if (getline <= 0) {
      # EOF:
      exit;
    }

    if ($1 == "node:") {
      # The next node is starting:
      exit;
    }
    # Good node data:
    print $0;
  }
}
