#!/bin/false 
#
# This script sets the environmental variables needed to run ivoxoar in place
# (e.g. without installing it to /usr/lib/python2.7/site-packages and
# /usr/local/bin). It changes your current session only.
#
# To use script, run from your shell:
#   source run-in-place.sh
#
# The following two commands might look right but they WILL NOT WORK:
#   ./run-in-place.sh
#   bash run-inplace.sh 
#

IVOXPATH="`pwd`/bin"
IVOXLIB="`pwd`/lib"

if [[ ":$PATH:" != *":$IVOXPATH:"* ]]; then
    export PATH="$IVOXPATH${PATH:+":$PATH"}"
fi

if [[ ":$PYTHONPATH:" != *":$IVOXLIB:"* ]]; then
    export PYTHONPATH="$IVOXLIB${PYTHONPATH:+":$PYTHONPATH"}"
fi
