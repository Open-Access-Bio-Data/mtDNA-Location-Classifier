#!/bin/bash

# Install EDirect tools and set up PATH
yes | sh -c "$(wget -q https://ftp.ncbi.nlm.nih.gov/entrez/entrezdirect/install-edirect.sh -O -)"
echo 'export PATH=$HOME/edirect:$PATH' >> ~/.bashrc
export PATH=$HOME/edirect:$PATH


