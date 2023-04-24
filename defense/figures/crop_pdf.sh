#! /bin/bash

for file in "$@"
do
    pdfcrop --margins '0 0 0 0' "$file" "$file";
done
