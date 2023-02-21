#! /bin/bash

i=1;
for file in "$@"
do
    pdfcrop --margins '0 0 0 0' "$file" "$file";
    i=$((i + 1));
done
