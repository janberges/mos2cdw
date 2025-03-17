#!/bin/bash

tar -czvf fig6.tar.gz --exclude='.git*' --exclude='archive.sh' --transform='s,^,fig6/,' `git ls-files`
