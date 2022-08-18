#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# to html
jupyter nbconvert --to markdown notebooks/0-finding-periodicity.ipynb
cp notebooks/0-finding-periodicity_files/* figures/

# export md and figures
jupyter nbconvert --to html notebooks/0-finding-periodicity.ipynb
cp notebooks/0-finding-periodicity.html ~/code/repos/miguelarbesu.xyz/content/post/cicada-songs-finding-periodicity/
