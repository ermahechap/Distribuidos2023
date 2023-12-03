#! /bin/bash
apt update
apt -y install build-essential libmpich-dev libopenmpi-dev libsndfile-dev fftw-dev libfftw3-dev nfs-kernel-server nfs-common ffmpeg openssh-server git
curl -L https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp -o /usr/local/bin/yt-dlp
chmod a+rx /usr/local/bin/yt-dlp