ffmpeg -f image2 -r 100 -i figure%%04d.png -vb 20M -vcodec mpeg4 -y movie.mp4
pause