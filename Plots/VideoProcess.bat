ffmpeg -r 100 -f image2 -i figure%%04d.png -vb 20M -vcodec libx264 -crf 15 -pix_fmt yuv420p output.mp4
pause