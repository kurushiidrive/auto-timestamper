# UNI Auto-Timestamper

Work-in-progress Auto-Timestamper for Under Night In-Birth, written in Python using OpenCV.

`kat.py` is the main source file. Run it with a path to an on-disk video file as the command line argument. For example:

`python kat.py UNI-video.mp4`

The program will fast-forward through the video, noting down the timestamp and the character matchup when it detects that a new match has started.

After reaching the end of the video or upon the user pressing 'q' to quit, the program will write to console/a text file the running timestamps for the matches it detected before program termination.

Structural similarity is used to determine which characters are fighting at a certain timestamp. The portraits are compared to "seed" images (found in `/uni_char/`), and the image with the best structural similarity (a value on [-1.0, 1.0], 1.0 being an exact match) is used to determine which character it is.

Later, I plan on using a siamese neural network instead of structural similarity, which is what all of the other `*.py` files are for.

[Video demonstration of early structural similarity-based program](https://youtu.be/FnLX1YT-hBQ)