# UNI Auto-Timestamper

Work-in-progress Auto-Timestamper for Under Night In-Birth VODs, written in Python using OpenCV.

## GUI app download

You can download KAT GUI, which is the graphical-interface-based auto-timestamper, from the [releases page](https://github.com/kurushiidrive/auto-timestamper/releases). It is recommended to get the latest version available. Instructions for it are in INSTRUCTIONS.txt in the zip. 

## For advanced users

`kat.py` is the main source file. If you wish to run it (or any other of the Python source files), make sure you have Python 3.6+ on your system. You can setup a virtualenv for the project through the following commands:

* Clone the repo to a directory of your liking
* `cd` into that directory
* Run `python -m venv env` to create a virtualenv
* Run `.\env\Scripts\activate.bat` to "enter" the virtualenv (if you're on Linux, do `./env/bin/activate` instead, but first make sure `activate` has execute permissions; can do this with `chmod u+x ./env/bin/activate`)
* Run `pip install -r requirements.txt` (installs required modules into virtualenv)

After completing the above steps, run `python kat.py` with either of the following as the command line argument:

* the path to an on-disk video file
* the URL of a YouTube video

For example:

* `python kat.py UNI-video.mp4`
* `python kat.py https://youtu.be/DRalAyuFMNg`

If you prefer a **graphical interface**, `kat_gui.py` is the GUI version. To run it, simply do `python kat_gui.py`.

## Brief explanation on how it works

The program will fast-forward through the video, noting down the timestamp and the character matchup when it detects that a new match has started.

After reaching the end of the video or upon the user pressing 'q' to quit (or clicking the 'Cancel' button for the GUI app), the program will write to console/a text file the running timestamps for the matches it detected before termination.

Structural similarity is used to determine which characters are fighting at a certain timestamp. The character portraits are compared to "seed" images (found in `/uni_char/seed/` and `/uni_char/training/`), and the image with the best structural similarity (a value on [-1.0, 1.0], 1.0 being an exact match) is used to determine which character it is.

Later, I plan on using a siamese neural network instead of structural similarity, which is what all of the other `*.py` files are for.

However, note that currently, the **structural similarity** method (`kat.py`, `kat_gui.py`) is more accurate than the **siamese neural network** (`kat_nn.py`).

More training data are needed to improve the performance of the siamese NN, but if you would like to try it yourself, do the following:

* (optional) train the NN with `python train_siamese_network.py` ~~(can't be done until I add the training images dir to the repo)~~
* run the NN-based auto-timestamper with `python kat_nn.py [video]`

## Limitations, Demonstrations, and Samples

At the moment, there is support for only vanilla UNI, UNIEL, and UNICLR videos. I will work on compatibility for UNIST later on.

Additionally, the auto-timestamper currently works on only videos where the game takes up the entire screen (resolution of the video doesn't matter). It will likely produce inaccurate or even no results if the game is scaled within the video. This is being worked on, so please be patient.

Finally, at the moment I am taking a naïve approach when it comes to determining the start of a new match. All that's looked for is "Recurring VOID Effect...", so if that does not appear before the start of a new match, then the auto-timestamper won't detect that match. I am thinking of a more robust way of tackling this, but if you have suggestions let me know.

[Video demonstration of early structural-similarity-based program](https://youtu.be/FnLX1YT-hBQ)

[Second video demonstration *w/ commentary* of structural-similarity-based program](https://youtu.be/OAD95oxNWZ4)

[KAT GUI v1.00 demonstration](https://youtu.be/ZIcjldFUlek)

Sample videos to test the auto-timestamper on:
* [旧UNI 2 16蒲田対戦会 part2 sm20109217](https://youtu.be/DRalAyuFMNg)
* [『アンダーナイトインヴァース エクセレイト』シングル 西セガ 4月23日](https://youtu.be/p3oiT4R-f6U)
* [『アンダーナイトインヴァース エクセレイト』3on3 西セガ 4月19日 part.3](https://youtu.be/6l41BlbkmLQ)
* [11月15日 アテナ日本橋 UNIランバト 予選リーグ1 sm19406595](https://youtu.be/qRaqwvV7wGU)