# Athena++, Cholla Visualization

Dependencies:
- ffmpeg (for video encoding)
- h5py
- pylab

### Usage

Run `python animate.py --help` for options.

**Example**

```
$ python script.py -c cholla_data/ -l Cholla -a athena_data/ -l Athena -o example_movie.mp4
```
Description:
- `-c --cholla` Flage used to specify the folder containg the `*.h5` files from cholla.
- `-l --label` Flag used to specify the label shown in the legend of the plot.
- `-a --athena` Flag used for athena's data output.
- `-o --out` Flag used to specify the name of the output movie. If no output file is specified, then an animation in a gui is shown.

*\*Note* - At least one of `-c` or `-a` must be specified, but multiple of each can be used. Also, the `-l` flag is optional; if it is not specified, then the name of the folder will be used in the legend.

### Warnings
If you encounter errors relating to multiprocessing, then try changing the threading method near the end of the script.