"""beat2body: music- and genre-conditioned dance skeleton generation.

Run `python -m beat2body --help` for the command line. Code is organized as:

    beat2body/paths.py      data folder layout (2d/ by default)
    beat2body/aist.py       AIST Dance DB naming: genres, dancers, songs, video lists
    beat2body/skeleton.py   COCO-17 joints and the mappings from each pose tool
    beat2body/video.py      fixed-fps frame reading and skeleton drawing
    beat2body/pose/         pose estimation, one module per tool (README 4.3)
    beat2body/prep/         frames, clean, audio, clips + split, Hugging Face sync (4.2, 4.4-4.7)
    beat2body/render.py     stick-figure videos of real or generated keypoints
"""
__version__ = "0.2.0"
