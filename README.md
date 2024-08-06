# skeleton2mjcf

This Python script is made to convert a skeleton and mesh to a MuJoCo bodies hierarchy, alongside with a skin.

It gets its data from a Blender scene, because I have not found a regular 3D model file type which contains the necessary data.

The script is not complete and I will probably never finish it (see the TODO items below) but the structure is there and it can already be used. Adding those features should not be really complicated for future contributors.

## Usage

Download the repository, install the dependencies (in a venv!) using
```sh
$ pip install -r requirements.txt
```

Launch the script with
```sh
$ python3 skeleton2mjcf.py [OPTIONS] INPUT_FILE OUTPUT_DIR
```

You can get a list of all available options with
```sh
$ python3 skeleton2mjcf.py --help
```

## What does it do?

1. Loads a Blender scene from the input file
1. For each object (selected in the scene or specified with the `--objects` parameter):
    1. Converts the hierarchy of bones to a hierarchy of MuJoCo bodies
        - bodies are positioned correctly
        - they do not have proper orientations (TODO)
        - bodies cannot move, there are no joint between them (TODO)
    1. Creates a MuJoCo skin according to the object mesh
        - skins are properly linked to the bodies and will move with them
        - skins do not have textures (TODO)
1. Writes a .xml file with the MuJoCo model in the output directory and saves the created skins in files

