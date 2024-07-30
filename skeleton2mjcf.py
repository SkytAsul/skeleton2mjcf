"""
A util similar to obj2mjcf made to convert a 3D skeleton to a MuJoCo MJCF file.

Unlike obj2mjcf, this util recreates a hierarchy of bodies and constructs the convex
hulls on each subpart of the body, thus making the constructed XML file almost ready-to-use.
"""

import dataclasses
from pathlib import Path
import logging
import shutil
from typing import Iterable
import tyro

import numpy as np
import numpy.typing as npt

from dm_control import mjcf
from dm_control.mjcf import Element as mjElement

import bpy
from bpy.types import Object as bObject, Armature as bArmature, Bone as bBone
from mathutils import Vector

@dataclasses.dataclass
class Options:
    file: tyro.conf.Positional[Path]
    """Path to the Blender scne."""
    output: tyro.conf.Positional[Path]
    """Path to the output directory."""
    objects: list[str] = dataclasses.field(default_factory=lambda: [])
    """Name of the objects to convert.
    If none specified, the selection in the Blender scene will be used.
    If no object is selected in Blender, then all objects will be converted."""
    use_pose_position: bool = False
    """Use the "pose" position of the armature, instead of the "rest" one."""
    overwrite: bool = False
    """Should the output directory be automatically deleted (if present).
    If not enabled, the user will be prompted.
    """
    reuse_old_assets: bool = True
    """Should the existing convex decomposition .obj files from another
    execution be reused.

    Only effective if the user answered "Yes" to the overwrite prompt
    or if the overwrite option is enabled.
    """
    verbose: bool = True
    """Should informations be printed to the standard output."""

class SkeletonCreator:
    def __init__(self, options: Options):
        self.options = options

    def _add_defaults(self, model: mjcf.RootElement):
        visual = model.default.add("default", dclass="visual")
        visual.geom.set_attributes(type="mesh", group=2, contype=0, conaffinity=0)

        collision = model.default.add("default", dclass="collision")
        collision.geom.set_attributes(type="mesh", group=3)
    
    def _load_scene(self):
        logging.info("Loading blender file %s...", self.options.file)
        bpy.ops.wm.open_mainfile(filepath=str(self.options.file))

    def _get_object_names(self) -> list[bObject]:
        objects = []
        # we do not keep a list of references to Object instances
        # as blender indicates against this.
        if self.options.objects:
            for object_name in self.options.objects:
                if object_name in bpy.data.objects:
                    objects.append(object_name)
                else:
                    raise ValueError("No object named", object_name)
        else:
            if bpy.context.selected_objects:
                for object in bpy.context.selected_objects:
                    objects.append(object.name)
            else:
                for object in bpy.data.objects:
                    objects.append(object.name)
        return objects
    
    def _add_body(self, name):
        body: mjElement = self.model.worldbody.add("body", name=name)
        object: bObject = bpy.data.objects[name]
        
        if not isinstance(object.data, bArmature):
            raise RuntimeError(name, "is not an Armature")
        armature = object.data

        armature.pose_position = "POSE" if self.options.use_pose_position else "REST"
        
        def p(array):
            return [f"{n:.3}" for n in array]

        def add_body_part(parent: mjElement, parent_head: Vector | None, bone: bBone):
            logging.info("Processing bone %s...", bone.name)

            # we use _local positions, because other ones are too complicated to work with
            if bone.use_connect:
                head_local = bone.parent.tail_local
            else:
                head_local = bone.head_local
            head = head_local
            tail = bone.tail_local

            # We put the MuJoCo body at the head. To keep local positions,
            # the tail location of the parent must be substracted.
            if parent_head is not None:
                head  = head - parent_head
                tail = tail - parent_head

            body: mjElement = parent.add("body", name=bone.name, pos=[*head])

            geom = body.add("geom", name=f"{bone.name}_collision",
                            type="capsule", rgba=".5 .5 .5 .5",
                            fromto = [0, 0, 0, *(tail - head)],
                            size = [bone.tail_radius])
            
            for sub_bone in bone.children:
                add_body_part(body, head_local, sub_bone)

        root_bone = armature.bones[0]
        add_body_part(body, None, root_bone)

    def main(self):
        if self.options.verbose:
            logging.getLogger().setLevel(logging.INFO)

        should_remove = False
        if self.options.output.exists():
            if not self.options.overwrite and input("Output already exists. Do you want to remove ? [Y/N] ").lower() != "y":
                return
            
            if self.options.reuse_old_assets:
                should_remove = True
            else:
                shutil.rmtree(self.options.output)
        else:
            self.options.reuse_old_assets = False

        self.model = mjcf.RootElement()
        self._add_defaults(self.model)

        self._load_scene()

        for object_name in self._get_object_names():
            self._add_body(object_name)

        if should_remove:
            shutil.rmtree(self.options.output)

        mjcf.export_with_assets(self.model, self.options.output, "Hand.xml", precision=3)

if __name__ == "__main__":
    options = tyro.cli(Options, description=__doc__)
    SkeletonCreator(options).main()
