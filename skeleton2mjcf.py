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
from dm_control.mjcf import skin
from dm_control.mjcf import Element as mjElement
from dm_control.mjcf.attribute import SkinAsset as mjSkinAsset
from dm_control.mjcf.skin import Skin as mjSkin, Bone as mjBone

import bpy
import bmesh
from bpy.types import Object as bObject, Armature as bArmature, Bone as bBone, Mesh as bMesh
from mathutils import Vector, Quaternion

# All our assets have unique names so we do not need the hash in the filename
class UnhashedAsset(mjcf.Asset):
    def __init__(self, contents, name, extension):
        super().__init__(contents, extension)
        self.name = name

    def get_vfs_filename(self):
        return self.name + self.extension

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
    orient_bodies: bool = False
    """Orient the bodies according to the direction of the bones."""
    no_hierarchy: bool = False
    """For debug purposes"""
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
        if self.options.objects:
            objects = [bpy.data.objects[object_name] for object_name in self.options.objects]
        else:
            if bpy.context.selected_objects:
                objects = bpy.context.selected_objects
            else:
                objects = bpy.data.objects
        
        # We turn this list of references to a list of strings, as Blender recommends.
        object_names = []
        for object in objects:
            if not isinstance(object.data, bArmature):
                raise RuntimeError(object.name, "is not an Armature")
            object_names.append(object.name)

        return object_names
    
    def _add_body(self, name):
        body: mjElement = self.model.worldbody.add("body", name=name)
        object: bObject = bpy.data.objects[name]
        armature: bArmature = object.data

        armature.pose_position = "POSE" if self.options.use_pose_position else "REST"

        def add_body_part(parent: mjElement, parent_head: Vector | None, parent_orientation: Quaternion | None, bone: bBone):
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
            if parent_head is not None and not options.no_hierarchy:
                # BEWARE: the -= operator does not seem to work properly on bpy Vectors
                head  = head - parent_head
                tail = tail - parent_head

            orientation = None
            if self.options.orient_bodies:
                v1 = Vector((0, 0, 1))
                axis = tail - head
                xyz = v1.cross(axis)
                w = axis.length + v1.dot(axis)
                orientation = Quaternion([w, *xyz])
                orientation.normalize()
                if parent_orientation is not None:
                    orientation = parent_orientation.rotation_difference(orientation)
                
                # TODO do position work

            body: mjElement = parent.add("body", name=bone.name, pos=[*head])

            if self.options.orient_bodies:
                body.set_attributes(quat=[*orientation])

            geom = body.add("geom", name=f"{bone.name}_collision",
                            type="capsule", rgba=".5 .5 .5 .5",
                            fromto = [0, 0, 0, *(tail - head)],
                            size = [bone.tail_radius])
            
            for sub_bone in bone.children:
                add_body_part(parent if self.options.no_hierarchy else body, head_local, orientation, sub_bone)

        root_bone = armature.bones[0]
        add_body_part(body, None, None, root_bone)

    def _add_skin(self, name: str):
        skin_obj = self._create_skin(name)

        self.model.deformable.add("skin", name=name, file=UnhashedAsset(skin.serialize(skin_obj), name, ".skn"))

    def _get_mesh(self, rig: bObject):
        for obj in bpy.data.objects:
            if obj.type == "MESH" and rig in [m.object for m in obj.modifiers if m.type == "ARMATURE"]:
                return obj
        raise ValueError("No mesh found for rig", rig.name)

    def _create_skin(self, name: str) -> mjSkin:
        rigObject: bObject = bpy.data.objects[name]
        armature: bArmature = rigObject.data
        meshObject = self._get_mesh(rigObject)
        mesh: bMesh = meshObject.data

        meshObject.select_set(state=True)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        meshObject.select_set(state=False)

        mesh_b = bmesh.new()
        mesh_b.from_mesh(mesh, face_normals=False, vertex_normals=False)

        vertices = np.array([coordinate for v in mesh_b.verts for coordinate in v.co]).reshape(-1, 3)
        texcoords = np.array([]).reshape(-1, 2)

        # faces_raw = mesh_b.faces
        faces_raw = bmesh.ops.triangulate(mesh_b, faces=mesh_b.faces)["faces"]
        faces = np.array([vert.index for face in faces_raw for vert in face.verts]).reshape(-1, 3)

        bones = []
        for bone in armature.bones:
            body: mjElement = self.model.find("body", bone.name)
            # bindpos = np.array(body.pos)
            # bindpos = np.array(bone.matrix_local)
            # bindquat = np.array(bone.matrix.to_quaternion())
            translation, rotation, scale = bone.matrix_local.decompose()
            bindpos = np.array(translation)
            # bindquat = np.array(rotation)
            bindquat = np.array([1, 0, 0, 0])

            vertex_group_id = meshObject.vertex_groups[bone.name].index
            vertex_ids = []
            vertex_weights = []
            for v in mesh.vertices:
                for g in v.groups:
                    if vertex_group_id == g.group:
                        vertex_ids.append(v.index)
                        vertex_weights.append(g.weight)
                        break
            
            vertex_ids = np.array(vertex_ids, dtype=np.int32)
            vertex_weights = np.array(vertex_weights, dtype=np.float32)

            bones.append(mjBone(lambda b=body: b, bindpos, bindquat, vertex_ids, vertex_weights))

        return mjSkin(vertices, texcoords, faces, bones)

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
            self._add_skin(object_name)

        if should_remove:
            shutil.rmtree(self.options.output)

        mjcf.export_with_assets(self.model, self.options.output, "Hand.xml", precision=3)

if __name__ == "__main__":
    options = tyro.cli(Options, description=__doc__)
    SkeletonCreator(options).main()
