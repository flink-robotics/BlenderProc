"""Allows rendering the content of the scene in the Flink dataset format."""

import os
import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from itertools import groupby
import uuid
import time

import numpy as np
import cv2
import bpy
from mathutils import Matrix, Vector
from bpy_extras.object_utils import world_to_camera_view

from blenderproc.python.types.MeshObjectUtility import MeshObject
from blenderproc.python.writer.WriterUtility import _WriterUtility


class HideMeshWithProperty:
    """Context manager that temporarily hides all objects without <property_name> attribute.

    Args:
        scene_objects (List[MeshObject]): The list of objects to hide.
        property_name (str): The name of the property to check.
        inverse (bool, optional): If True, hide objects that have the property. Defaults to False.
        DEBUG (bool, optional): If True, print debug information. Defaults to False.

    Example:
        ```python
        with HideMeshWithProperty():
            # Only objects with <property_name> will be visible here
            # Perform operations on visible objects
        # All objects are restored to their original visibility state here
        ```
    """

    def __init__(self, scene_objects: List[MeshObject], property_name: str, inverse: bool = False, DEBUG: bool = False):
        """Initialize the context manager."""
        self.scene_objects = scene_objects
        self.hidden_objects: List[MeshObject] = []
        self.property_name = property_name
        self.inverse = inverse
        self.DEBUG = DEBUG

    def __enter__(self) -> 'HideMeshWithProperty':
        """Hide all objects without <property_name> attribute.

        Returns:
            HideMeshWithProperty: The context manager instance.
        """
        # Store and hide objects that don't have the <property_name> attribute
        for obj in self.scene_objects:
            if obj.has_cp(self.property_name) != self.inverse:
                self.hidden_objects.append(obj)
                obj.blender_obj.hide_render = True
                obj.blender_obj.hide_viewport = True
                if self.DEBUG:
                    print(
                        f"Hiding object {obj.blender_obj.name}, property {self.property_name}: {obj.has_cp(self.property_name)}, inverse: {self.inverse}")

        return self

    def __exit__(self, exc_type: Optional[type], exc_val: Optional[Exception], exc_tb: Optional[object]) -> None:
        """Restore the original visibility state of all hidden objects.

        Args:
            exc_type: The exception type if an exception was raised in the context.
            exc_val: The exception value if an exception was raised in the context.
            exc_tb: The traceback if an exception was raised in the context.
        """
        # Restore visibility of hidden objects
        for obj in self.hidden_objects:
            obj.blender_obj.hide_render = False
            obj.blender_obj.hide_viewport = False
            if self.DEBUG:
                print(f"Restoring visibility of object {obj.blender_obj.name}")


class AuxiliaryCube:
    """Context manager that adds an auxiliary cube to the scene."""

    def __init__(self, location: Vector, scale: Vector):
        self.location = location
        self.scale = scale
        self.cube = None

    def __enter__(self) -> bpy.types.Object:
        """Add an auxiliary cube to the scene."""
        bpy.ops.mesh.primitive_cube_add(
            location=self.location, scale=self.scale)
        self.cube = bpy.context.object
        return self.cube

    def __exit__(self, exc_type: Optional[type], exc_val: Optional[Exception], exc_tb: Optional[object]) -> None:
        """Remove the auxiliary cube from the scene."""
        bpy.data.objects.remove(self.cube, do_unlink=True)


def is_ray_hit_vertex(vtx: Vector, ray_origin: Vector, ray_direction: Vector, helper_cube_scale: float = 0.0001, DEBUG: bool = False) -> bool:
    """Checks if a vertex is occluded by objects in the scene w.r.t. a given ray.
    https://github.com/DLR-RM/BlenderProc/issues/990#issuecomment-1764989373

    Args:
        vtx (Vector): the world space x, y and z coordinates of the vertex.
        ray_origin (Vector): origin point of the ray
        ray_direction (Vector): direction vector of the ray
        helper_cube_scale (float, optional): Scale of helper geometry. Defaults to 0.0001.
        DEBUG (bool, optional): Enable debug visualization. Defaults to False.

    Returns:
        boolean: visibility
    """
    vtx = Vector(vtx)
    ray_origin = Vector(ray_origin)
    ray_direction = Vector(ray_direction).normalized()

    # add small cube around coord to make sure the ray will intersect
    # as the ray_cast is not always accurate
    # cf https://blender.stackexchange.com/a/87755
    bpy.ops.mesh.primitive_cube_add(location=vtx, scale=(
        helper_cube_scale, helper_cube_scale, helper_cube_scale))
    cube = bpy.context.object

    hit, location, _, _, _, _ = bpy.context.scene.ray_cast(
        bpy.context.view_layer.depsgraph,
        origin=ray_origin + ray_direction * 0.0001,  # avoid self intersection
        direction=ray_direction,
    )

    if DEBUG:
        print(f"hit location: {location}")
        bpy.ops.mesh.primitive_ico_sphere_add(
            location=location, scale=(
                helper_cube_scale, helper_cube_scale, helper_cube_scale)
        )

    # remove the auxiliary cube
    if not DEBUG:
        bpy.data.objects.remove(cube, do_unlink=True)

    if not hit:
        raise ValueError(
            "No hit found, this should not happen as the ray should always hit the vertex itself.")
    # if the hit is the vertex itself, it is not occluded
    if (location - vtx).length < helper_cube_scale * 2:
        return False
    return True


def is_object_occluded_for_scene_camera(camera: bpy.types.Object, obj: bpy.types.Object, DEBUG: bool = False) -> bool:
    """Checks if all vertices of an object are occluded by objects in the scene w.r.t. the camera.

    Args:
        obj (bpy.types.Object): the object.

    Returns:
        boolean: visibility
    """
    for vertex in obj.data.vertices:
        coords = obj.matrix_world @ vertex.co
        if not is_ray_hit_vertex(coords, camera.location, coords - camera.location, DEBUG=DEBUG):
            return False
    return True


def is_object_all_visible(obj: bpy.types.Object, DEBUG: bool = False) -> bool:
    """Checks if all vertices of an object are visible by objects in the scene w.r.t. the camera.

    Args:
        obj (bpy.types.Object): the object.

    Returns:
        boolean: visibility
    """
    for vertex in obj.data.vertices:
        coords = obj.matrix_world @ vertex.co
        if not is_ray_hit_vertex(coords, DEBUG=DEBUG):
            return False
    return True


def is_ray_hit_obj(vtx: Vector, ray_origin: Vector, ray_direction: Vector, helper_cube_scale: float = 0.0001, DEBUG: bool = False) -> bool:
    """Checks if a vertex is occluded by objects in the scene w.r.t. a given ray.

    Args:
        vtx (Vector): the world space x, y and z coordinates of the vertex.
        ray_origin (Vector): origin point of the ray
        ray_direction (Vector): direction vector of the ray
        helper_cube_scale (float, optional): Scale of helper geometry. Defaults to 0.0001.
        DEBUG (bool, optional): Enable debug visualization. Defaults to False.

    Returns:
        boolean: visibility
    """
    vtx = Vector(vtx)
    ray_origin = Vector(ray_origin)
    ray_direction = Vector(ray_direction).normalized()

    # add small cube around coord to make sure the ray will intersect
    # as the ray_cast is not always accurate
    # cf https://blender.stackexchange.com/a/87755
    with AuxiliaryCube(location=vtx, scale=(
            helper_cube_scale, helper_cube_scale, helper_cube_scale)):

        hit, location, _, _, _, _ = bpy.context.scene.ray_cast(
            bpy.context.view_layer.depsgraph,
            origin=ray_origin + ray_direction * 0.0001,  # avoid self intersection
            direction=ray_direction,
        )

        if DEBUG:
            print(f"hit location: {location}")
            bpy.ops.mesh.primitive_ico_sphere_add(
                location=location, scale=(
                    helper_cube_scale, helper_cube_scale, helper_cube_scale)
            )

        if not hit:
            raise ValueError(
                "No hit found, this should not happen as the ray should always hit the vertex itself.")
        # if the hit is the vertex itself, it is not occluded
        if (location - vtx).length < helper_cube_scale * 2:
            return False
    return True


# https://github.com/DLR-RM/BlenderProc/issues/990#issuecomment-1764989373
def object_hit_by_ray(ray_origin: Vector, ray_direction: Vector, DEBUG: bool = False) -> Optional[bpy.types.Object]:
    """Checks if a vertex is occluded by objects in the scene w.r.t. a given ray.

    Args:
        ray_origin (Vector): origin point of the ray
        ray_direction (Vector): direction vector of the ray
        DEBUG (bool, optional): Enable debug visualization. Defaults to False.

    Returns:
        bpy.types.Object: the object that was hit
    """
    hit, location, _, _, obj, _ = bpy.context.scene.ray_cast(
        bpy.context.view_layer.depsgraph,
        origin=ray_origin + ray_direction * 0.0001,  # avoid self intersection
        direction=ray_direction,
    )

    if DEBUG:
        print(
            f"hit location: {location} with object {obj.name if obj else 'None'}")

    if hit:
        return obj
    else:
        return None


def is_object_pickable(obj: bpy.types.Object, vertex_min_distance: float = 0.01, helper_cube_scale: float = 0.0001, DEBUG: bool = False) -> bool:
    """Checks if the object is pickable, the object is pickable only if there's no other object on the top of it.

    Args:
        obj (bpy.types.Object): The object to check for pickability.
        vertex_min_distance (float, optional): Minimum distance between sampled vertices in 2D projection. Defaults to 0.01.
        helper_cube_scale (float, optional): Scale of helper cube for ray intersection. Defaults to 0.0001.
        DEBUG (bool, optional): Enable debug visualization. Defaults to False.

    Returns:
        bool: True if the object is pickable (no occlusion from above), False otherwise.
    """

    # Project all vertices onto a 2D plane parallel to XY-plane
    projected_vertices = [obj.matrix_world @
                          vtx.co for vtx in obj.data.vertices]
    # Convert to numpy array for better performance
    projected_vertices = np.array(projected_vertices)

    # Sample vertices based on minimum distance in 2D projection
    sampled_vertices = []
    if len(projected_vertices) > 0:
        # Start with the first vertex
        sampled_vertices.append(projected_vertices[0].tolist())

        # For each remaining vertex, check distance to all sampled vertices
        for i in range(1, len(projected_vertices)):
            vertex = projected_vertices[i]

            # Calculate distances to all sampled vertices (only using x,y coordinates)
            if sampled_vertices:
                sampled_np = np.array(sampled_vertices)[
                    :, :2]  # Only x,y coordinates
                dists = np.sqrt(np.sum((sampled_np - vertex[:2])**2, axis=1))

                # If all distances are greater than minimum, add this vertex
                if np.all(dists >= vertex_min_distance):
                    sampled_vertices.append(vertex.tolist())

    if DEBUG:
        print(
            f"Sampled {len(sampled_vertices)} vertices out of {len(projected_vertices)}")

    # Perform ray casting from above for each sampled vertex
    for x, y, z in sampled_vertices:
        # Create ray from infinity in +z direction (top-down)
        ray_origin = Vector((x, y, 1000.0))  # Far away in +z direction
        # Pointing down in -z direction
        ray_direction = Vector((0.0, 0.0, -1.0))

        # Add small cube around coordinate to ensure ray intersection
        with AuxiliaryCube(location=Vector((x, y, z)), scale=(helper_cube_scale, helper_cube_scale, helper_cube_scale)) as cube:
            # Cast ray and check for intersection
            hit_obj = object_hit_by_ray(ray_origin, ray_direction, DEBUG=DEBUG)

            if not hit_obj:
                raise ValueError(
                    "No hit found, this should not happen as the ray should always hit the vertex itself.")

            # If the hit object is not our target object, then our object is occluded
            if hit_obj.name != obj.name and hit_obj.name != cube.name:
                return False

    return True


def is_object_all_within_view(camera: bpy.types.Object, obj: bpy.types.Object, DEBUG: bool = False) -> bool:
    """Checks if all vertices of an object are visible within the camera's view.

    Args:
        camera (bpy.types.Object): the camera.
        obj (bpy.types.Object): the object.

    Returns:
        boolean: visibility
    """
    cs, ce = camera.data.clip_start, camera.data.clip_end

    for v in obj.data.vertices:
        co_ndc = world_to_camera_view(
            bpy.context.scene, camera, obj.matrix_world @ v.co)
        # check wether point is inside frustum
        if not (0.0 < co_ndc.x < 1.0 and
                0.0 < co_ndc.y < 1.0 and
                cs < co_ndc.z < ce):
            return False
    return True


def is_obj_outside_rollcage(obj: bpy.types.Object, rollcage: MeshObject, margin: float = 0.01, min_z_clearance: float = 0.05) -> bool:
    """
    Returns True if the object is completely outside the rollcage volume.
    Combines bounding box corners and object center for robustness.
    """
    rollcage_obj = rollcage.blender_obj
    obj = obj if isinstance(obj, bpy.types.Object) else obj.blender_obj

    # World-space rollcage bounds
    rollcage_bb = [rollcage_obj.matrix_world @ Vector(corner) for corner in rollcage_obj.bound_box]
    rollcage_min = Vector((min(v[i] for v in rollcage_bb) for i in range(3)))
    rollcage_max = Vector((max(v[i] for v in rollcage_bb) for i in range(3)))

    # Object bounding box corners
    obj_bb = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]

    # Check if any corner is clearly inside all axes
    for corner in obj_bb:
        x_inside = rollcage_min.x - margin <= corner.x <= rollcage_max.x + margin
        y_inside = rollcage_min.y - margin <= corner.y <= rollcage_max.y + margin
        z_inside = corner.z >= (rollcage_min.z + min_z_clearance)
        if x_inside and y_inside and z_inside:
            return False  # At least one corner is clearly inside volume

    # Fallback: check center of mass
    center = obj.matrix_world @ obj.location
    cx_inside = rollcage_min.x - margin <= center.x <= rollcage_max.x + margin
    cy_inside = rollcage_min.y - margin <= center.y <= rollcage_max.y + margin
    cz_inside = center.z >= (rollcage_min.z + min_z_clearance)
    if cx_inside and cy_inside and cz_inside:
        return False

    return True  # Fully outside



def binary_mask_to_rle(binary_mask: np.ndarray, bbox: List[int]) -> List[int]:
    """Converts a binary mask to COCOs run-length encoding (RLE) format. Instead of outputting
    a mask image, you give a list of start pixels and how many pixels after each of those
    starts are included in the mask.
    :param binary_mask: a 2D binary numpy array where '1's represent the object
    :return: Mask in RLE format
    """
    rle: List[int] = []

    # Extract mask within bbox
    x, y, w, h = bbox
    mask_roi = binary_mask[y:y+h, x:x+w]
    # Flatten mask in column-major order and group consecutive values
    mask_flat = mask_roi.ravel()
    # Initialize RLE encoding
    for value, group in groupby(mask_flat):
        # Count length of current group
        group_length = len(list(group))

        # Add length and value (0 or 1)
        rle.append(group_length)
        rle.append(int(value))

    return rle


def rle_to_binary_mask(rle: List[int], width: int, height: int) -> np.ndarray:
    """Converts a run-length encoding (RLE) to a binary mask.

    :param rle: List of integers representing the RLE encoding
    :param width: Width of the output mask
    :param height: Height of the output mask
    :return: Binary mask as a numpy array
    """
    # Input validation
    if not rle:
        return np.zeros((0, 0), dtype=np.int32)

    if len(rle) % 2 != 0:
        print(f"Error: RLE size is not even: {len(rle)}")
        return np.zeros((0, 0), dtype=np.int32)

    # Verify total size matches dimensions
    total_pixels = sum(rle[i] for i in range(0, len(rle), 2))
    if total_pixels != width * height:
        print(f"Error: RLE size {total_pixels} does not match image size: "
              "{width}x{height}={width * height}")
        return np.zeros((0, 0), dtype=np.int32)

    # Create mask and fill using RLE
    mask = np.zeros((height, width), dtype=np.int32)
    i, j = 0, 0

    for idx in range(0, len(rle), 2):
        count = rle[idx]
        pixel = rle[idx + 1]

        for _ in range(count):
            mask[i, j] = pixel
            j += 1
            if j == width:
                j = 0
                i += 1

    return mask


def add_gaussian_shifts(depth, std=1/2.0):

    rows, cols = depth.shape
    gaussian_shifts = np.random.normal(0, std, size=(rows, cols, 2))
    gaussian_shifts = gaussian_shifts.astype(np.float32)
    breakpoint()

    # creating evenly spaced coordinates
    xx = np.linspace(0, cols-1, cols)
    yy = np.linspace(0, rows-1, rows)

    # get xpixels and ypixels
    xp, yp = np.meshgrid(xx, yy)

    xp = xp.astype(np.float32)
    yp = yp.astype(np.float32)

    xp_interp = np.minimum(np.maximum(
        xp + gaussian_shifts[:, :, 0], 0.0), cols)
    yp_interp = np.minimum(np.maximum(
        yp + gaussian_shifts[:, :, 1], 0.0), rows)

    depth_interp = cv2.remap(depth, xp_interp, yp_interp, cv2.INTER_LINEAR)
    # depth_interp[depth == 0] = 0

    return depth_interp


def apply_depth_noise(depth: np.ndarray, noise_p1: float = 0.05, noise_p2: float = 0.2, noise_p3: float = 0.01, noise_a1: float = 0.5) -> np.ndarray:
    """Applies normal noise to the depth map.

    :param depth: Depth map to apply noise to. In meters.
    :param noise_level: Standard deviation of the noise.
    :return: Noisy depth map.
    """

    # Create a copy of the depth map for processing
    processed_depth = depth.copy()

    # Define the kernel size for neighborhood consideration
    kernel_size = 5

    # Create a mean filter kernel for valid depth values
    # We'll use OpenCV's filter2D for efficient convolution
    kernel = np.ones((kernel_size, kernel_size), np.float32) / \
        (kernel_size * kernel_size)

    # Create a mask of valid depth values
    valid_mask = (depth > 0).astype(np.float32)

    # Apply the mean filter to both the depth map and the mask
    depth_sum = cv2.filter2D(depth, -1, kernel)
    weight_sum = cv2.filter2D(valid_mask, -1, kernel)

    # Avoid division by zero
    weight_sum[weight_sum == 0] = 1.0

    # Calculate the weighted average of valid depths in the neighborhood
    neighborhood_mean = depth_sum / weight_sum

    # Generate binary salt and pepper noise mask for the entire image
    noise_mask = np.random.random(processed_depth.shape)
    # Salt noise (set to 1) for noise_p1 percent of valid pixels
    salt_mask = (noise_mask < noise_p1)
    weight_mask = np.zeros_like(processed_depth)
    weight_mask[salt_mask] = 1.0

    # Blend the original depth with the neighborhood mean (90% original, 10% neighborhood)
    processed_depth[processed_depth > 0] = ((1 - noise_a1) * processed_depth[processed_depth > 0] + noise_a1 * neighborhood_mean[processed_depth > 0]) * \
        weight_mask[processed_depth > 0] + \
        (1 - weight_mask[processed_depth > 0]) * \
        processed_depth[processed_depth > 0]

    # Generate binary salt and pepper noise mask for the entire image
    noise_mask = np.random.random(processed_depth.shape)
    # Salt noise (set to 1) for noise_p2 percent of valid pixels
    salt_mask = (noise_mask < noise_p2)
    weight_mask = np.zeros_like(processed_depth)
    weight_mask[salt_mask] = 1.0

    # Add normal noise to the depth with a fixed maximum noise of 10 cm (0.1 m)
    fixed_noise_sigma = noise_p3   # 1cm
    noise = np.random.normal(0, fixed_noise_sigma, processed_depth.shape)

    processed_depth[processed_depth > 0] = np.maximum(
        0, processed_depth[processed_depth > 0] + noise[processed_depth > 0] * weight_mask[processed_depth > 0])

    return processed_depth


def write_flink(output_dir: str,
                instance_segmaps: List[np.ndarray],
                instance_attribute_maps: List[dict],
                rollcage: bpy.types.Object,
                depths: Optional[List[np.ndarray]] = None, colors: Optional[List[np.ndarray]] = None,
                color_file_format: str = "PNG", dataset: str = "", noisy_depth: dict | None = None,
                jpg_quality: int = 95,
                camera_id: str = "CAMID",
                tags: List[str] | List[List[str]] = []) -> None:
    """ Writes images, depth maps, labels and metadata in the Flink dataset format.

    :param output_dir: Path to the output directory.
    :param depths: List of depth images in m to save.
    :param colors: List of color images to save.
    :param color_file_format: File type to save color images. Available: "PNG", "JPEG"
    :param dataset: Name of the dataset. Used as a subdirectory name.
    :param jpg_quality: JPEG quality if color_file_format is "JPEG".
    :param camera_id: Camera ID prefix for filenames.
    :param tags: List of tags to add to the metadata. Can be a single list of tags or a list of lists of tags. If it is a list of lists, each list should correspond to a frame. If it is a single list, the same tags will be added to all frames.
    """
    # Select target objects
    scene_objects = []
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH' and not obj.hide_render:
            scene_objects.append(MeshObject(obj))
        elif obj.hide_render:
            print(f"object {obj.name} is hidden")

    with HideMeshWithProperty(scene_objects, property_name="flink_metadata", inverse=True):
        # check data conformity
        assert len(instance_attribute_maps) == len(
            instance_segmaps), "instance_attribute_maps and instance_segmaps must have the same length."
        assert isinstance(
            tags, list), "Tags must be a list of lists or a single list."
        if len(tags) > 0:
            if isinstance(tags[0], list):
                # tags is a list of lists
                assert len(tags) == len(
                    instance_segmaps), "Tags and instance_segmaps must have the same length."
            else:
                # tags is a single list
                tags = [tags] * len(instance_segmaps)
        else:
            tags = [[]] * len(instance_segmaps)

        if depths is not None:
            assert len(depths) == len(
                instance_segmaps), "Depths and instance_segmaps must have the same length."
        if colors is not None:
            assert len(colors) == len(
                instance_segmaps), "Colors and instance_segmaps must have the same length."

        # Create output directories
        dataset_dir: Path = Path(output_dir) / \
            dataset if dataset else Path(output_dir)

        # Create required subdirectories
        subdirs = ['depth', 'images', 'labels', 'metadata']
        for subdir in subdirs:
            dir_path = dataset_dir / subdir
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
            else:
                raise FileExistsError(
                    f"The output folder already exists: {dir_path}")

        # Run pickability check once for each object, because it's expensive and fixed for all frames
        pickability_info_cache = {}

        # Go through all frames
        for frame_id in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end):
            # Set frame
            bpy.context.scene.frame_set(frame_id)

            # Generate timestamp-based filename
            timestamp = time.time()
            filename_stem = f"{frame_id:012d}_{timestamp}"

            # Save color image
            if colors is not None:
                color_idx = frame_id - bpy.context.scene.frame_start
                color = colors[color_idx]
                # Convert RGB to BGR for OpenCV
                color_bgr = color.copy()
                color_bgr[..., :3] = color_bgr[..., :3][..., ::-1]

                if color_file_format == "PNG":
                    image_path = os.path.join(
                        dataset_dir, 'images', f"{filename_stem}.png")
                    cv2.imwrite(image_path, color_bgr)
                elif color_file_format == "JPEG":
                    image_path = os.path.join(
                        dataset_dir, 'images', f"{filename_stem}.jpg")
                    cv2.imwrite(image_path, color_bgr, [
                                int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
                else:
                    raise ValueError(f"Unknown color_file_format: "
                                     "{color_file_format}")

            # Save depth image
            if depths is not None:
                depth_idx = frame_id - bpy.context.scene.frame_start
                depth = depths[depth_idx]
                if noisy_depth is not None:
                    depth = apply_depth_noise(depth, **noisy_depth)
                    # depth = add_gaussian_shifts(depth)
                depth[depth > 1e2] = 0    # set invalid depth to 0
                # Convert depth to uint16 PNG
                # Convert to mm
                depth_mm = (depth * 1000).astype(np.uint16)
                depth_path = os.path.join(dataset_dir, 'depth', f"{filename_stem}.png")
                cv2.imwrite(depth_path, depth_mm)

            # Generate and save metadata
            metadata = _FlinkWriterUtility.get_frame_metadata(tags[frame_id])
            metadata_path = os.path.join(
                dataset_dir, 'metadata', f"{filename_stem}.json")
            _FlinkWriterUtility.write_json(metadata_path, metadata)

            labels = _FlinkWriterUtility.get_frame_labels(
                instance_segmaps[frame_id], instance_attribute_maps[frame_id], scene_objects, pickability_info_cache, rollcage)
            labels_path = os.path.join(dataset_dir, 'labels', f"{filename_stem}.json")
            _FlinkWriterUtility.write_json(labels_path, labels)


class _FlinkWriterUtility:
    """Utility class for the FlinkWriter."""

    @staticmethod
    def write_json(path: str, content: Dict[str, Any]) -> None:
        """Writes content to a JSON file.

        :param path: Path to output JSON file.
        :param content: Dictionary to save.
        """
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(content, f, indent=2)

    @staticmethod
    def get_frame_metadata(tags: List[str]) -> Dict[str, Any]:
        """Generates metadata for the current frame according to the Flink schema.

        :param tags: List of tags to add to the metadata.
        :return: Dictionary containing camera metadata.
        """
        cam_K = _WriterUtility.get_cam_attribute(
            bpy.context.scene.camera, 'cam_K')
        cam_matrix = [list(map(float, row)) for row in cam_K]

        # Get camera pose
        cam2world_matrix = Matrix(_WriterUtility.get_cam_attribute(bpy.context.scene.camera, 'cam2world_matrix',
                                                                   local_frame_change=["X", "-Y", "-Z"]))

        # Extract rotation matrix from cam2world_matrix (3x3 upper left block)
        quat = cam2world_matrix.to_quaternion()

        # Convert rotation matrix to axis-angle representation
        # This gives us a vector whose direction is the rotation axis and magnitude is the rotation angle
        rotation = [a * quat.angle for a in quat.axis]
        position = list(cam2world_matrix.to_translation())

        return {
            "metadata": {
                "camera_matrix": cam_matrix,
                "position": position,
                "rotation": rotation,
                "tags": tags
            }
        }

    @staticmethod
    def get_frame_labels(inst_segmap: np.ndarray, inst_attribute_map: dict, objs: List[MeshObject], pickability_info_cache: Dict[int, Dict[str, str]], rollcage: bpy.types.Object):
        """Generates coco annotations for images

        :param inst_segmap: instance segmentation map
        :param inst_attribute_map: per-frame mappings with idx, class and optionally attributes
        :param objs: list of objects in the scene
        :return: dict containing coco annotations
        """

        obj_id_2_mesh_map = {}
        for obj in objs:
            if not obj.has_cp("obj_id"):
                continue
            assert isinstance(obj, MeshObject), "Only MeshObject is supported"
            obj_id_2_mesh_map[obj.get_cp("obj_id")] = obj

        inst_idx_2_obj_id_map = {}
        for inst_attr in inst_attribute_map:
            # skip background
            if inst_attr["obj_id"] is not None:
                inst_idx_2_obj_id_map[inst_attr["idx"]] = inst_attr["obj_id"]

        annotations: List[Dict[str, Union[str, int]]] = []

        # Go through all objects visible in this image
        instances = np.unique(inst_segmap)
        # Remove background
        instances = np.delete(instances, np.where(instances == 0))

        for inst_idx in instances:
            if inst_idx in inst_idx_2_obj_id_map:
                # Calc object mask
                binary_inst_mask = np.where(inst_segmap == inst_idx, 1, 0)
                if np.all(binary_inst_mask == 0):
                    print(f'skipping all zero instance {inst_idx}')
                    continue

                obj_mesh = obj_id_2_mesh_map[inst_idx_2_obj_id_map[inst_idx]]

                flink_metadata: dict = obj_mesh.get_cp("flink_metadata")

                assert flink_metadata["obj_id"] == inst_idx_2_obj_id_map[
                    inst_idx], f"Object id mismatch, expected {inst_idx_2_obj_id_map[inst_idx]}, but got {flink_metadata['obj_id']}"

                annotation = _FlinkWriterUtility.create_annotation_info(
                    inst_idx_2_obj_id_map[inst_idx],
                    flink_metadata["category_id"],
                    binary_inst_mask,
                    obj_mesh,
                    pickability_info_cache,
                    rollcage
                )
                annotations.append(annotation)

        flink_labels = {
            "segmentations": annotations
        }

        return flink_labels

    @staticmethod
    def get_pickability_info(obj_id: int, obj_mesh: MeshObject, cached_pickability_info: Dict[int, Dict[str, str]], rollcage: bpy.types.Object) -> Dict[str, str]:
        """Returns pickability info for an object"""
        if obj_id in cached_pickability_info:
            print(f"obj {obj_id} pickability info already cached")
            return cached_pickability_info[obj_id]

        time_start = time.time()
        if is_obj_outside_rollcage(obj_mesh.blender_obj, rollcage):
            cached_pickability_info[obj_id] = {
                "blenderproc": "difficult",
            }
        elif is_object_pickable(obj_mesh.blender_obj, DEBUG=False):
            cached_pickability_info[obj_id] = {
                "blenderproc": "free",
            }
        else:
            cached_pickability_info[obj_id] = {
                "blenderproc": "occupied"
            }
        time_end = time.time()
        print(
            f"obj {obj_id} pickability check time taken: {time_end - time_start} seconds")
        return cached_pickability_info[obj_id]

    @staticmethod
    def create_annotation_info(object_id: int, category_id: str, binary_mask: np.ndarray, obj_mesh: MeshObject, pickability_info_cache: Dict[int, Dict[str, str]], rollcage: bpy.types.Object) -> Optional[Dict[str, Union[str, int]]]:
        """Creates info section of coco annotation

        :param category_id: name of the category
        :param binary_mask: A binary image mask of the object with the shape [H, W].
        """

        bounding_box = _FlinkWriterUtility.bbox_from_binary_mask(binary_mask)

        mask = binary_mask_to_rle(binary_mask, bounding_box)

        w2obj = Matrix(obj_mesh.get_local2world_mat())

        dim_local = obj_mesh.get_bound_box(
            local_coords=True) * obj_mesh.get_scale()  # Get world coordinates
        min_point = np.min(dim_local, axis=0)
        max_point = np.max(dim_local, axis=0)
        center = (min_point + max_point) / 2

        quat = w2obj.to_quaternion()

        pose_estimation = {
            "obj_name": obj_mesh.get_name(),
            "w2obj_t": list(w2obj.to_translation()),
            "w2obj_r": [a * quat.angle for a in quat.axis],
            "obj2center": center.tolist(),
            "size": (max_point - min_point).tolist()
        }

        # Do the ray casting to see if the object is within the view
        all_within_view = is_object_all_within_view(
            bpy.context.scene.camera, obj_mesh.blender_obj)

        pick_class = _FlinkWriterUtility.get_pickability_info(
            object_id, obj_mesh, pickability_info_cache, rollcage)

        annotation_info: Dict[str, Union[str, int]] = {
            "category_id": category_id,
            "bbox": bounding_box,
            "mask": mask,
            "pick_class": pick_class,
            "all_within_view": all_within_view,
            "object_id": object_id,
            "pose_estimation": pose_estimation,
            "shortuuid": str(uuid.uuid4())[:8]
        }
        return annotation_info

    @staticmethod
    def bbox_from_binary_mask(binary_mask: np.ndarray) -> List[int]:
        """ Returns the smallest bounding box containing all pixels marked "1" in the given image mask.

        :param binary_mask: A binary image mask with the shape [H, W].
        :return: The bounding box represented as [x, y, width, height]
        """
        # Find all columns and rows that contain 1s
        rows = np.any(binary_mask, axis=1)
        cols = np.any(binary_mask, axis=0)
        # Find the min and max col/row index that contain 1s
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        # Calc height and width
        h = rmax - rmin + 1
        w = cmax - cmin + 1
        return [int(cmin), int(rmin), int(w), int(h)]
