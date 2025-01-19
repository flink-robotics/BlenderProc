"""Allows rendering the content of the scene in the Flink dataset format."""

import os
import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from itertools import groupby
import uuid

import numpy as np
import cv2
import bpy
from mathutils import Matrix

from blenderproc.python.types.MeshObjectUtility import MeshObject
from blenderproc.python.writer.WriterUtility import _WriterUtility

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


def write_flink(output_dir: str,
                instance_segmaps: List[np.ndarray],
                instance_attribute_maps: List[dict],
                depths: Optional[List[np.ndarray]] = None, colors: Optional[List[np.ndarray]] = None,
                color_file_format: str = "PNG", dataset: str = "",
                jpg_quality: int = 95,
                camera_id: str = "CAMID") -> None:
    """ Writes images, depth maps, labels and metadata in the Flink dataset format.

    :param output_dir: Path to the output directory.
    :param depths: List of depth images in m to save.
    :param colors: List of color images to save.
    :param color_file_format: File type to save color images. Available: "PNG", "JPEG"
    :param dataset: Name of the dataset. Used as a subdirectory name.
    :param jpg_quality: JPEG quality if color_file_format is "JPEG".
    :param camera_id: Camera ID prefix for filenames.
    """
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

    # Select target objects
    target_objects = []
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH' and not obj.hide_render:
            target_objects.append(MeshObject(obj))

    # Go through all frames
    for frame_id in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end):
        # Set frame
        bpy.context.scene.frame_set(frame_id)

        # Generate timestamp-based filename
        timestamp = f"{frame_id:012d}"
        filename_stem = f"{camera_id}_{timestamp}"

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
            depth_path = os.path.join(
                dataset_dir, 'depth', f"{filename_stem}.png")
            # Convert depth to uint16 PNG
            depth_mm = (depth * 1000).astype(np.uint16)  # Convert to mm
            cv2.imwrite(depth_path, depth_mm)

        # Generate and save metadata
        metadata = _FlinkWriterUtility.get_frame_metadata()
        metadata_path = os.path.join(
            dataset_dir, 'metadata', f"{filename_stem}.json")
        _FlinkWriterUtility.write_json(metadata_path, metadata)

        labels = _FlinkWriterUtility.get_frame_labels(
            instance_segmaps[frame_id], instance_attribute_maps[frame_id], target_objects)
        labels_path = os.path.join(
            dataset_dir, 'labels', f"{filename_stem}.json")
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
    def get_frame_metadata() -> Dict[str, Any]:
        """Generates metadata for the current frame according to the Flink schema.

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
                "tags": []  # Optional tags can be added here
            }
        }

    @staticmethod
    def get_frame_labels(inst_segmap: np.ndarray, inst_attribute_map: dict, objs: List[MeshObject]):
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

                annotation = _FlinkWriterUtility.create_annotation_info(
                    inst_idx_2_obj_id_map[inst_idx],
                    "box",
                    binary_inst_mask,
                    obj_mesh
                )
                annotations.append(annotation)

        flink_labels = {
            "segmentations": annotations
        }

        return flink_labels

    @staticmethod
    def create_annotation_info(object_id: int, category_id: int, binary_mask: np.ndarray, obj_mesh: MeshObject) -> Optional[Dict[str, Union[str, int]]]:
        """Creates info section of coco annotation

        :param category_id: Id of the category
        :param binary_mask: A binary image mask of the object with the shape [H, W].
        """

        bounding_box = _FlinkWriterUtility.bbox_from_binary_mask(binary_mask)

        mask = binary_mask_to_rle(binary_mask, bounding_box)

        w2obj = Matrix(obj_mesh.get_local2world_mat())

        dim_local = obj_mesh.get_bound_box(
            local_coords=True)  # Get world coordinates
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

        annotation_info: Dict[str, Union[str, int]] = {
            "category_id": str(category_id),
            "bbox": bounding_box,
            "mask": mask,
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
