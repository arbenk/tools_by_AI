import bpy
import math
import os
import html
import re
import json
import uuid
from mathutils import Vector, Matrix
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    CollectionProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import Operator, Panel, PropertyGroup
from bpy_extras.io_utils import ExportHelper


# -----------------------------------------------------------------------------
# Constants / tags / units
# -----------------------------------------------------------------------------

ADDON_VERSION = (1, 7, 0)
GENERATED_TAG = "flatfab_generated"
SOURCE_TAG = "flatfab_source_object"
WIDTH_TAG = "flatfab_width_mm"
HEIGHT_TAG = "flatfab_height_mm"
SHEET_TAG = "flatfab_sheet_preview"
SHEET_OBJECT_NAME = "FLATFAB_SHEET_PREVIEW"
LOCK_TAG = "flatfab_lock_group"
PROJECTION_X_TAG = "flatfab_projection_x"
PROJECTION_Y_TAG = "flatfab_projection_y"
PROJECTION_VERSION_TAG = "flatfab_projection_version"
UV_MAP_NAME = "FlatFab_Layout"
LOCK_NAME_RE = re.compile(r"(?:^|[_\-.])(lock[0-9A-Za-z]+)(?=$|[_\-.])", re.IGNORECASE)



def mm_to_bu(scene, mm):
    """Millimeters -> Blender world units using Scene Unit Scale."""
    scale_length = scene.unit_settings.scale_length or 1.0
    return (mm / 1000.0) / scale_length


def bu_to_mm(scene, value):
    """Blender world units -> millimeters using Scene Unit Scale."""
    scale_length = scene.unit_settings.scale_length or 1.0
    return value * scale_length * 1000.0


def mm_to_pt(mm):
    return mm * 72.0 / 25.4


def clean_name(name):
    """Filename-safe-ish name while preserving Unicode where the OS supports it."""
    forbidden = '<>:"/\\|?*\0'
    result = "".join("_" if ch in forbidden else ch for ch in name).strip()
    return result or "flatfab_layout"


# -----------------------------------------------------------------------------
# Geometry analysis / flattening
# -----------------------------------------------------------------------------


def polygon_world_data(mesh, matrix_world):
    world_verts = [matrix_world @ v.co for v in mesh.vertices]
    face_data = []

    for poly in mesh.polygons:
        pts = [world_verts[i] for i in poly.vertices]
        if len(pts) < 3:
            continue

        area_vec = Vector((0.0, 0.0, 0.0))
        for i, p in enumerate(pts):
            q = pts[(i + 1) % len(pts)]
            area_vec += p.cross(q)
        area_vec *= 0.5

        area = area_vec.length
        if area <= 1.0e-12:
            continue

        face_data.append(
            {
                "poly": poly,
                "verts": pts,
                "normal": area_vec.normalized(),
                "area": area,
            }
        )

    return world_verts, face_data


def find_dominant_axis(face_data, angle_tol_deg):
    """Cluster face normals ignoring sign and return the largest-area direction."""
    cos_tol = math.cos(math.radians(angle_tol_deg))
    clusters = []

    for fd in sorted(face_data, key=lambda item: item["area"], reverse=True):
        n = fd["normal"]
        area = fd["area"]
        matched = None

        for cluster in clusters:
            if abs(n.dot(cluster["axis"])) >= cos_tol:
                matched = cluster
                break

        if matched is None:
            clusters.append(
                {
                    "axis": n.copy(),
                    "sum_vec": n * area,
                    "area": area,
                }
            )
            continue

        aligned = n if n.dot(matched["axis"]) >= 0.0 else -n
        matched["sum_vec"] += aligned * area
        matched["area"] += area
        if matched["sum_vec"].length > 1.0e-12:
            matched["axis"] = matched["sum_vec"].normalized()

    if not clusters:
        return None

    return max(clusters, key=lambda c: c["area"])["axis"].normalized()


def axis_from_mode(obj, face_data, mode, angle_tol_deg):
    if mode == "AUTO":
        return find_dominant_axis(face_data, angle_tol_deg)

    local_axis = {
        "LOCAL_X": Vector((1.0, 0.0, 0.0)),
        "LOCAL_Y": Vector((0.0, 1.0, 0.0)),
        "LOCAL_Z": Vector((0.0, 0.0, 1.0)),
    }[mode]

    normal_matrix = obj.matrix_world.to_3x3().inverted_safe().transposed()
    axis = normal_matrix @ local_axis
    if axis.length <= 1.0e-12:
        return None
    return axis.normalized()


def choose_surface_faces(face_data, world_verts, axis, angle_tol_deg, plane_tol):
    """Pick the more complete of the two extreme plate faces."""
    cos_tol = math.cos(math.radians(angle_tol_deg))
    projections = [co.dot(axis) for co in world_verts]
    if not projections:
        return []

    min_d = min(projections)
    max_d = max(projections)
    side_min = []
    side_max = []

    for fd in face_data:
        if abs(fd["normal"].dot(axis)) < cos_tol:
            continue

        ds = [p.dot(axis) for p in fd["verts"]]
        if max(abs(d - min_d) for d in ds) <= plane_tol:
            side_min.append(fd)
        if max(abs(d - max_d) for d in ds) <= plane_tol:
            side_max.append(fd)

    area_min = sum(fd["area"] for fd in side_min)
    area_max = sum(fd["area"] for fd in side_max)
    return side_max if area_max >= area_min else side_min


def make_in_plane_basis(obj, axis):
    """Use object local axes where possible so flattened orientation is stable."""
    mw3 = obj.matrix_world.to_3x3()
    candidates = [
        mw3 @ Vector((1.0, 0.0, 0.0)),
        mw3 @ Vector((0.0, 1.0, 0.0)),
        mw3 @ Vector((0.0, 0.0, 1.0)),
    ]

    best = None
    best_len = -1.0
    for candidate in candidates:
        projected = candidate - axis * candidate.dot(axis)
        length = projected.length
        if length > best_len:
            best = projected
            best_len = length

    if best is None or best.length < 1.0e-10:
        helper = Vector((1.0, 0.0, 0.0))
        if abs(axis.dot(helper)) > 0.9:
            helper = Vector((0.0, 1.0, 0.0))
        best = helper - axis * helper.dot(axis)

    u = best.normalized()
    v = axis.cross(u).normalized()
    return u, v


def boundary_edges_from_faces(surface_faces):
    """Remove coplanar internal edges and keep outer/hole/slot boundaries."""
    edge_counts = {}

    for fd in surface_faces:
        vertices = list(fd["poly"].vertices)
        count = len(vertices)
        for i in range(count):
            a = vertices[i]
            b = vertices[(i + 1) % count]
            key = (a, b) if a < b else (b, a)
            edge_counts[key] = edge_counts.get(key, 0) + 1

    return [edge for edge, count in edge_counts.items() if count == 1]


def normalize_2d(coords):
    min_x = min(p.x for p in coords)
    min_y = min(p.y for p in coords)
    return [Vector((p.x - min_x, p.y - min_y, 0.0)) for p in coords]


def rotate_long_side_horizontal(coords):
    width = max(p.x for p in coords) - min(p.x for p in coords)
    height = max(p.y for p in coords) - min(p.y for p in coords)

    if height <= width:
        return coords

    rotated = [Vector((p.y, -p.x, 0.0)) for p in coords]
    return normalize_2d(rotated)


def mesh_for_object(obj, depsgraph, use_modifiers):
    """Return (mesh, matrix_world, should_remove_mesh)."""
    if use_modifiers:
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = bpy.data.meshes.new_from_object(eval_obj, depsgraph=depsgraph)
        return mesh, eval_obj.matrix_world.copy(), True

    return obj.data, obj.matrix_world.copy(), False


def build_flat_projection_geometry(obj, depsgraph, scene, settings):
    """Build the exact local 2D geometry and affine source->flat projection used by flattening."""
    mesh, matrix_world, remove_mesh = mesh_for_object(
        obj, depsgraph, settings.use_modifiers
    )

    try:
        world_verts, face_data = polygon_world_data(mesh, matrix_world)
        if not face_data:
            raise RuntimeError("没有可用的面")

        axis = axis_from_mode(
            obj,
            face_data,
            settings.projection_mode,
            settings.angle_tolerance,
        )
        if axis is None:
            raise RuntimeError("无法识别板件主平面")

        surface_faces = choose_surface_faces(
            face_data,
            world_verts,
            axis,
            settings.angle_tolerance,
            mm_to_bu(scene, settings.plane_tolerance_mm),
        )
        if not surface_faces:
            raise RuntimeError(
                "找不到完整板面；可提高‘平面距离容差’，或改用 Local X/Y/Z 投影"
            )

        boundary_edges = boundary_edges_from_faces(surface_faces)
        if not boundary_edges:
            raise RuntimeError("没有检测到可用边界")

        u, v = make_in_plane_basis(obj, axis)
        used_indices = sorted({idx for edge in boundary_edges for idx in edge})
        index_map = {old: new for new, old in enumerate(used_indices)}

        raw_u = {old_i: world_verts[old_i].dot(u) for old_i in used_indices}
        raw_v = {old_i: world_verts[old_i].dot(v) for old_i in used_indices}
        min_u = min(raw_u.values())
        min_v = min(raw_v.values())
        max_u = max(raw_u.values())
        max_v = max(raw_v.values())
        width_before = max_u - min_u
        height_before = max_v - min_v
        rotate_90 = settings.long_side_horizontal and height_before > width_before

        coords = []
        for old_i in used_indices:
            x = raw_u[old_i] - min_u
            y = raw_v[old_i] - min_v
            if rotate_90:
                coords.append(Vector((y, width_before - x, 0.0)))
            else:
                coords.append(Vector((x, y, 0.0)))

        edges = [(index_map[a], index_map[b]) for a, b in boundary_edges]

        # Store an affine mapping from the source object's local vertex coordinates
        # directly into the generated flat object's local XY. This makes later UV
        # updates independent of source-object transforms and exactly follows the
        # same orientation chosen when the flat outline was created.
        local_u = matrix_world.to_3x3().transposed() @ u
        local_v = matrix_world.to_3x3().transposed() @ v
        offset_u = matrix_world.translation.dot(u) - min_u
        offset_v = matrix_world.translation.dot(v) - min_v

        if rotate_90:
            row_x = (local_v.x, local_v.y, local_v.z, offset_v)
            row_y = (-local_u.x, -local_u.y, -local_u.z, width_before - offset_u)
        else:
            row_x = (local_u.x, local_u.y, local_u.z, offset_u)
            row_y = (local_v.x, local_v.y, local_v.z, offset_v)

        width = max(p.x for p in coords) - min(p.x for p in coords)
        height = max(p.y for p in coords) - min(p.y for p in coords)
        return {
            "coords": coords,
            "edges": edges,
            "row_x": row_x,
            "row_y": row_y,
            "width": width,
            "height": height,
        }
    finally:
        if remove_mesh:
            bpy.data.meshes.remove(mesh)


def store_projection_metadata(flat_obj, projection):
    flat_obj[PROJECTION_X_TAG] = list(projection["row_x"])
    flat_obj[PROJECTION_Y_TAG] = list(projection["row_y"])
    flat_obj[PROJECTION_VERSION_TAG] = 1


def projection_rows_from_flat(flat_obj):
    try:
        row_x = tuple(float(v) for v in flat_obj[PROJECTION_X_TAG])
        row_y = tuple(float(v) for v in flat_obj[PROJECTION_Y_TAG])
    except Exception:
        return None
    if len(row_x) != 4 or len(row_y) != 4:
        return None
    return row_x, row_y


def flatten_object(obj, depsgraph, target_collection, scene, settings):
    projection = build_flat_projection_geometry(obj, depsgraph, scene, settings)
    coords = projection["coords"]
    edges = projection["edges"]

    mesh_name = f"{settings.prefix}{obj.name}_MESH"
    object_name = f"{settings.prefix}{obj.name}"
    new_mesh = bpy.data.meshes.new(mesh_name)
    new_mesh.from_pydata(coords, edges, [])
    new_mesh.update()

    new_obj = bpy.data.objects.new(object_name, new_mesh)
    target_collection.objects.link(new_obj)

    new_obj[GENERATED_TAG] = True
    new_obj[SOURCE_TAG] = obj.name
    new_obj[WIDTH_TAG] = round(bu_to_mm(scene, projection["width"]), 4)
    new_obj[HEIGHT_TAG] = round(bu_to_mm(scene, projection["height"]), 4)
    new_obj["flatfab_version"] = ".".join(map(str, ADDON_VERSION))
    store_projection_metadata(new_obj, projection)
    new_obj.display_type = "WIRE"

    return new_obj, projection["width"], projection["height"]


# -----------------------------------------------------------------------------
# Parametric plate joints
# -----------------------------------------------------------------------------

JOINT_HELPER_TAG = "flatfab_joint_helper"
JOINT_ID_TAG = "flatfab_joint_id"
JOINT_TARGET_TAG = "flatfab_joint_target"
JOINT_OPERATION_TAG = "flatfab_joint_operation"
JOINT_HELPER_COLLECTION = "FLATFAB_JOINT_HELPERS"
JOINT_MOD_PREFIX = "FFJ_"


def _cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _poly_area2(poly):
    total = 0.0
    for i, p in enumerate(poly):
        q = poly[(i + 1) % len(poly)]
        total += p[0] * q[1] - q[0] * p[1]
    return total * 0.5


def _point_in_poly2(point, poly):
    x, y = point
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)):
            denom = (yj - yi) if abs(yj - yi) > 1.0e-15 else 1.0e-15
            x_hit = (xj - xi) * (y - yi) / denom + xi
            if x < x_hit:
                inside = not inside
        j = i
    return inside


def _boundary_loops(boundary_edges):
    adjacency = {}
    unused = set()
    for a, b in boundary_edges:
        key = (a, b) if a < b else (b, a)
        unused.add(key)
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)

    loops = []
    while unused:
        a, b = next(iter(unused))
        start = a
        prev = None
        cur = a
        loop = [cur]
        while True:
            candidates = []
            for nb in adjacency.get(cur, []):
                key = (cur, nb) if cur < nb else (nb, cur)
                if key in unused:
                    candidates.append(nb)
            if not candidates:
                break
            if prev is not None and len(candidates) > 1 and prev in candidates:
                candidates.remove(prev)
            nxt = candidates[0]
            key = (cur, nxt) if cur < nxt else (nxt, cur)
            unused.remove(key)
            prev, cur = cur, nxt
            if cur == start:
                break
            loop.append(cur)
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def _solidify_info(obj, axis):
    for mod in obj.modifiers:
        if mod.type != "SOLIDIFY" or not mod.show_viewport:
            continue
        try:
            local_n = (obj.matrix_world.to_3x3().transposed() @ axis).normalized()
            scale = (obj.matrix_world.to_3x3() @ local_n).length
            signed_thickness = float(mod.thickness) * scale
            return abs(signed_thickness), signed_thickness * float(mod.offset) * 0.5
        except Exception:
            continue
    return None


def joint_plate_frame(obj, scene, settings):
    """Stable manufacturing frame from the source mesh, before FlatFab booleans.

    Keeps *all* planar boundary loops.  The largest loop is still used as the
    primary outside boundary for edge-contact discovery, while all loops are
    retained for even/odd polygon clipping (concave outlines, holes and
    disjoint islands).
    """
    if obj.type != "MESH" or not obj.data.vertices or not obj.data.polygons:
        raise RuntimeError(f"{obj.name} 不是可识别的板材 Mesh")

    world_verts, face_data = polygon_world_data(obj.data, obj.matrix_world)
    axis = axis_from_mode(obj, face_data, settings.projection_mode, settings.angle_tolerance)
    if axis is None:
        raise RuntimeError(f"{obj.name} 无法识别板材法线")

    projections = [p.dot(axis) for p in world_verts]
    pmin, pmax = min(projections), max(projections)
    surface_faces = choose_surface_faces(
        face_data,
        world_verts,
        axis,
        settings.angle_tolerance,
        mm_to_bu(scene, settings.plane_tolerance_mm),
    )
    if not surface_faces:
        surface_faces = [fd for fd in face_data if abs(fd["normal"].dot(axis)) > 0.9]
    boundary = boundary_edges_from_faces(surface_faces)
    if not boundary:
        raise RuntimeError(f"{obj.name} 找不到板材外边界")

    u, v = make_in_plane_basis(obj, axis)
    loops = _boundary_loops(boundary)
    if not loops:
        raise RuntimeError(f"{obj.name} 找不到闭合板材边界")

    loop_worlds = []
    loop_polys = []
    loop_areas = []
    for indices in loops:
        pts_world = [world_verts[i].copy() for i in indices]
        pts_2d = [(p.dot(u), p.dot(v)) for p in pts_world]
        if len(pts_2d) < 3:
            continue
        loop_worlds.append(pts_world)
        loop_polys.append(pts_2d)
        loop_areas.append(abs(_poly_area2(pts_2d)))

    if not loop_polys:
        raise RuntimeError(f"{obj.name} 找不到有效板材边界")

    outer_index = max(range(len(loop_polys)), key=lambda i: loop_areas[i])
    outer_world = loop_worlds[outer_index]
    outer_2d = loop_polys[outer_index]
    centroid = sum(outer_world, Vector((0.0, 0.0, 0.0))) / len(outer_world)

    base_d = sum(p.dot(axis) for p in outer_world) / len(outer_world)
    solidify = _solidify_info(obj, axis)
    if solidify is not None and (pmax - pmin) <= mm_to_bu(scene, 0.01):
        thickness, center_shift = solidify
        center_d = base_d + center_shift
    else:
        thickness = max(0.0, pmax - pmin)
        center_d = (pmin + pmax) * 0.5

    if thickness <= mm_to_bu(scene, 0.001):
        raise RuntimeError(f"{obj.name} 没有可识别板厚；请先添加/启用实体化修改器")

    return {
        "obj": obj,
        "n": axis.normalized(),
        "u": u.normalized(),
        "v": v.normalized(),
        "outer_world": outer_world,
        "outer_2d": outer_2d,
        "loops_world": loop_worlds,
        "loops_2d": loop_polys,
        "loop_areas": loop_areas,
        "outer_index": outer_index,
        "area_2d": max(loop_areas) if loop_areas else 0.0,
        "centroid": centroid,
        "center_d": center_d,
        "thickness": thickness,
        "surface_ds": (center_d - thickness * 0.5, center_d + thickness * 0.5),
    }

def _project_point_to_frame_2d(frame, point):
    return (point.dot(frame["u"]), point.dot(frame["v"]))


def _project_point_to_plane(point, normal, d):
    return point + normal * (d - point.dot(normal))

def _frame_contains_2d(frame, point):
    """Even/odd containment across every boundary loop.

    This is intentionally loop-orientation independent, so it works for holes,
    concave outlines and multiple disconnected planar islands produced by the
    source mesh.
    """
    loops = frame.get("loops_2d") or [frame["outer_2d"]]
    inside = False
    for poly in loops:
        if len(poly) >= 3 and _point_in_poly2(point, poly):
            inside = not inside
    return inside


def _distance_point_segment2(point, a, b):
    px, py = point
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom <= 1.0e-20:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / denom
    t = max(0.0, min(1.0, t))
    qx, qy = ax + dx * t, ay + dy * t
    return math.hypot(px - qx, py - qy)


def _frame_boundary_clearance_2d(frame, point):
    loops = frame.get("loops_2d") or [frame["outer_2d"]]
    best = None
    for poly in loops:
        for i, a in enumerate(poly):
            b = poly[(i + 1) % len(poly)]
            d = _distance_point_segment2(point, a, b)
            if best is None or d < best:
                best = d
    return best if best is not None else 0.0


def _merge_intervals(intervals, eps=1.0e-8):
    if not intervals:
        return []
    items = sorted((min(a, b), max(a, b)) for a, b in intervals if abs(b - a) > eps)
    if not items:
        return []
    merged = [list(items[0])]
    for lo, hi in items[1:]:
        if lo <= merged[-1][1] + eps:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [(a, b) for a, b in merged]


def _line_plane_intersection(n1, d1, n2, d2):
    direction = n1.cross(n2)
    denom = direction.length_squared
    if denom <= 1.0e-16:
        return None, None
    point = (d1 * n2.cross(direction) + d2 * direction.cross(n1)) / denom
    return point, direction.normalized()


def _line_segments_inside_frame(frame, point, direction):
    """Clip an infinite 3D line against a planar plate using all polygon loops.

    The return value can contain multiple disjoint intervals.  Midpoint
    even/odd testing removes hole intervals and preserves concave/disconnected
    regions without any external geometry dependency.
    """
    p = _project_point_to_frame_2d(frame, point)
    d = (direction.dot(frame["u"]), direction.dot(frame["v"]))
    dlen2 = d[0] * d[0] + d[1] * d[1]
    if dlen2 <= 1.0e-16:
        return []

    loops = frame.get("loops_2d") or [frame["outer_2d"]]
    hits = []
    for poly in loops:
        for i, a in enumerate(poly):
            b = poly[(i + 1) % len(poly)]
            e = (b[0] - a[0], b[1] - a[1])
            den = _cross2(d, e)
            ap = (a[0] - p[0], a[1] - p[1])
            if abs(den) <= 1.0e-12:
                continue
            t = _cross2(ap, e) / den
            edge_t = _cross2(ap, d) / den
            if -1.0e-8 <= edge_t <= 1.0 + 1.0e-8:
                hits.append(t)

    if len(hits) < 2:
        return []
    hits.sort()
    unique = []
    for value in hits:
        if not unique or abs(value - unique[-1]) > 1.0e-7:
            unique.append(value)

    segments = []
    for a, b in zip(unique, unique[1:]):
        if b - a <= 1.0e-8:
            continue
        mid = (a + b) * 0.5
        test = (p[0] + d[0] * mid, p[1] + d[1] * mid)
        if _frame_contains_2d(frame, test):
            segments.append((a, b))
    return _merge_intervals(segments)

def _common_line_segments(frame_a, frame_b):
    """Return every disjoint overlap interval of the two plate center planes."""
    point, direction = _line_plane_intersection(
        frame_a["n"], frame_a["center_d"], frame_b["n"], frame_b["center_d"]
    )
    if point is None:
        return None
    seg_a = _line_segments_inside_frame(frame_a, point, direction)
    seg_b = _line_segments_inside_frame(frame_b, point, direction)
    overlaps = []
    for a0, a1 in seg_a:
        for b0, b1 in seg_b:
            lo, hi = max(a0, b0), min(a1, b1)
            if hi - lo > 1.0e-8:
                overlaps.append((lo, hi))
    overlaps = _merge_intervals(overlaps)
    if not overlaps:
        return None
    return {
        "origin": point,
        "direction": direction,
        "overlaps": overlaps,
        "segments_a": seg_a,
        "segments_b": seg_b,
    }


def _common_line_segment(frame_a, frame_b):
    """Compatibility wrapper returning the longest common interval."""
    common = _common_line_segments(frame_a, frame_b)
    if common is None:
        return None
    lo, hi = max(common["overlaps"], key=lambda pair: pair[1] - pair[0])
    result = dict(common)
    result.update({
        "start_t": lo,
        "end_t": hi,
        "start": common["origin"] + common["direction"] * lo,
        "end": common["origin"] + common["direction"] * hi,
    })
    return result

def _edge_face_contact(frame_a, frame_b, scene, settings):
    """Find an edge of A that lies on an actual solid surface of B.

    Contact length is clipped against B's full polygon set rather than a BBox
    projection.  This fixes face-sheet + Solidify workflows and naturally
    supports concave receivers and holes.
    """
    n_a, n_b = frame_a["n"], frame_b["n"]
    dot = abs(n_a.dot(n_b))
    max_dot = math.sin(math.radians(settings.joint_angle_tolerance_deg))
    if dot > max_dot:
        raise RuntimeError("边插要求两块板材板面近似垂直")

    expected_line = n_a.cross(n_b)
    if expected_line.length <= 1.0e-10:
        raise RuntimeError("两块板材没有稳定交线")
    expected_line.normalize()
    tol = mm_to_bu(scene, settings.joint_detect_tolerance_mm)
    best = None
    outer = frame_a["outer_world"]

    for i, p in enumerate(outer):
        q = outer[(i + 1) % len(outer)]
        edge = q - p
        length = edge.length
        if length <= 1.0e-10:
            continue
        ed = edge / length
        parallel = abs(ed.dot(expected_line))
        if parallel < math.cos(math.radians(settings.joint_angle_tolerance_deg)):
            continue

        for surface_d in frame_b["surface_ds"]:
            dp = abs(p.dot(n_b) - surface_d)
            dq = abs(q.dot(n_b) - surface_d)
            dist = max(dp, dq)
            if dist > max(tol * 2.0, 1.0e-7):
                continue

            # Clip the projected A edge against B's face polygon(s).  t is in
            # Blender units because ed is normalized.
            projected_p = _project_point_to_plane(p, n_b, frame_b["center_d"])
            clipped = _line_segments_inside_frame(frame_b, projected_p, ed)
            for c0, c1 in clipped:
                lo, hi = max(0.0, c0), min(length, c1)
                if hi - lo <= 1.0e-8:
                    continue
                start = p + ed * lo
                end = p + ed * hi
                mid = (start + end) * 0.5
                mid2 = _project_point_to_frame_2d(
                    frame_b, _project_point_to_plane(mid, n_b, frame_b["center_d"])
                )
                clearance = _frame_boundary_clearance_2d(frame_b, mid2)
                inward_sign = 1.0 if frame_b["center_d"] >= surface_d else -1.0
                candidate = {
                    "score": (dist, -clearance, -(hi - lo)),
                    "distance": dist,
                    "clearance": clearance,
                    "start": start,
                    "end": end,
                    "direction": ed,
                    "into_b": n_b * inward_sign,
                    "surface_d": surface_d,
                }
                if best is None or candidate["score"] < best["score"]:
                    best = candidate

    if best is None or best["distance"] > tol:
        distance_mm = bu_to_mm(scene, best["distance"]) if best else None
        suffix = f"（最近 {distance_mm:.3f} mm）" if distance_mm is not None else ""
        raise RuntimeError(f"找不到边贴面关系，请把其中一块板的实体边贴到另一块板的实体表面{suffix}")
    return best


def _try_edge_face_contact(frame_a, frame_b, scene, settings):
    try:
        return _edge_face_contact(frame_a, frame_b, scene, settings)
    except Exception:
        return None


def _resolve_edge_insert_roles(obj_a, obj_b, frame_a, frame_b, scene, settings):
    """Automatically decide edge/tooth plate versus face/slot plate."""
    ab = _try_edge_face_contact(frame_a, frame_b, scene, settings)
    ba = _try_edge_face_contact(frame_b, frame_a, scene, settings)
    if ab is None and ba is None:
        raise RuntimeError("自动识别失败：没有找到‘板边 → 另一板面’的贴合关系")
    if ab is not None and ba is None:
        return obj_a, obj_b, frame_a, frame_b, ab
    if ba is not None and ab is None:
        return obj_b, obj_a, frame_b, frame_a, ba

    # At a true T-joint the receiver-face contact is normally farther from the
    # receiver boundary.  Edge-edge corner contacts have near-zero clearance
    # in both directions and are better handled by 边连接.
    key_ab = (ab["distance"], -ab["clearance"], -(ab["end"] - ab["start"]).length)
    key_ba = (ba["distance"], -ba["clearance"], -(ba["end"] - ba["start"]).length)
    chosen = (obj_a, obj_b, frame_a, frame_b, ab) if key_ab <= key_ba else (obj_b, obj_a, frame_b, frame_a, ba)
    other = ba if chosen[0] is obj_a else ab
    if chosen[4]["clearance"] <= mm_to_bu(scene, settings.joint_detect_tolerance_mm) * 0.25 and other["clearance"] <= mm_to_bu(scene, settings.joint_detect_tolerance_mm) * 0.25:
        raise RuntimeError("检测结果更像边-边贴合；边插只处理‘边插面’，请改用边连接或调整装配位置")
    return chosen

def _coplanar_edge_contact(frame_a, frame_b, scene, settings):
    normal_dot = abs(frame_a["n"].dot(frame_b["n"]))
    if normal_dot < math.cos(math.radians(settings.joint_angle_tolerance_deg)):
        raise RuntimeError("边连接要求 A/B 板面共面或近似共面")
    center_b = _project_point_to_plane(frame_b["centroid"], frame_b["n"], frame_b["center_d"])
    plane_gap = abs(center_b.dot(frame_a["n"]) - frame_a["center_d"])
    tol = mm_to_bu(scene, settings.joint_detect_tolerance_mm)
    if plane_gap > tol:
        raise RuntimeError(f"A/B 中心板面不共面，距离约 {bu_to_mm(scene, plane_gap):.3f} mm")

    best = None
    pa = frame_a["outer_world"]
    pb = frame_b["outer_world"]
    for i, a0 in enumerate(pa):
        a1 = pa[(i + 1) % len(pa)]
        da = a1 - a0
        if da.length <= 1.0e-10:
            continue
        d = da.normalized()
        for j, b0 in enumerate(pb):
            b1 = pb[(j + 1) % len(pb)]
            db = b1 - b0
            if db.length <= 1.0e-10:
                continue
            if abs(d.dot(db.normalized())) < math.cos(math.radians(settings.joint_angle_tolerance_deg)):
                continue
            at = sorted((a0.dot(d), a1.dot(d)))
            bt = sorted((b0.dot(d), b1.dot(d)))
            lo, hi = max(at[0], bt[0]), min(at[1], bt[1])
            if hi - lo <= 1.0e-8:
                continue
            amid = a0 + d * (((lo + hi) * 0.5) - a0.dot(d))
            bmid = b0 + d * (((lo + hi) * 0.5) - b0.dot(d))
            lateral = bmid - amid
            lateral -= d * lateral.dot(d)
            gap = lateral.length
            if best is None or gap < best["gap"]:
                baseline_mid = (amid + bmid) * 0.5
                start = baseline_mid + d * (lo - baseline_mid.dot(d))
                end = baseline_mid + d * (hi - baseline_mid.dot(d))
                toward_b = frame_b["centroid"] - frame_a["centroid"]
                toward_b -= d * toward_b.dot(d)
                toward_b -= frame_a["n"] * toward_b.dot(frame_a["n"])
                if toward_b.length <= 1.0e-10:
                    toward_b = lateral
                if toward_b.length <= 1.0e-10:
                    toward_b = frame_a["n"].cross(d)
                best = {
                    "gap": gap,
                    "start": start,
                    "end": end,
                    "direction": d if (end - start).dot(d) >= 0 else -d,
                    "out_a": toward_b.normalized(),
                }
    if best is None or best["gap"] > tol:
        gap_mm = bu_to_mm(scene, best["gap"]) if best else None
        suffix = f"（最近边距 {gap_mm:.3f} mm）" if gap_mm is not None else ""
        raise RuntimeError(f"找不到 A/B 共面的贴合边{suffix}")
    return best


def _tooth_intervals(length, params, scene):
    mode = params.get("tooth_mode", "AVERAGE")
    if length <= 1.0e-10:
        return []
    if mode == "AVERAGE":
        if params.get("average_input", "COUNT") == "WIDTH":
            wanted = mm_to_bu(scene, max(0.001, float(params.get("tooth_width_mm", 20.0))))
            n = max(1, int(round(max(1.0, (length / wanted - 1.0) * 0.5))))
        else:
            n = max(1, int(params.get("tooth_count", 4)))
        segment = length / (2 * n + 1)
        return [(segment * (2 * i + 1), segment * (2 * i + 2)) for i in range(n)]

    if mode == "POINT":
        n = max(1, int(params.get("tooth_count", 2)))
        width = mm_to_bu(scene, max(0.001, float(params.get("tooth_width_mm", 50.0))))
        free = length - n * width
        gap = free / (n + 1) if free > 0.0 else -1.0
        if gap <= 0.0 or width > gap + 1.0e-9:
            max_width = length / (2 * n + 1)
            raise RuntimeError(
                f"点齿宽度不能超过均匀间隙；当前最多约 {bu_to_mm(scene, max_width):.2f} mm"
            )
        result = []
        cursor = gap
        for _i in range(n):
            result.append((cursor, cursor + width))
            cursor += width + gap
        return result

    cursor = 0.0
    result = []
    for marker in params.get("markers", []):
        cursor += mm_to_bu(scene, max(0.0, float(marker.get("gap_mm", 0.0))))
        width = mm_to_bu(scene, max(0.001, float(marker.get("width_mm", 1.0))))
        if cursor + width > length + 1.0e-8:
            raise RuntimeError("标记齿超出贴合边长度，请缩短间距/宽度")
        result.append((cursor, cursor + width))
        cursor += width
    if not result:
        raise RuntimeError("标记齿模式至少需要一个标记")
    return result


def _joint_helper_collection(scene):
    collection = bpy.data.collections.get(JOINT_HELPER_COLLECTION)
    if collection is None:
        collection = bpy.data.collections.new(JOINT_HELPER_COLLECTION)
        scene.collection.children.link(collection)
    return collection


def _make_helper_mesh(scene, joint_id, target, operation, name, verts, faces):
    collection = _joint_helper_collection(scene)
    mesh = bpy.data.meshes.new(name + "_MESH")
    mesh.from_pydata([tuple(v) for v in verts], [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    obj[JOINT_HELPER_TAG] = True
    obj[JOINT_ID_TAG] = joint_id
    obj[JOINT_TARGET_TAG] = target.name
    obj[JOINT_OPERATION_TAG] = operation
    obj.display_type = "WIRE"
    obj.hide_render = True
    obj.hide_select = True
    try:
        obj.hide_set(True)
    except Exception:
        pass
    return obj


def _make_box_helper(scene, joint_id, target, operation, name, center, ax, ay, az, dims):
    ax, ay, az = ax.normalized(), ay.normalized(), az.normalized()
    hx, hy, hz = dims[0] * 0.5, dims[1] * 0.5, dims[2] * 0.5
    verts = []
    for sx, sy, sz in ((-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)):
        verts.append(center + ax * (sx * hx) + ay * (sy * hy) + az * (sz * hz))
    faces = [(0,1,2,3),(4,7,6,5),(0,4,5,1),(1,5,6,2),(2,6,7,3),(4,0,3,7)]
    return _make_helper_mesh(scene, joint_id, target, operation, name, verts, faces)


def _make_prism_helper(scene, joint_id, target, operation, name, polygon, normal, thickness):
    n = normal.normalized() * (thickness * 0.5)
    verts = [p - n for p in polygon] + [p + n for p in polygon]
    count = len(polygon)
    faces = [tuple(range(count - 1, -1, -1)), tuple(range(count, count * 2))]
    for i in range(count):
        j = (i + 1) % count
        faces.append((i, j, count + j, count + i))
    return _make_helper_mesh(scene, joint_id, target, operation, name, verts, faces)


def _add_joint_boolean(target, helper, joint_id, operation, label):
    mod = target.modifiers.new(name=f"{JOINT_MOD_PREFIX}{joint_id[:8]}_{label}", type="BOOLEAN")
    mod.operation = operation
    if hasattr(mod, "solver"):
        mod.solver = "EXACT"
    mod.object = helper
    try:
        mod[JOINT_ID_TAG] = joint_id
    except Exception:
        pass
    return mod


def clear_joint_geometry(joint_id):
    removed_mods = 0
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        for mod in list(obj.modifiers):
            tagged = False
            try:
                tagged = str(mod.get(JOINT_ID_TAG, "")) == joint_id
            except Exception:
                tagged = False
            if tagged or (mod.name.startswith(JOINT_MOD_PREFIX + joint_id[:8] + "_")):
                obj.modifiers.remove(mod)
                removed_mods += 1
    removed_helpers = 0
    for obj in list(bpy.data.objects):
        if obj.get(JOINT_HELPER_TAG, False) and str(obj.get(JOINT_ID_TAG, "")) == joint_id:
            bpy.data.objects.remove(obj, do_unlink=True)
            removed_helpers += 1
    return removed_mods, removed_helpers


def _joint_epsilon(scene):
    return max(mm_to_bu(scene, 0.02), 1.0e-7)


def _generate_edge_insert(scene, settings, record, params, a, b, fa, fb):
    tooth_obj, slot_obj, tooth_frame, slot_frame, contact = _resolve_edge_insert_roles(
        a, b, fa, fb, scene, settings
    )
    # Persist resolved roles so later regeneration no longer depends on which
    # object happened to be active when the record was created.
    record.object_a = tooth_obj.name
    record.object_b = slot_obj.name
    record.name = f"EDGE_INSERT:{tooth_obj.name}→{slot_obj.name}"

    direction = contact["direction"].normalized()
    start, end = contact["start"], contact["end"]
    length = (end - start).length
    intervals = _tooth_intervals(length, params, scene)
    tol = mm_to_bu(scene, float(params.get("tolerance_mm", 0.2)))
    width_only = bool(params.get("tolerance_width_only", False))
    eps = _joint_epsilon(scene)
    created = 0

    for i, (s0, s1) in enumerate(intervals, 1):
        w = s1 - s0
        center_line = start + direction * ((s0 + s1) * 0.5)
        into_slot = contact["into_b"].normalized()
        center_tooth = _project_point_to_plane(center_line, tooth_frame["n"], tooth_frame["center_d"])
        tab_depth = slot_frame["thickness"]
        tab_center = center_tooth + into_slot * (tab_depth * 0.5 - eps * 0.5)
        tab = _make_box_helper(
            scene, record.joint_id, tooth_obj, "UNION", f"FFJ_{record.joint_id[:8]}_TAB_{i:02d}",
            tab_center, direction, tooth_frame["n"], into_slot,
            (w, tooth_frame["thickness"], tab_depth + eps),
        )
        _add_joint_boolean(tooth_obj, tab, record.joint_id, "UNION", f"TAB_{i:02d}")

        slot_center = _project_point_to_plane(center_tooth, slot_frame["n"], slot_frame["center_d"])
        mate_thickness = tooth_frame["thickness"] + (0.0 if width_only else tol * 2.0)
        slot = _make_box_helper(
            scene, record.joint_id, slot_obj, "DIFFERENCE", f"FFJ_{record.joint_id[:8]}_SLOT_{i:02d}",
            slot_center, direction, tooth_frame["n"], into_slot,
            (w + tol * 2.0, mate_thickness, slot_frame["thickness"] + eps * 4.0),
        )
        _add_joint_boolean(slot_obj, slot, record.joint_id, "DIFFERENCE", f"SLOT_{i:02d}")
        created += 2
    return created, f"边插 {len(intervals)} 齿 / 自动：{tooth_obj.name} 插齿 → {slot_obj.name} 插槽 / 贴合 {bu_to_mm(scene, length):.2f} mm"

def _find_containing_interval(intervals, lo, hi, eps=1.0e-8):
    for a, b in intervals:
        if a <= lo + eps and b >= hi - eps:
            return (a, b)
    return None


def _generate_cross(scene, settings, record, params, a, b, fa, fb):
    common = _common_line_segments(fa, fb)
    if common is None:
        raise RuntimeError("十字插：两块板材中心面没有有效重叠区间")
    d = common["direction"].normalized()
    sin_angle = fa["n"].cross(fb["n"]).length
    if sin_angle <= math.sin(math.radians(2.0)):
        raise RuntimeError("十字插要求两块板材明显交叉，不能近似平行")

    tol = mm_to_bu(scene, float(params.get("tolerance_mm", 0.2)))
    width_only = bool(params.get("tolerance_width_only", False))
    reverse = bool(params.get("cross_reverse", False))
    eps = _joint_epsilon(scene)
    created = 0

    q_a = fa["n"].cross(d).normalized()
    q_b = fb["n"].cross(d).normalized()
    strip_a = fb["thickness"] / sin_angle + (0.0 if width_only else tol * 2.0)
    strip_b = fa["thickness"] / sin_angle + (0.0 if width_only else tol * 2.0)

    for idx, (lo, hi) in enumerate(common["overlaps"], 1):
        if hi - lo <= eps:
            continue
        mid = (lo + hi) * 0.5
        halves = [(lo, mid), (mid, hi)]
        # Exactly two equal halves: one removed from A, one from B.  Reverse
        # swaps ownership; there is intentionally no user segment count.
        for half_index, (t0, t1) in enumerate(halves):
            cut_a = (half_index == 0) ^ reverse
            center = common["origin"] + d * ((t0 + t1) * 0.5)
            half_len = max(eps, t1 - t0)
            if cut_a:
                helper = _make_box_helper(
                    scene, record.joint_id, a, "DIFFERENCE", f"FFJ_{record.joint_id[:8]}_CROSS_A_{idx:02d}_{half_index}",
                    center, d, q_a, fa["n"],
                    (half_len + tol * 2.0, strip_a, fa["thickness"] + eps * 4.0),
                )
                _add_joint_boolean(a, helper, record.joint_id, "DIFFERENCE", f"CROSS_A_{idx:02d}_{half_index}")
            else:
                helper = _make_box_helper(
                    scene, record.joint_id, b, "DIFFERENCE", f"FFJ_{record.joint_id[:8]}_CROSS_B_{idx:02d}_{half_index}",
                    center, d, q_b, fb["n"],
                    (half_len + tol * 2.0, strip_b, fb["thickness"] + eps * 4.0),
                )
                _add_joint_boolean(b, helper, record.joint_id, "DIFFERENCE", f"CROSS_B_{idx:02d}_{half_index}")
            created += 1

        # If one plate's valid line interval is contained inside the other's,
        # it needs an entry slit through the containing plate.  Choose a side
        # from reverse, falling back to the only side with available material.
        sa = _find_containing_interval(common["segments_a"], lo, hi)
        sb = _find_containing_interval(common["segments_b"], lo, hi)
        if sa and sb:
            len_a, len_b = sa[1] - sa[0], sb[1] - sb[0]
            mover = receiver = None
            mover_frame = receiver_frame = None
            receiver_seg = None
            strip = None
            q_receiver = None
            # Entry relief is only valid when the common overlap is the *full*
            # interval of the moving plate and is contained inside the other
            # plate.  This prevents a concave outline/hole from accidentally
            # opening a long slit across unrelated overlap intervals.
            a_is_inner = abs(sa[0] - lo) <= eps and abs(sa[1] - hi) <= eps and len_a + eps < len_b
            b_is_inner = abs(sb[0] - lo) <= eps and abs(sb[1] - hi) <= eps and len_b + eps < len_a
            if a_is_inner:
                mover, receiver = a, b
                mover_frame, receiver_frame = fa, fb
                receiver_seg, strip, q_receiver = sb, strip_b, q_b
            elif b_is_inner:
                mover, receiver = b, a
                mover_frame, receiver_frame = fb, fa
                receiver_seg, strip, q_receiver = sa, strip_a, q_a

            if mover is not None:
                left = (receiver_seg[0], lo)
                right = (hi, receiver_seg[1])
                preferred = right if reverse else left
                fallback = left if reverse else right
                r0, r1 = preferred if preferred[1] - preferred[0] > eps else fallback
                if r1 - r0 > eps:
                    center = common["origin"] + d * ((r0 + r1) * 0.5)
                    helper = _make_box_helper(
                        scene, record.joint_id, receiver, "DIFFERENCE", f"FFJ_{record.joint_id[:8]}_ENTRY_{idx:02d}",
                        center, d, q_receiver, receiver_frame["n"],
                        ((r1 - r0) + tol * 2.0 + eps, strip, receiver_frame["thickness"] + eps * 4.0),
                    )
                    _add_joint_boolean(receiver, helper, record.joint_id, "DIFFERENCE", f"ENTRY_{idx:02d}")
                    created += 1

    angle = math.degrees(math.asin(max(0.0, min(1.0, sin_angle))))
    total = sum(hi - lo for lo, hi in common["overlaps"])
    return created, f"十字插 {len(common['overlaps'])} 个重叠区间 / 每区间 1/2 : 1/2 / 交角 {angle:.2f}° / 总重叠 {bu_to_mm(scene, total):.2f} mm"

def _dovetail_polygon(base_center, direction, outward, width, depth, ratio):
    d = direction.normalized()
    o = outward.normalized()
    base_half = width * 0.5
    tip_half = width * (1.0 + ratio) * 0.5
    return [
        base_center - d * base_half,
        base_center + d * base_half,
        base_center + o * depth + d * tip_half,
        base_center + o * depth - d * tip_half,
    ]


def _generate_edge_connect(scene, settings, record, params, a, b, fa, fb):
    contact = _coplanar_edge_contact(fa, fb, scene, settings)
    d = contact["direction"].normalized()
    start, end = contact["start"], contact["end"]
    length = (end - start).length
    n = max(1, int(params.get("tooth_count", 4)))
    segment = length / (2 * n + 1)
    depth = mm_to_bu(scene, max(0.001, float(params.get("edge_depth_mm", 10.0))))
    tol = mm_to_bu(scene, float(params.get("tolerance_mm", 0.2)))
    width_only = bool(params.get("tolerance_width_only", False))
    wedge = bool(params.get("edge_wedge", False))
    ratio = max(0.0, min(0.45, float(params.get("edge_wedge_ratio", 0.20)))) if wedge else 0.0
    eps = _joint_epsilon(scene)
    out_a = contact["out_a"].normalized()
    created = 0

    for i in range(2 * n + 1):
        s0, s1 = segment * i, segment * (i + 1)
        center = start + d * ((s0 + s1) * 0.5)
        width = s1 - s0
        owner_a = (i % 2 == 0)
        if owner_a:
            owner, receiver = a, b
            fo, fr = fa, fb
            outward = out_a
            label = f"EDGE_A_{i:02d}"
        else:
            owner, receiver = b, a
            fo, fr = fb, fa
            outward = -out_a
            label = f"EDGE_B_{i:02d}"

        # Tab starts a tiny amount inside its owner so UNION has real volume
        # overlap rather than only a coplanar touching face.
        tab_base = center - outward * eps
        tab_poly = _dovetail_polygon(tab_base, d, outward, width, depth + eps * 2.0, ratio)
        tab_poly = [_project_point_to_plane(p, fo["n"], fo["center_d"]) for p in tab_poly]
        tab = _make_prism_helper(
            scene, record.joint_id, owner, "UNION", f"FFJ_{record.joint_id[:8]}_{label}_TAB",
            tab_poly, fo["n"], fo["thickness"] + eps * 2.0,
        )
        _add_joint_boolean(owner, tab, record.joint_id, "UNION", label + "_TAB")

        # The cutter deliberately crosses the seam and enters the receiver.
        # This fixes the previous failure where B could merely touch the cutter
        # at a coplanar boundary and therefore remain uncut with Exact Boolean.
        cutter_width = width + tol * 2.0
        cutter_depth = depth + (0.0 if width_only else tol) + eps * 4.0
        cutter_base = center - outward * (tol + eps * 2.0)
        cut_poly = _dovetail_polygon(cutter_base, d, outward, cutter_width, cutter_depth, ratio)
        cut_poly = [_project_point_to_plane(p, fr["n"], fr["center_d"]) for p in cut_poly]
        cutter = _make_prism_helper(
            scene, record.joint_id, receiver, "DIFFERENCE", f"FFJ_{record.joint_id[:8]}_{label}_SLOT",
            cut_poly, fr["n"], fr["thickness"] + eps * 4.0,
        )
        _add_joint_boolean(receiver, cutter, record.joint_id, "DIFFERENCE", label + "_SLOT")
        created += 2

    return created, f"边连接 {2*n+1} 段锯齿 / A、B 双向齿槽 / 深度 {bu_to_mm(scene, depth):.2f} mm" + (" / 楔形" if wedge else "")

def _through_role_score(frame, common):
    """Heuristic: a local/narrow plate is usually the penetrating member."""
    total_support = sum(b - a for a, b in _merge_intervals(frame.get("_through_segments", [])))
    return (total_support, frame.get("area_2d", 0.0))


def _generate_through(scene, settings, record, params, a, b, fa, fb):
    common = _common_line_segments(fa, fb)
    if common is None:
        raise RuntimeError("贯穿：两块板材中心面没有有效重叠区间")
    d = common["direction"].normalized()
    sin_angle = fa["n"].cross(fb["n"]).length
    if sin_angle <= math.sin(math.radians(2.0)):
        raise RuntimeError("贯穿要求两块板材非平行")

    # Static geometry can be symmetric, so use the plate with the shorter line
    # support (then smaller face) as the default penetrator.  A dedicated
    # reverse switch remains available for intentionally opposite assembly.
    fa["_through_segments"] = common["segments_a"]
    fb["_through_segments"] = common["segments_b"]
    score_a = _through_role_score(fa, common)
    score_b = _through_role_score(fb, common)
    penetrator, receiver, fp, fr = (a, b, fa, fb) if score_a <= score_b else (b, a, fb, fa)
    if bool(params.get("through_reverse", False)):
        penetrator, receiver, fp, fr = receiver, penetrator, fr, fp

    # Recompute clipping in resolved role order, because the stored segment
    # lists and projection directions now need to match penetrator/receiver.
    common = _common_line_segments(fp, fr)
    if common is None:
        raise RuntimeError("贯穿：自动角色判定后没有有效投影区间")
    d = common["direction"].normalized()
    sin_angle = fp["n"].cross(fr["n"]).length

    tol = mm_to_bu(scene, float(params.get("tolerance_mm", 0.2)))
    width_only = bool(params.get("tolerance_width_only", False))
    eps = _joint_epsilon(scene)
    q_receiver = fr["n"].cross(d).normalized()
    projected_thickness = fp["thickness"] / sin_angle + (0.0 if width_only else tol * 2.0)
    created = 0

    for i, (lo, hi) in enumerate(common["overlaps"], 1):
        length = hi - lo
        if length <= eps:
            continue
        center = common["origin"] + d * ((lo + hi) * 0.5)
        helper = _make_box_helper(
            scene, record.joint_id, receiver, "DIFFERENCE", f"FFJ_{record.joint_id[:8]}_THROUGH_{i:02d}",
            center, d, q_receiver, fr["n"],
            (length + tol * 2.0, projected_thickness, fr["thickness"] + eps * 4.0),
        )
        _add_joint_boolean(receiver, helper, record.joint_id, "DIFFERENCE", f"THROUGH_{i:02d}")
        created += 1

    if not created:
        raise RuntimeError("贯穿：polygon clipping 后没有可生成的有效槽")
    record.object_a = penetrator.name
    record.object_b = receiver.name
    record.name = f"THROUGH:{penetrator.name}→{receiver.name}"
    total = sum(hi - lo for lo, hi in common["overlaps"])
    return created, f"贯穿 {len(common['overlaps'])} 个独立槽 / 自动：{penetrator.name} → {receiver.name} / 总长 {bu_to_mm(scene, total):.2f} mm / 槽宽 {bu_to_mm(scene, projected_thickness):.2f} mm"

def joint_params_from_settings(settings):
    return {
        "tolerance_mm": settings.joint_tolerance_mm,
        "tolerance_width_only": settings.joint_tolerance_width_only,
        "tooth_mode": settings.joint_tooth_mode,
        "average_input": settings.joint_average_input,
        "tooth_count": settings.joint_tooth_count,
        "tooth_width_mm": settings.joint_tooth_width_mm,
        "cross_reverse": settings.joint_cross_reverse,
        "through_reverse": settings.joint_through_reverse,
        "edge_depth_mm": settings.joint_edge_depth_mm,
        "edge_wedge": settings.joint_edge_wedge,
        "edge_wedge_ratio": settings.joint_edge_wedge_ratio,
        "markers": [
            {"gap_mm": item.gap_mm, "width_mm": item.width_mm}
            for item in settings.joint_markers
        ],
    }

def generate_joint_record(context, record):
    scene = context.scene
    settings = scene.flatfab_settings
    a = bpy.data.objects.get(record.object_a)
    b = bpy.data.objects.get(record.object_b)
    if a is None or b is None:
        raise RuntimeError("连接记录中的 A/B 物体已不存在")
    if a.type != "MESH" or b.type != "MESH":
        raise RuntimeError("A/B 必须都是 Mesh")

    clear_joint_geometry(record.joint_id)
    params = json.loads(record.params_json or "{}")
    fa = joint_plate_frame(a, scene, settings)
    fb = joint_plate_frame(b, scene, settings)

    if record.joint_type == "EDGE_INSERT":
        count, detail = _generate_edge_insert(scene, settings, record, params, a, b, fa, fb)
    elif record.joint_type == "CROSS":
        count, detail = _generate_cross(scene, settings, record, params, a, b, fa, fb)
    elif record.joint_type == "EDGE_CONNECT":
        count, detail = _generate_edge_connect(scene, settings, record, params, a, b, fa, fb)
    elif record.joint_type == "THROUGH":
        count, detail = _generate_through(scene, settings, record, params, a, b, fa, fb)
    else:
        raise RuntimeError("未知连接类型")

    record.status = detail
    return count, detail


def _find_joint_record(scene, joint_id):
    for item in scene.flatfab_joints:
        if item.joint_id == joint_id:
            return item
    return None


def _joint_record_index(scene, joint_id):
    for index, item in enumerate(scene.flatfab_joints):
        if item.joint_id == joint_id:
            return index
    return -1


def _square_row_width_from_sizes(sizes, gap):
    if not sizes:
        return 0.0
    max_w = max(w for w, h in sizes)
    area = sum(max(0.0, w) * max(0.0, h) for w, h in sizes)
    padded = area + gap * sum(w + h for w, h in sizes) + gap * gap * len(sizes)
    return max(max_w, math.sqrt(max(padded, 0.0)))


def write_layout_uv_world_object(helper_obj, source_obj, flat_obj, rows, scene, page):
    """Give UNION helper geometry the same atlas projection as its source plate."""
    if helper_obj.type != "MESH" or not helper_obj.data.loops:
        return
    mesh = helper_obj.data
    uv_layer = mesh.uv_layers.get(UV_MAP_NAME) or mesh.uv_layers.new(name=UV_MAP_NAME)
    modern_uv = getattr(uv_layer, "uv", None)
    legacy_uv = getattr(uv_layer, "data", None)
    inv_source = source_obj.matrix_world.inverted_safe()
    width, height = page["width"], page["height"]
    ox, oy = page["offset_x"], page["offset_y"]
    for loop in mesh.loops:
        world_co = helper_obj.matrix_world @ mesh.vertices[loop.vertex_index].co
        local_co = inv_source @ world_co
        flat_local = project_source_local_to_flat(rows, local_co)
        world_flat = flat_obj.matrix_world @ flat_local
        u = (bu_to_mm(scene, world_flat.x) + ox) / width
        v = (bu_to_mm(scene, world_flat.y) + oy) / height
        if modern_uv is not None:
            modern_uv[loop.index].vector = (u, v)
        elif legacy_uv is not None:
            legacy_uv[loop.index].uv = (u, v)
    mesh.update()


def update_union_helper_uvs(source_obj, flat_obj, rows, scene, page):
    count = 0
    for helper in bpy.data.objects:
        if not helper.get(JOINT_HELPER_TAG, False):
            continue
        if str(helper.get(JOINT_TARGET_TAG, "")) != source_obj.name:
            continue
        if str(helper.get(JOINT_OPERATION_TAG, "")) != "UNION":
            continue
        write_layout_uv_world_object(helper, source_obj, flat_obj, rows, scene, page)
        count += 1
    return count


# -----------------------------------------------------------------------------
# Layout helpers
# -----------------------------------------------------------------------------


def get_or_create_collection(scene, name):
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
        scene.collection.children.link(collection)
    return collection


def generated_objects(collection):
    return [
        obj
        for obj in collection.objects
        if obj.get(GENERATED_TAG, False) and not obj.get(SHEET_TAG, False)
    ]


def clear_generated(collection):
    objects = list(generated_objects(collection))
    for obj in objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    return len(objects)


def remove_sheet_preview(collection):
    previews = [obj for obj in collection.objects if obj.get(SHEET_TAG, False)]
    for obj in previews:
        bpy.data.objects.remove(obj, do_unlink=True)
    return len(previews)


def object_nominal_size_bu(scene, obj):
    width_mm = float(obj.get(WIDTH_TAG, bu_to_mm(scene, obj.dimensions.x)))
    height_mm = float(obj.get(HEIGHT_TAG, bu_to_mm(scene, obj.dimensions.y)))
    return mm_to_bu(scene, width_mm), mm_to_bu(scene, height_mm)


def object_world_xy_bounds_bu(obj):
    """Current evaluated transform bounds in world XY, in Blender units."""
    if obj.type != "MESH" or not obj.data.vertices:
        x, y = obj.matrix_world.translation.x, obj.matrix_world.translation.y
        return (x, y, x, y)
    mw = obj.matrix_world
    pts = [mw @ v.co for v in obj.data.vertices]
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def combined_world_xy_bounds_bu(objects):
    bounds = [object_world_xy_bounds_bu(obj) for obj in objects]
    return (
        min(b[0] for b in bounds),
        min(b[1] for b in bounds),
        max(b[2] for b in bounds),
        max(b[3] for b in bounds),
    )


def normalize_lock_name(value):
    value = str(value or "").strip()
    if not value:
        return ""
    if not value.lower().startswith("lock"):
        value = "lock" + value
    return value.casefold()


def detected_lock_group(obj, use_name_groups=True):
    """Custom property wins; otherwise recognize tokens such as _lock1."""
    manual = normalize_lock_name(obj.get(LOCK_TAG, ""))
    if manual:
        return manual
    if use_name_groups:
        match = LOCK_NAME_RE.search(obj.name)
        if match:
            return normalize_lock_name(match.group(1))
        source = str(obj.get(SOURCE_TAG, ""))
        match = LOCK_NAME_RE.search(source)
        if match:
            return normalize_lock_name(match.group(1))
    return ""


def build_layout_units(objects, use_lock_groups=True, use_name_groups=True):
    """Build movable packing units. Locked members retain relative transforms."""
    groups = {}
    order = []
    for obj in objects:
        lock = detected_lock_group(obj, use_name_groups) if use_lock_groups else ""
        key = ("LOCK", lock) if lock else ("OBJECT", obj.name, obj.as_pointer())
        if key not in groups:
            groups[key] = {"key": key, "lock": lock, "members": []}
            order.append(key)
        groups[key]["members"].append(obj)

    units = []
    for key in order:
        unit = groups[key]
        bounds = combined_world_xy_bounds_bu(unit["members"])
        unit["bounds"] = bounds
        unit["width"] = bounds[2] - bounds[0]
        unit["height"] = bounds[3] - bounds[1]
        unit["name"] = unit["lock"] or unit["members"][0].name
        units.append(unit)
    return units


def sort_layout_units(units, sort_mode):
    if sort_mode == "NAME":
        return sorted(units, key=lambda unit: unit["name"].casefold())
    if sort_mode == "AREA_DESC":
        return sorted(units, key=lambda unit: unit["width"] * unit["height"], reverse=True)
    if sort_mode == "LONG_DESC":
        return sorted(units, key=lambda unit: max(unit["width"], unit["height"]), reverse=True)
    return list(units)


def translate_object_world_xy(obj, dx, dy):
    mw = obj.matrix_world.copy()
    mw.translation += Vector((dx, dy, 0.0))
    obj.matrix_world = mw


def translate_unit(unit, dx, dy):
    for obj in unit["members"]:
        translate_object_world_xy(obj, dx, dy)
    b = unit["bounds"]
    unit["bounds"] = (b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy)


def arrange_preserve_orientation(scene, settings, objects):
    """Pack by translation only. Rotation, mirror, scale and lock-group offsets stay untouched."""
    units = build_layout_units(
        objects,
        use_lock_groups=settings.use_lock_groups,
        use_name_groups=settings.use_name_lock_groups,
    )
    units = sort_layout_units(units, settings.sort_mode)
    gap = mm_to_bu(scene, settings.gap_mm)

    if settings.use_sheet_size:
        sheet_w = mm_to_bu(scene, settings.sheet_width_mm)
        sheet_h = mm_to_bu(scene, settings.sheet_height_mm)
        margin = mm_to_bu(scene, settings.sheet_margin_mm)
        start_x = margin
        start_y = margin
        max_x = max(start_x, sheet_w - margin)
        max_y = max(start_y, sheet_h - margin)
        row_limit = max_x
    else:
        start_x = 0.0
        start_y = 0.0
        max_y = float("inf")
        row_limit = (
            mm_to_bu(scene, settings.row_width_mm)
            if settings.row_width_mm > 0.0
            else 0.0
        )

    if settings.square_packing:
        target_width = _square_row_width_from_sizes(
            [(unit["width"], unit["height"]) for unit in units], gap
        )
        if target_width > 0.0:
            row_limit = start_x + target_width
            if settings.use_sheet_size:
                row_limit = min(row_limit, max_x)

    cursor_x = start_x
    cursor_y = start_y
    row_height = 0.0
    overflow = []

    for unit in units:
        item_w = unit["width"]
        item_h = unit["height"]

        if row_limit > 0.0 and cursor_x > start_x and cursor_x + item_w > row_limit + 1.0e-9:
            cursor_x = start_x
            cursor_y += row_height + gap
            row_height = 0.0

        old_min_x, old_min_y = unit["bounds"][0], unit["bounds"][1]
        translate_unit(unit, cursor_x - old_min_x, cursor_y - old_min_y)

        if settings.use_sheet_size and (
            cursor_x + item_w > max_x + 1.0e-9
            or cursor_y + item_h > max_y + 1.0e-9
        ):
            overflow.append(unit["name"])

        cursor_x += item_w + gap
        row_height = max(row_height, item_h)

    locked_units = [unit for unit in units if unit["lock"]]
    locked_parts = sum(len(unit["members"]) for unit in locked_units)
    return {
        "units": units,
        "overflow": overflow,
        "group_count": len(locked_units),
        "locked_parts": locked_parts,
    }


def transform_objects_about_selection_center(objects, transform_2d):
    """Apply a world-space 2D transform around the selected set's bbox center."""
    if not objects:
        return
    b = combined_world_xy_bounds_bu(objects)
    cx = (b[0] + b[2]) * 0.5
    cy = (b[1] + b[3]) * 0.5
    to_origin = Matrix.Translation(Vector((-cx, -cy, 0.0)))
    back = Matrix.Translation(Vector((cx, cy, 0.0)))
    matrix = back @ transform_2d @ to_origin
    for obj in objects:
        obj.matrix_world = matrix @ obj.matrix_world


def mirror_objects_world(objects, axis):
    reflect = Matrix.Identity(4)
    if axis == "X":
        reflect[0][0] = -1.0
    else:
        reflect[1][1] = -1.0
    transform_objects_about_selection_center(objects, reflect)


def rotate_objects_world(objects, radians):
    rotate = Matrix.Rotation(radians, 4, "Z")
    transform_objects_about_selection_center(objects, rotate)


def sort_layout_objects(scene, objects, sort_mode):
    def size(obj):
        return object_nominal_size_bu(scene, obj)

    if sort_mode == "NAME":
        return sorted(objects, key=lambda obj: obj.name.casefold())
    if sort_mode == "AREA_DESC":
        return sorted(objects, key=lambda obj: size(obj)[0] * size(obj)[1], reverse=True)
    if sort_mode == "LONG_DESC":
        return sorted(objects, key=lambda obj: max(size(obj)), reverse=True)
    return list(objects)


def set_part_transform(obj, x, y, width, height, rotate_90=False):
    if rotate_90:
        # Local geometry begins at (0,0). +90° rotates it to x=[-h,0], y=[0,w].
        # Offset by +height so its lower-left bounds land on (x,y).
        obj.location = (x + height, y, 0.0)
        obj.rotation_euler = (0.0, 0.0, math.radians(90.0))
    else:
        obj.location = (x, y, 0.0)
        obj.rotation_euler = (0.0, 0.0, 0.0)
    obj.scale = (1.0, 1.0, 1.0)


def arrange_objects(scene, settings, objects):
    """Shelf packing using bounding rectangles, optional 90-degree rotation."""
    objects = sort_layout_objects(scene, objects, settings.sort_mode)
    gap = mm_to_bu(scene, settings.gap_mm)

    if settings.use_sheet_size:
        sheet_w = mm_to_bu(scene, settings.sheet_width_mm)
        sheet_h = mm_to_bu(scene, settings.sheet_height_mm)
        margin = mm_to_bu(scene, settings.sheet_margin_mm)
        start_x = margin
        start_y = margin
        max_x = max(start_x, sheet_w - margin)
        max_y = max(start_y, sheet_h - margin)
        row_limit = max_x
    else:
        start_x = 0.0
        start_y = 0.0
        max_y = float("inf")
        row_limit = (
            mm_to_bu(scene, settings.row_width_mm)
            if settings.row_width_mm > 0.0
            else 0.0
        )

    if settings.square_packing:
        target_width = _square_row_width_from_sizes(
            [object_nominal_size_bu(scene, obj) for obj in objects], gap
        )
        if target_width > 0.0:
            row_limit = start_x + target_width
            if settings.use_sheet_size:
                row_limit = min(row_limit, max_x)

    cursor_x = start_x
    cursor_y = start_y
    row_height = 0.0
    overflow = []
    rotated_count = 0

    for obj in objects:
        width, height = object_nominal_size_bu(scene, obj)
        rotate = False

        def fits_in_row(w):
            return row_limit <= 0.0 or cursor_x + w <= row_limit + 1.0e-9

        # If the part cannot fit in the remaining width but its 90° orientation can,
        # use rotation before deciding to start a new row.
        if settings.allow_rotate_90 and not fits_in_row(width) and fits_in_row(height):
            rotate = True

        item_w, item_h = (height, width) if rotate else (width, height)

        if row_limit > 0.0 and cursor_x > start_x and cursor_x + item_w > row_limit + 1.0e-9:
            cursor_x = start_x
            cursor_y += row_height + gap
            row_height = 0.0

            # Re-evaluate orientation in a fresh row. Prefer the orientation that fits.
            rotate = False
            if settings.allow_rotate_90:
                avail = row_limit - start_x
                normal_fits = width <= avail + 1.0e-9
                rotated_fits = height <= avail + 1.0e-9
                if not normal_fits and rotated_fits:
                    rotate = True
                elif normal_fits and rotated_fits and height < width:
                    # Prefer smaller packed width only when it also does not increase row height badly.
                    rotate = height < width and width <= max(height * 1.8, height + gap)

            item_w, item_h = (height, width) if rotate else (width, height)

        set_part_transform(obj, cursor_x, cursor_y, width, height, rotate)
        if rotate:
            rotated_count += 1

        if settings.use_sheet_size:
            if (
                cursor_x + item_w > max_x + 1.0e-9
                or cursor_y + item_h > max_y + 1.0e-9
            ):
                overflow.append(obj.name)

        cursor_x += item_w + gap
        row_height = max(row_height, item_h)

    return {
        "objects": objects,
        "overflow": overflow,
        "rotated_count": rotated_count,
    }


def enabled_bevel_objects(objects):
    result = []
    for obj in objects:
        if any(mod.type == "BEVEL" and mod.show_viewport for mod in obj.modifiers):
            result.append(obj.name)
    return result


def create_sheet_preview(scene, settings):
    collection = get_or_create_collection(
        scene, settings.output_collection.strip() or "FLAT_LAYOUT"
    )
    remove_sheet_preview(collection)

    if not settings.use_sheet_size:
        return None

    w = mm_to_bu(scene, settings.sheet_width_mm)
    h = mm_to_bu(scene, settings.sheet_height_mm)
    coords = [(0.0, 0.0, 0.0), (w, 0.0, 0.0), (w, h, 0.0), (0.0, h, 0.0)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    mesh = bpy.data.meshes.new(f"{SHEET_OBJECT_NAME}_MESH")
    mesh.from_pydata(coords, edges, [])
    mesh.update()
    obj = bpy.data.objects.new(SHEET_OBJECT_NAME, mesh)
    obj[SHEET_TAG] = True
    obj["flatfab_sheet_width_mm"] = settings.sheet_width_mm
    obj["flatfab_sheet_height_mm"] = settings.sheet_height_mm
    obj.display_type = "WIRE"
    obj.show_in_front = True
    collection.objects.link(obj)
    return obj


def world_xy_bounds_mm(scene, objects):
    points = []
    for obj in objects:
        if obj.type != "MESH":
            continue
        mw = obj.matrix_world
        for v in obj.data.vertices:
            p = mw @ v.co
            points.append((bu_to_mm(scene, p.x), bu_to_mm(scene, p.y)))
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


# -----------------------------------------------------------------------------
# Export geometry helpers
# -----------------------------------------------------------------------------


def edge_chains_from_mesh(obj, scene):
    """Return edge chains in world XY, millimeters. Closed chains repeat first point."""
    mesh = obj.data
    mw = obj.matrix_world
    coords = {
        i: (bu_to_mm(scene, (mw @ v.co).x), bu_to_mm(scene, (mw @ v.co).y))
        for i, v in enumerate(mesh.vertices)
    }

    adjacency = {i: [] for i in coords}
    unused = set()
    for edge in mesh.edges:
        a, b = edge.vertices
        key = (a, b) if a < b else (b, a)
        unused.add(key)
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)

    chains = []

    def take_edge(a, b):
        key = (a, b) if a < b else (b, a)
        if key in unused:
            unused.remove(key)
            return True
        return False

    # Open chains / branches first.
    starts = [i for i, nbs in adjacency.items() if len(nbs) != 2 and len(nbs) > 0]
    for start in starts:
        for nb in list(adjacency[start]):
            key = (start, nb) if start < nb else (nb, start)
            if key not in unused:
                continue
            chain = [start]
            prev = None
            cur = start
            nxt = nb
            while True:
                if not take_edge(cur, nxt):
                    break
                chain.append(nxt)
                prev, cur = cur, nxt
                candidates = [
                    x
                    for x in adjacency.get(cur, [])
                    if x != prev
                    and ((cur, x) if cur < x else (x, cur)) in unused
                ]
                if len(candidates) != 1:
                    break
                nxt = candidates[0]
            if len(chain) >= 2:
                chains.append([coords[i] for i in chain])

    # Remaining components should usually be closed loops.
    while unused:
        a, b = next(iter(unused))
        chain = [a]
        prev = None
        cur = a
        nxt = b
        while True:
            if not take_edge(cur, nxt):
                break
            chain.append(nxt)
            prev, cur = cur, nxt
            candidates = [
                x
                for x in adjacency.get(cur, [])
                if x != prev
                and ((cur, x) if cur < x else (x, cur)) in unused
            ]
            if not candidates:
                break
            nxt = candidates[0]
            if nxt == chain[0]:
                if take_edge(cur, nxt):
                    chain.append(nxt)
                break
        if len(chain) >= 2:
            chains.append([coords[i] for i in chain])

    return chains


def export_object_data(scene, objects):
    data = []
    for obj in objects:
        chains = edge_chains_from_mesh(obj, scene)
        if not chains:
            continue
        pts = [p for chain in chains for p in chain]
        min_x = min(p[0] for p in pts)
        min_y = min(p[1] for p in pts)
        max_x = max(p[0] for p in pts)
        max_y = max(p[1] for p in pts)
        data.append(
            {
                "name": obj.name,
                "source": str(obj.get(SOURCE_TAG, obj.name)),
                "chains": chains,
                "bbox": (min_x, min_y, max_x, max_y),
                "center": ((min_x + max_x) * 0.5, (min_y + max_y) * 0.5),
            }
        )
    return data


def compute_export_page(settings, object_data):
    pts = [p for d in object_data for chain in d["chains"] for p in chain]
    if not pts:
        raise RuntimeError("没有可导出的轮廓")

    min_x = min(p[0] for p in pts)
    min_y = min(p[1] for p in pts)
    max_x = max(p[0] for p in pts)
    max_y = max(p[1] for p in pts)
    margin = settings.export_margin_mm

    if settings.export_page_mode == "SHEET":
        # Manufacturing sheet mode must preserve the exact Blender XY placement.
        # This is critical after manual rotation/mirror/locked-group positioning.
        page_w = settings.sheet_width_mm
        page_h = settings.sheet_height_mm
        offset_x = 0.0
        offset_y = 0.0
        overflow = (
            min_x < -1.0e-6
            or min_y < -1.0e-6
            or max_x > page_w + 1.0e-6
            or max_y > page_h + 1.0e-6
        )
    else:
        page_w = (max_x - min_x) + margin * 2.0
        page_h = (max_y - min_y) + margin * 2.0
        offset_x = margin - min_x
        offset_y = margin - min_y
        overflow = False

    if page_w <= 0.0 or page_h <= 0.0:
        raise RuntimeError("导出页面尺寸无效")

    return {
        "width": page_w,
        "height": page_h,
        "offset_x": offset_x,
        "offset_y": offset_y,
        "overflow": overflow,
        "source_bbox": (min_x, min_y, max_x, max_y),
    }


def project_source_local_to_flat(rows, co):
    row_x, row_y = rows
    x = row_x[0] * co.x + row_x[1] * co.y + row_x[2] * co.z + row_x[3]
    y = row_y[0] * co.x + row_y[1] * co.y + row_y[2] * co.z + row_y[3]
    return Vector((x, y, 0.0))


def write_layout_uv(source_obj, flat_obj, rows, scene, page):
    """Write UVs on the original plate model so one full-page layout texture aligns exactly."""
    if source_obj.type != "MESH":
        raise RuntimeError("源物体不是 Mesh")

    # Linked duplicates need different UV positions on the atlas, so make the
    # processed object single-user before modifying its mesh UV data.
    if source_obj.data.users > 1:
        source_obj.data = source_obj.data.copy()

    mesh = source_obj.data
    if not mesh.loops:
        raise RuntimeError("源网格没有可写入 UV 的面")

    uv_layer = mesh.uv_layers.get(UV_MAP_NAME)
    if uv_layer is None:
        uv_layer = mesh.uv_layers.new(name=UV_MAP_NAME)

    try:
        mesh.uv_layers.active = uv_layer
        mesh.uv_layers.active_render = uv_layer
    except Exception:
        pass

    width = page["width"]
    height = page["height"]
    ox = page["offset_x"]
    oy = page["offset_y"]
    min_u = float("inf")
    min_v = float("inf")
    max_u = float("-inf")
    max_v = float("-inf")

    # Blender 4.2+ exposes UV coordinates as Float2 corner attributes. Keep a
    # legacy fallback so the same add-on remains usable on nearby versions.
    modern_uv = getattr(uv_layer, "uv", None)
    legacy_uv = getattr(uv_layer, "data", None)

    for loop in mesh.loops:
        co = mesh.vertices[loop.vertex_index].co
        flat_local = project_source_local_to_flat(rows, co)
        world = flat_obj.matrix_world @ flat_local
        u = (bu_to_mm(scene, world.x) + ox) / width
        v = (bu_to_mm(scene, world.y) + oy) / height

        if modern_uv is not None:
            modern_uv[loop.index].vector = (u, v)
        elif legacy_uv is not None:
            legacy_uv[loop.index].uv = (u, v)
        else:
            raise RuntimeError("当前 Blender 版本无法访问 UV 数据")

        min_u = min(min_u, u)
        min_v = min(min_v, v)
        max_u = max(max_u, u)
        max_v = max(max_v, v)

    mesh.update()
    return (min_u, min_v, max_u, max_v)


def get_projection_rows_for_uv(flat_obj, source_obj, depsgraph, scene, settings):
    rows = projection_rows_from_flat(flat_obj)
    if rows is not None:
        return rows, False

    # Compatibility with layouts created by FlatFab 1.4.x: rebuild the same
    # affine projection once, then store it on the flat outline for future use.
    projection = build_flat_projection_geometry(source_obj, depsgraph, scene, settings)
    store_projection_metadata(flat_obj, projection)
    return (projection["row_x"], projection["row_y"]), True


def transformed_chain(chain, page):
    ox, oy = page["offset_x"], page["offset_y"]
    return [(x + ox, y + oy) for x, y in chain]


def ensure_extension(filepath, fmt):
    ext = "." + fmt.lower()
    root, current = os.path.splitext(filepath)
    if current.lower() == ext:
        return filepath
    if current.lower() in {".svg", ".dxf", ".pdf"}:
        return root + ext
    return filepath + ext


def export_svg(filepath, settings, object_data, page):
    width = page["width"]
    height = page["height"]
    stroke = max(settings.export_line_width_mm, 0.001)
    lines = [
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.6f}mm" '
            f'height="{height:.6f}mm" viewBox="0 0 {width:.6f} {height:.6f}">'
        ),
        "  <title>FlatFab 1:1 Manufacturing Layout</title>",
        "  <desc>Coordinates use millimeters. Print/import at 100% / 1:1.</desc>",
    ]

    if settings.export_sheet_border:
        lines.append(
            f'  <rect id="sheet-border" x="0" y="0" width="{width:.6f}" height="{height:.6f}" '
            f'fill="none" stroke="black" stroke-width="{stroke:.6f}" />'
        )

    lines.append(
        f'  <g id="cut" fill="none" stroke="black" stroke-width="{stroke:.6f}" '
        'stroke-linecap="butt" stroke-linejoin="miter">'
    )
    for d in object_data:
        safe_id = html.escape(clean_name(d["name"]), quote=True)
        lines.append(f'    <g id="{safe_id}" data-source="{html.escape(d["source"], quote=True)}">')
        for chain in d["chains"]:
            tchain = transformed_chain(chain, page)
            if not tchain:
                continue
            cmd = [f"M {tchain[0][0]:.6f},{height - tchain[0][1]:.6f}"]
            for x, y in tchain[1:]:
                cmd.append(f"L {x:.6f},{height - y:.6f}")
            if len(tchain) > 2 and tchain[0] == tchain[-1]:
                cmd.append("Z")
            lines.append(f'      <path d="{" ".join(cmd)}" />')
        lines.append("    </g>")
    lines.append("  </g>")

    if settings.export_labels:
        font_size = settings.label_size_mm
        lines.append(
            f'  <g id="engrave-labels" fill="black" stroke="none" '
            f'font-family="sans-serif" font-size="{font_size:.6f}mm" text-anchor="middle">'
        )
        for d in object_data:
            cx = d["center"][0] + page["offset_x"]
            cy = height - (d["center"][1] + page["offset_y"])
            label = html.escape(d["source"])
            lines.append(f'    <text x="{cx:.6f}" y="{cy:.6f}">{label}</text>')
        lines.append("  </g>")

    lines.append("</svg>")
    with open(filepath, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))


def dxf_pair(code, value):
    return f"{code}\n{value}\n"


def export_dxf(filepath, settings, object_data, page):
    out = []
    add = out.append
    add(dxf_pair(0, "SECTION")); add(dxf_pair(2, "HEADER"))
    add(dxf_pair(9, "$ACADVER")); add(dxf_pair(1, "AC1015"))
    add(dxf_pair(9, "$INSUNITS")); add(dxf_pair(70, 4))  # millimeters
    add(dxf_pair(9, "$MEASUREMENT")); add(dxf_pair(70, 1))
    add(dxf_pair(0, "ENDSEC"))

    add(dxf_pair(0, "SECTION")); add(dxf_pair(2, "TABLES"))
    add(dxf_pair(0, "TABLE")); add(dxf_pair(2, "LAYER")); add(dxf_pair(70, 4))
    for layer_name, color in (("0", 7), ("CUT", 7), ("ENGRAVE", 3), ("SHEET", 8)):
        add(dxf_pair(0, "LAYER"))
        add(dxf_pair(100, "AcDbSymbolTableRecord"))
        add(dxf_pair(100, "AcDbLayerTableRecord"))
        add(dxf_pair(2, layer_name)); add(dxf_pair(70, 0)); add(dxf_pair(62, color)); add(dxf_pair(6, "CONTINUOUS"))
    add(dxf_pair(0, "ENDTAB")); add(dxf_pair(0, "ENDSEC"))

    add(dxf_pair(0, "SECTION")); add(dxf_pair(2, "ENTITIES"))

    def polyline(layer, chain, apply_page_transform=True):
        tchain = transformed_chain(chain, page) if apply_page_transform else list(chain)
        if not tchain:
            return
        closed = len(tchain) > 2 and tchain[0] == tchain[-1]
        pts = tchain[:-1] if closed else tchain
        add(dxf_pair(0, "LWPOLYLINE"))
        add(dxf_pair(100, "AcDbEntity")); add(dxf_pair(8, layer))
        add(dxf_pair(100, "AcDbPolyline")); add(dxf_pair(90, len(pts))); add(dxf_pair(70, 1 if closed else 0))
        for x, y in pts:
            add(dxf_pair(10, f"{x:.6f}")); add(dxf_pair(20, f"{y:.6f}"))

    for d in object_data:
        for chain in d["chains"]:
            polyline("CUT", chain)

    if settings.export_sheet_border:
        w, h = page["width"], page["height"]
        polyline("SHEET", [(0, 0), (w, 0), (w, h), (0, h), (0, 0)], apply_page_transform=False)

    if settings.export_labels:
        for d in object_data:
            cx = d["center"][0] + page["offset_x"]
            cy = d["center"][1] + page["offset_y"]
            # DXF R2000 TEXT codepages vary by reader. Keep UTF-8 source string;
            # modern readers usually preserve it, while laser software may ignore text.
            txt = d["source"].replace("\n", " ").replace("\r", " ")
            add(dxf_pair(0, "TEXT")); add(dxf_pair(100, "AcDbEntity")); add(dxf_pair(8, "ENGRAVE"))
            add(dxf_pair(100, "AcDbText")); add(dxf_pair(10, f"{cx:.6f}")); add(dxf_pair(20, f"{cy:.6f}")); add(dxf_pair(30, "0.0"))
            add(dxf_pair(40, f"{settings.label_size_mm:.6f}")); add(dxf_pair(1, txt))

    add(dxf_pair(0, "ENDSEC")); add(dxf_pair(0, "EOF"))
    with open(filepath, "w", encoding="utf-8", newline="\n") as f:
        f.write("".join(out))


def pdf_escape_text(text):
    # Built-in Helvetica cannot represent arbitrary Unicode without embedding a font.
    # Keep PDF self-contained and dependency-free by replacing unsupported characters.
    ascii_text = text.encode("latin-1", "replace").decode("latin-1")
    return ascii_text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_pdf_bytes(page_w_pt, page_h_pt, content_bytes):
    objects = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    objects.append(
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_w_pt:.6f} {page_h_pt:.6f}] "
            f"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ).encode("ascii")
    )
    objects.append(
        b"<< /Length " + str(len(content_bytes)).encode("ascii") + b" >>\nstream\n" + content_bytes + b"\nendstream"
    )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    data = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(data))
        data.extend(f"{idx} 0 obj\n".encode("ascii"))
        data.extend(obj)
        data.extend(b"\nendobj\n")

    xref = len(data)
    data.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    data.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        data.extend(f"{off:010d} 00000 n \n".encode("ascii"))
    data.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(data)


def export_pdf(filepath, settings, object_data, page):
    page_w_pt = mm_to_pt(page["width"])
    page_h_pt = mm_to_pt(page["height"])
    if page_w_pt > 14400.0 or page_h_pt > 14400.0:
        raise RuntimeError("PDF 页面超过传统 200 英寸 / 14400 pt 限制；请缩小板材页面或改用 SVG/DXF")

    content = []
    line_w = max(mm_to_pt(settings.export_line_width_mm), 0.01)
    content.append("q")
    content.append("0 G")
    content.append(f"{line_w:.6f} w")

    for d in object_data:
        for chain in d["chains"]:
            tchain = transformed_chain(chain, page)
            if len(tchain) < 2:
                continue
            x0, y0 = tchain[0]
            content.append(f"{mm_to_pt(x0):.6f} {mm_to_pt(y0):.6f} m")
            for x, y in tchain[1:]:
                content.append(f"{mm_to_pt(x):.6f} {mm_to_pt(y):.6f} l")
            if len(tchain) > 2 and tchain[0] == tchain[-1]:
                content.append("h")
            content.append("S")

    if settings.export_sheet_border:
        w, h = page_w_pt, page_h_pt
        content.append(f"0 0 m {w:.6f} 0 l {w:.6f} {h:.6f} l 0 {h:.6f} l h S")
    content.append("Q")

    if settings.export_labels:
        font_pt = mm_to_pt(settings.label_size_mm)
        for d in object_data:
            cx = mm_to_pt(d["center"][0] + page["offset_x"])
            cy = mm_to_pt(d["center"][1] + page["offset_y"])
            txt = pdf_escape_text(d["source"])
            content.append("BT")
            content.append(f"/F1 {font_pt:.6f} Tf")
            content.append(f"{cx:.6f} {cy:.6f} Td")
            content.append(f"({txt}) Tj")
            content.append("ET")

    content_bytes = ("\n".join(content) + "\n").encode("latin-1", "replace")
    pdf = build_pdf_bytes(page_w_pt, page_h_pt, content_bytes)
    with open(filepath, "wb") as f:
        f.write(pdf)


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------


class FLATFAB_OT_set_scene_unit(Operator):
    bl_idname = "flatfab.set_scene_unit"
    bl_label = "设置板材单位"
    bl_description = "快速把场景设置为公制毫米或厘米"
    bl_options = {"REGISTER", "UNDO"}

    unit: EnumProperty(
        items=(("MM", "毫米", "毫米"), ("CM", "厘米", "厘米")),
        default="MM",
    )

    def execute(self, context):
        units = context.scene.unit_settings
        units.system = "METRIC"
        if self.unit == "CM":
            units.scale_length = 0.01
            units.length_unit = "CENTIMETERS"
            label = "厘米 (cm)"
        else:
            units.scale_length = 0.001
            units.length_unit = "MILLIMETERS"
            label = "毫米 (mm)"
        context.scene.flatfab_last_result = f"场景单位已设置为 {label}。已有连接如需保持当前参数含义，请执行重算。"
        self.report({"INFO"}, context.scene.flatfab_last_result)
        return {"FINISHED"}


class FLATFAB_OT_apply_solidify(Operator):
    bl_idname = "flatfab.apply_solidify"
    bl_label = "添加 / 更新实体化"
    bl_description = "给所有选中板材添加 Solidify；已有 Solidify 时直接更新厚度和偏移"
    bl_options = {"REGISTER", "UNDO"}

    @staticmethod
    def _local_thickness(obj, context, desired_world):
        try:
            world_verts, face_data = polygon_world_data(obj.data, obj.matrix_world)
            settings = context.scene.flatfab_settings
            axis = axis_from_mode(obj, face_data, settings.projection_mode, settings.angle_tolerance)
            if axis is None:
                return desired_world
            local_n = (obj.matrix_world.to_3x3().transposed() @ axis).normalized()
            scale = (obj.matrix_world.to_3x3() @ local_n).length
            if scale > 1.0e-12:
                return desired_world / scale
        except Exception:
            pass
        return desired_world

    def execute(self, context):
        scene = context.scene
        settings = scene.flatfab_settings
        selected = [
            obj for obj in context.selected_objects
            if obj.type == "MESH"
            and not obj.get(GENERATED_TAG, False)
            and not obj.get(JOINT_HELPER_TAG, False)
        ]
        if not selected:
            self.report({"ERROR"}, "请先选择至少一个板材 Mesh")
            return {"CANCELLED"}

        desired_world = mm_to_bu(scene, settings.solidify_thickness_mm)
        offset = 1.0 if settings.solidify_offset == "POSITIVE" else -1.0
        added = updated = 0
        for obj in selected:
            mod = next((m for m in obj.modifiers if m.type == "SOLIDIFY"), None)
            if mod is None:
                mod = obj.modifiers.new(name="FlatFab Solidify", type="SOLIDIFY")
                added += 1
            else:
                updated += 1
            mod.thickness = self._local_thickness(obj, context, desired_world)
            mod.offset = offset
            if hasattr(mod, "use_even_offset"):
                mod.use_even_offset = True
            if hasattr(mod, "use_quality_normals"):
                mod.use_quality_normals = True
            mod.show_viewport = True
            mod.show_render = True

        sign = "+1" if offset > 0 else "-1"
        scene.flatfab_last_result = (
            f"实体化：{len(selected)} 件，厚度 {settings.solidify_thickness_mm:g} mm，偏移 {sign}；"
            f"新增 {added} / 更新 {updated}。"
        )
        self.report({"INFO"}, scene.flatfab_last_result)
        return {"FINISHED"}


class FLATFAB_PG_JointMarker(PropertyGroup):
    gap_mm: FloatProperty(
        name="间距 (mm)",
        description="从上一齿末端（第一个标记从边起点）到当前齿起点的距离",
        default=10.0,
        min=0.0,
        soft_max=500.0,
        precision=2,
    )
    width_mm: FloatProperty(
        name="宽度 (mm)",
        default=60.0,
        min=0.001,
        soft_max=500.0,
        precision=2,
    )


class FLATFAB_PG_JointRecord(PropertyGroup):
    joint_id: StringProperty(name="连接 ID")
    name: StringProperty(name="连接名称")
    joint_type: EnumProperty(
        name="连接类型",
        items=(
            ("EDGE_INSERT", "边插", "自动识别边插面：边侧生成插齿，面侧生成插槽"),
            ("CROSS", "十字插", "两板重叠后沿交线方向半齿镶嵌"),
            ("EDGE_CONNECT", "边连接", "两块共面板材的锯齿/楔形连接"),
            ("THROUGH", "贯穿", "自动判断贯穿片/接收板，并在接收板生成多区间投影槽"),
        ),
        default="EDGE_INSERT",
    )
    object_a: StringProperty(name="A")
    object_b: StringProperty(name="B")
    params_json: StringProperty(name="参数")
    status: StringProperty(name="状态")


class FLATFAB_OT_add_joint_marker(Operator):
    bl_idname = "flatfab.add_joint_marker"
    bl_label = "增加标记"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        item = context.scene.flatfab_settings.joint_markers.add()
        item.gap_mm = 10.0
        item.width_mm = 60.0
        return {"FINISHED"}


class FLATFAB_OT_remove_joint_marker(Operator):
    bl_idname = "flatfab.remove_joint_marker"
    bl_label = "删除标记"
    bl_options = {"REGISTER", "UNDO"}
    index: IntProperty(default=-1)

    def execute(self, context):
        markers = context.scene.flatfab_settings.joint_markers
        if 0 <= self.index < len(markers):
            markers.remove(self.index)
        return {"FINISHED"}


class FLATFAB_OT_create_joint(Operator):
    bl_idname = "flatfab.create_joint"
    bl_label = "生成参数化连接"
    bl_description = "使用两个选中板件创建连接；边插和贯穿会自动判定几何角色，不依赖活动对象"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        settings = scene.flatfab_settings
        selected = [
            o for o in context.selected_objects
            if o.type == "MESH"
            and not o.get(GENERATED_TAG, False)
            and not o.get(JOINT_HELPER_TAG, False)
        ]
        if len(selected) != 2:
            self.report({"ERROR"}, "请选择恰好两个原始板件 Mesh")
            return {"CANCELLED"}

        # Stable initial order only; role-aware generators can rewrite the
        # record after geometric detection.  Active-object order is irrelevant.
        a, b = sorted(selected, key=lambda obj: obj.name.casefold())
        record = scene.flatfab_joints.add()
        record.joint_id = uuid.uuid4().hex
        record.joint_type = settings.joint_type
        record.object_a = a.name
        record.object_b = b.name
        record.name = f"{settings.joint_type}:{a.name}↔{b.name}"
        record.params_json = json.dumps(joint_params_from_settings(settings), ensure_ascii=False)
        try:
            count, detail = generate_joint_record(context, record)
        except Exception as exc:
            clear_joint_geometry(record.joint_id)
            index = _joint_record_index(scene, record.joint_id)
            if index >= 0:
                scene.flatfab_joints.remove(index)
            self.report({"ERROR"}, str(exc)[:250])
            return {"CANCELLED"}

        scene.flatfab_last_result = f"已生成 {record.name}：{detail}；{count} 个连接辅助体/布尔步骤。"
        self.report({"INFO"}, scene.flatfab_last_result[:250])
        return {"FINISHED"}

class FLATFAB_OT_regenerate_joint(Operator):
    bl_idname = "flatfab.regenerate_joint"
    bl_label = "重新生成连接"
    bl_options = {"REGISTER", "UNDO"}
    joint_id: StringProperty()

    def execute(self, context):
        record = _find_joint_record(context.scene, self.joint_id)
        if record is None:
            self.report({"ERROR"}, "找不到连接记录")
            return {"CANCELLED"}
        try:
            count, detail = generate_joint_record(context, record)
        except Exception as exc:
            self.report({"ERROR"}, str(exc)[:250])
            return {"CANCELLED"}
        context.scene.flatfab_last_result = f"已重新生成 {record.name}：{detail}。"
        self.report({"INFO"}, context.scene.flatfab_last_result[:250])
        return {"FINISHED"}


class FLATFAB_OT_regenerate_all_joints(Operator):
    bl_idname = "flatfab.regenerate_all_joints"
    bl_label = "重新生成全部连接"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        ok, failed = 0, []
        for record in context.scene.flatfab_joints:
            try:
                generate_joint_record(context, record)
                ok += 1
            except Exception as exc:
                failed.append(f"{record.name}: {exc}")
        msg = f"已重新生成 {ok} 个连接"
        if failed:
            msg += f"；失败 {len(failed)} 个：" + "; ".join(failed[:2])
            self.report({"WARNING"}, msg[:250])
        else:
            self.report({"INFO"}, msg)
        context.scene.flatfab_last_result = msg
        return {"FINISHED"}


class FLATFAB_OT_clear_joint_geometry(Operator):
    bl_idname = "flatfab.clear_joint_geometry"
    bl_label = "清除连接几何（保留参数）"
    bl_options = {"REGISTER", "UNDO"}
    joint_id: StringProperty(default="")

    def execute(self, context):
        records = []
        if self.joint_id:
            record = _find_joint_record(context.scene, self.joint_id)
            if record:
                records = [record]
        else:
            records = list(context.scene.flatfab_joints)
        mods = helpers = 0
        for record in records:
            m, h = clear_joint_geometry(record.joint_id)
            mods += m; helpers += h
            record.status = "几何已清除，参数保留，可重新生成"
        context.scene.flatfab_last_result = f"已清除 {mods} 个连接修改器 / {helpers} 个辅助体；连接参数仍保留。"
        self.report({"INFO"}, context.scene.flatfab_last_result)
        return {"FINISHED"}


class FLATFAB_OT_delete_joint(Operator):
    bl_idname = "flatfab.delete_joint"
    bl_label = "删除连接"
    bl_options = {"REGISTER", "UNDO"}
    joint_id: StringProperty()

    def execute(self, context):
        scene = context.scene
        clear_joint_geometry(self.joint_id)
        index = _joint_record_index(scene, self.joint_id)
        if index >= 0:
            scene.flatfab_joints.remove(index)
        scene.flatfab_last_result = "已删除连接定义及其 FlatFab 连接几何。"
        self.report({"INFO"}, scene.flatfab_last_result)
        return {"FINISHED"}


class FLATFAB_OT_delete_all_joints(Operator):
    bl_idname = "flatfab.delete_all_joints"
    bl_label = "删除全部连接定义"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        for record in list(scene.flatfab_joints):
            clear_joint_geometry(record.joint_id)
        scene.flatfab_joints.clear()
        scene.flatfab_last_result = "已删除全部参数化连接定义和连接几何。"
        self.report({"INFO"}, scene.flatfab_last_result)
        return {"FINISHED"}



class FLATFAB_OT_load_joint_params(Operator):
    bl_idname = "flatfab.load_joint_params"
    bl_label = "载入连接参数"
    bl_options = {"REGISTER", "UNDO"}
    joint_id: StringProperty()

    def execute(self, context):
        record = _find_joint_record(context.scene, self.joint_id)
        if record is None:
            self.report({"ERROR"}, "找不到连接记录")
            return {"CANCELLED"}
        settings = context.scene.flatfab_settings
        params = json.loads(record.params_json or "{}")
        settings.joint_type = record.joint_type
        settings.joint_tolerance_mm = float(params.get("tolerance_mm", settings.joint_tolerance_mm))
        settings.joint_tolerance_width_only = bool(params.get("tolerance_width_only", settings.joint_tolerance_width_only))
        settings.joint_tooth_mode = params.get("tooth_mode", settings.joint_tooth_mode)
        settings.joint_average_input = params.get("average_input", settings.joint_average_input)
        settings.joint_tooth_count = int(params.get("tooth_count", settings.joint_tooth_count))
        settings.joint_tooth_width_mm = float(params.get("tooth_width_mm", settings.joint_tooth_width_mm))
        settings.joint_cross_reverse = bool(params.get("cross_reverse", settings.joint_cross_reverse))
        settings.joint_through_reverse = bool(params.get("through_reverse", settings.joint_through_reverse))
        settings.joint_edge_depth_mm = float(params.get("edge_depth_mm", settings.joint_edge_depth_mm))
        settings.joint_edge_wedge = bool(params.get("edge_wedge", settings.joint_edge_wedge))
        settings.joint_edge_wedge_ratio = float(params.get("edge_wedge_ratio", settings.joint_edge_wedge_ratio))
        settings.joint_markers.clear()
        for marker in params.get("markers", []):
            item = settings.joint_markers.add()
            item.gap_mm = float(marker.get("gap_mm", 0.0))
            item.width_mm = float(marker.get("width_mm", 1.0))
        context.scene.flatfab_last_result = f"已把 {record.name} 的参数载入上方编辑区。"
        self.report({"INFO"}, context.scene.flatfab_last_result)
        return {"FINISHED"}

class FLATFAB_OT_apply_joint_params(Operator):
    bl_idname = "flatfab.apply_joint_params"
    bl_label = "应用当前参数并重算"
    bl_options = {"REGISTER", "UNDO"}
    joint_id: StringProperty()

    def execute(self, context):
        record = _find_joint_record(context.scene, self.joint_id)
        if record is None:
            self.report({"ERROR"}, "找不到连接记录")
            return {"CANCELLED"}
        settings = context.scene.flatfab_settings
        if settings.joint_type != record.joint_type:
            self.report({"ERROR"}, "上方连接类型与该记录不同；请先点“载入”再编辑")
            return {"CANCELLED"}
        record.params_json = json.dumps(joint_params_from_settings(settings), ensure_ascii=False)
        try:
            _count, detail = generate_joint_record(context, record)
        except Exception as exc:
            self.report({"ERROR"}, str(exc)[:250])
            return {"CANCELLED"}
        context.scene.flatfab_last_result = f"已应用参数并重算 {record.name}：{detail}。"
        self.report({"INFO"}, context.scene.flatfab_last_result[:250])
        return {"FINISHED"}


class FLATFAB_PG_Settings(PropertyGroup):
    output_collection: StringProperty(
        name="输出 Collection",
        default="FLAT_LAYOUT",
        description="展平结果存放位置",
    )
    prefix: StringProperty(name="名称前缀", default="FLAT_")
    solidify_thickness_mm: FloatProperty(
        name="板厚 (mm)", default=5.0, min=0.001, soft_max=100.0, precision=3,
        description="添加/更新实体化修改器时使用的实际板厚",
    )
    solidify_offset: EnumProperty(
        name="实体化偏移",
        items=(("POSITIVE", "+1", "沿板面正法线实体化"), ("NEGATIVE", "-1", "沿板面反法线实体化")),
        default="POSITIVE",
    )
    projection_mode: EnumProperty(
        name="板面识别",
        description="自动寻找最大板面，或直接指定物体的局部厚度轴",
        items=(
            ("AUTO", "自动（主平面）", "按平行面总面积自动识别板面"),
            ("LOCAL_Z", "Local Z", "把物体 Local Z 当作板厚方向"),
            ("LOCAL_Y", "Local Y", "把物体 Local Y 当作板厚方向"),
            ("LOCAL_X", "Local X", "把物体 Local X 当作板厚方向"),
        ),
        default="AUTO",
    )
    angle_tolerance: FloatProperty(
        name="角度容差 (°)",
        description="判断两个面是否近似平行，单位为度",
        default=2.0,
        min=0.05,
        max=15.0,
        precision=2,
    )
    plane_tolerance_mm: FloatProperty(
        name="平面距离容差 (mm)", default=0.05, min=0.001, soft_max=2.0, precision=3
    )
    gap_mm: FloatProperty(
        name="零件间距 (mm)", default=10.0, min=0.0, soft_max=100.0, precision=2
    )
    row_width_mm: FloatProperty(
        name="自由排版宽度 (mm)",
        description="未启用板材尺寸时使用；0 = 不换行",
        default=1000.0,
        min=0.0,
        soft_max=5000.0,
        precision=1,
    )
    use_sheet_size: BoolProperty(
        name="按板材尺寸排版",
        description="使用指定板材宽高与边距作为排版边界",
        default=True,
    )
    sheet_width_mm: FloatProperty(
        name="板材宽度 (mm)", default=600.0, min=1.0, soft_max=5000.0, precision=1
    )
    sheet_height_mm: FloatProperty(
        name="板材高度 (mm)", default=400.0, min=1.0, soft_max=5000.0, precision=1
    )
    sheet_margin_mm: FloatProperty(
        name="板材边距 (mm)", default=10.0, min=0.0, soft_max=100.0, precision=2
    )
    sort_mode: EnumProperty(
        name="排版排序",
        items=(
            ("AREA_DESC", "面积大优先", "较大零件先排"),
            ("LONG_DESC", "长边大优先", "长边较大的零件先排"),
            ("NAME", "按名称", "按物体名称排序"),
        ),
        default="AREA_DESC",
    )
    long_side_horizontal: BoolProperty(
        name="展平时长边横放",
        description="若展平后高度大于宽度，则把局部几何旋转 90°",
        default=True,
    )
    allow_rotate_90: BoolProperty(
        name="排版允许旋转 90°",
        description="当剩余宽度不足时允许把零件旋转 90°以改善矩形排版",
        default=True,
    )
    use_modifiers: BoolProperty(
        name="包含修改器结果",
        description="读取 Boolean 等修改器计算后的最终几何",
        default=True,
    )
    clear_before_build: BoolProperty(
        name="生成前清空旧结果",
        description="只删除本插件生成并带标记的旧展平对象",
        default=True,
    )
    use_lock_groups: BoolProperty(
        name="锁组作为整体",
        description="保持方向排版时，把同一 lock 组内的多个零件视为一个整体移动",
        default=True,
    )
    use_name_lock_groups: BoolProperty(
        name="从名称识别锁组",
        description="识别名称中的 _lock1、_lock2 等 token；源物体名称也会被检查",
        default=True,
    )
    lock_group_name: StringProperty(
        name="锁组名称",
        description="给选中展平零件指定锁组，例如 lock1；不写 lock 前缀也会自动补上",
        default="lock1",
    )

    # Parametric joints
    joint_type: EnumProperty(
        name="连接类型",
        items=(
            ("EDGE_INSERT", "边插", "自动识别边贴面关系；边侧插齿，面侧插槽"),
            ("CROSS", "十字插", "两板交叉重叠，每个有效区间固定二等分为互补半齿"),
            ("EDGE_CONNECT", "边连接", "A/B 共面贴边，生成锯齿或楔形拼接"),
            ("THROUGH", "贯穿", "自动判断贯穿方，并在接收板生成多区间投影槽"),
        ),
        default="EDGE_INSERT",
    )
    joint_tolerance_mm: FloatProperty(
        name="公差 (mm)", default=0.2, min=0.0, soft_max=5.0, precision=3,
        description="公差只放大插槽/插孔，不改变插齿",
    )
    joint_tolerance_width_only: BoolProperty(
        name="仅宽度公差",
        default=False,
        description="启用后只放大齿宽/槽长方向，不再放大用于容纳另一块板材厚度的槽宽",
    )
    joint_detect_tolerance_mm: FloatProperty(
        name="贴合识别容差 (mm)", default=1.0, min=0.01, soft_max=10.0, precision=3,
        description="用于识别边插的边-面贴合、或两块共面板是否真正贴边",
    )
    joint_angle_tolerance_deg: FloatProperty(
        name="角度识别容差 (°)", default=5.0, min=0.1, max=20.0, precision=2,
    )
    joint_tooth_mode: EnumProperty(
        name="齿模式",
        items=(
            ("AVERAGE", "平均齿", "沿整条贴合边自动均匀分配锯齿"),
            ("POINT", "点齿", "按数量均匀放置固定宽度的局部齿"),
            ("MARK", "标记齿", "按逐个标记的间距和宽度生成不均匀齿"),
        ),
        default="AVERAGE",
    )
    joint_average_input: EnumProperty(
        name="平均齿输入",
        items=(("COUNT", "按齿数", "指定 A 齿数量"), ("WIDTH", "按目标齿宽", "按目标齿宽自动估算齿数并重新均分")),
        default="COUNT",
    )
    joint_tooth_count: IntProperty(name="齿数", default=4, min=1, soft_max=50)
    joint_tooth_width_mm: FloatProperty(
        name="齿宽 (mm)", default=50.0, min=0.001, soft_max=500.0, precision=2,
    )
    joint_markers: CollectionProperty(type=FLATFAB_PG_JointMarker)
    joint_cross_reverse: BoolProperty(
        name="反转半齿 / 进入方向", default=False,
        description="交换十字插两块板各自承担的 1/2 槽，并反转优先进入方向",
    )
    joint_through_reverse: BoolProperty(
        name="反转贯穿方", default=False,
        description="贯穿默认自动选择较局部的板作为贯穿片；自动判断不符合装配意图时反转",
    )
    joint_edge_depth_mm: FloatProperty(
        name="连接齿深度 (mm)", default=10.0, min=0.001, soft_max=200.0, precision=2,
    )
    joint_edge_wedge: BoolProperty(name="楔形 / 拼图齿", default=False)
    joint_edge_wedge_ratio: FloatProperty(
        name="楔形扩张", default=0.20, min=0.0, max=0.45, subtype="FACTOR", precision=2,
        description="齿尖相对齿根的横向扩张比例",
    )

    square_packing: BoolProperty(
        name="方形拼板",
        default=False,
        description="自动估算接近正方形的排版行宽；固定板材模式下仍受板材可用宽度限制",
    )

    # UI fold state
    ui_fold_setup: BoolProperty(name="展开单位与实体化", default=True)
    ui_fold_joints: BoolProperty(name="展开参数化连接", default=True)
    ui_fold_flatten: BoolProperty(name="展开 1:1 展平", default=True)
    ui_fold_transform: BoolProperty(name="展开方向与锁组", default=False)
    ui_fold_layout: BoolProperty(name="展开排版", default=True)
    ui_fold_export: BoolProperty(name="展开导出", default=True)
    ui_fold_uv: BoolProperty(name="展开 UV", default=True)
    ui_fold_manage: BoolProperty(name="展开输出管理", default=False)

    export_format: EnumProperty(
        name="导出格式",
        items=(
            ("SVG", "SVG", "矢量切割文件，页面单位明确为 mm"),
            ("DXF", "DXF", "AutoCAD DXF，坐标单位为 mm"),
            ("PDF", "PDF", "矢量 PDF，页面按 1:1 物理尺寸生成"),
        ),
        default="SVG",
    )
    export_scope: EnumProperty(
        name="导出范围",
        items=(
            ("ALL", "全部展平结果", "导出输出 Collection 中的全部 FlatFab 零件"),
            ("SELECTED", "仅选中的展平结果", "只导出当前选中的 FlatFab 零件"),
        ),
        default="ALL",
    )
    export_page_mode: EnumProperty(
        name="页面尺寸",
        items=(
            ("CONTENT", "按内容包围框", "按实际轮廓自动生成最小页面并添加导出边距"),
            ("SHEET", "按板材尺寸", "使用板材宽度和高度作为 SVG/PDF 页面或 DXF 布局边界"),
        ),
        default="SHEET",
    )
    export_margin_mm: FloatProperty(
        name="导出边距 (mm)", default=5.0, min=0.0, soft_max=100.0, precision=2
    )
    export_line_width_mm: FloatProperty(
        name="线宽 (mm)",
        description="SVG/PDF 可见线宽；DXF 几何仍是零宽中心线",
        default=0.10,
        min=0.001,
        soft_max=2.0,
        precision=3,
    )
    export_sheet_border: BoolProperty(
        name="导出板材边框",
        description="把页面/板材矩形也写入文件；激光软件可能会把它识别为切割线",
        default=False,
    )
    export_labels: BoolProperty(
        name="导出零件名称",
        description="增加文字标签作为 ENGRAVE/文本信息；默认关闭以避免被误切",
        default=False,
    )
    label_size_mm: FloatProperty(
        name="标签字号 (mm)", default=4.0, min=0.5, soft_max=30.0, precision=1
    )


# -----------------------------------------------------------------------------
# Operators
# -----------------------------------------------------------------------------


class FLATFAB_OT_flatten_selected(Operator):
    bl_idname = "flatfab.flatten_selected"
    bl_label = "1:1 展平选中板件"
    bl_description = "将选中的刚性平板零件正投影为真实尺寸 XY 切割轮廓"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT" and any(
            obj.type == "MESH" for obj in context.selected_objects
        )

    def execute(self, context):
        scene = context.scene
        settings = scene.flatfab_settings
        source_objects = [obj for obj in context.selected_objects if obj.type == "MESH"]

        if not source_objects:
            self.report({"ERROR"}, "请先选中至少一个 Mesh 板件")
            return {"CANCELLED"}

        output_collection = get_or_create_collection(
            scene, settings.output_collection.strip() or "FLAT_LAYOUT"
        )

        if settings.clear_before_build:
            clear_generated(output_collection)

        depsgraph = context.evaluated_depsgraph_get()
        created = []
        failed = []
        bevels = enabled_bevel_objects(source_objects) if settings.use_modifiers else []

        for obj in source_objects:
            try:
                flat_obj, _width, _height = flatten_object(
                    obj, depsgraph, output_collection, scene, settings
                )
                created.append(flat_obj)
            except Exception as exc:
                failed.append((obj.name, str(exc)))

        layout_info = None
        if created:
            layout_info = arrange_objects(scene, settings, created)
            if settings.use_sheet_size:
                create_sheet_preview(scene, settings)

            bpy.ops.object.select_all(action="DESELECT")
            for obj in created:
                obj.select_set(True)
            context.view_layer.objects.active = created[0]

        if failed:
            short_fail = "; ".join(f"{name}: {msg}" for name, msg in failed[:3])
            scene.flatfab_last_result = f"完成 {len(created)}，失败 {len(failed)}。{short_fail}"
        else:
            scene.flatfab_last_result = f"完成 {len(created)} 个板件。"

        if layout_info and layout_info["rotated_count"]:
            scene.flatfab_last_result += f" 排版旋转 {layout_info['rotated_count']} 个。"
        if layout_info and layout_info["overflow"]:
            scene.flatfab_last_result += f" 有 {len(layout_info['overflow'])} 个超出板材边界。"
        if bevels:
            scene.flatfab_last_result += (
                f" 注意：{len(bevels)} 个源物体启用了 Bevel，外轮廓可能取到倒角后的平面。"
            )

        if not created:
            self.report({"ERROR"}, scene.flatfab_last_result)
            return {"CANCELLED"}

        if failed or bevels or (layout_info and layout_info["overflow"]):
            self.report({"WARNING"}, scene.flatfab_last_result[:250])
        else:
            self.report({"INFO"}, f"已展平并排版 {len(created)} 个板件")
        return {"FINISHED"}


class FLATFAB_OT_repack(Operator):
    bl_idname = "flatfab.repack"
    bl_label = "重新排版"
    bl_description = "按当前板材、间距、旋转和排序设置重新排列已有展平结果"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        settings = scene.flatfab_settings
        collection = bpy.data.collections.get(settings.output_collection)
        if collection is None:
            self.report({"ERROR"}, "找不到输出 Collection")
            return {"CANCELLED"}

        objects = generated_objects(collection)
        if not objects:
            self.report({"ERROR"}, "输出 Collection 中没有本插件生成的对象")
            return {"CANCELLED"}

        info = arrange_objects(scene, settings, objects)
        if settings.use_sheet_size:
            create_sheet_preview(scene, settings)
        else:
            remove_sheet_preview(collection)

        scene.flatfab_last_result = f"已重新排版 {len(objects)} 个板件"
        if info["rotated_count"]:
            scene.flatfab_last_result += f"，其中旋转 {info['rotated_count']} 个"
        if info["overflow"]:
            scene.flatfab_last_result += f"；{len(info['overflow'])} 个超出板材边界"
            self.report({"WARNING"}, scene.flatfab_last_result)
        else:
            scene.flatfab_last_result += "。"
            self.report({"INFO"}, scene.flatfab_last_result)
        return {"FINISHED"}


class FLATFAB_OT_repack_preserve(Operator):
    bl_idname = "flatfab.repack_preserve"
    bl_label = "保持方向排版"
    bl_description = "只平移现有零件进行排版；保留当前旋转、镜像、缩放以及锁组内部相对位置"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        settings = scene.flatfab_settings
        collection = bpy.data.collections.get(settings.output_collection)
        if collection is None:
            self.report({"ERROR"}, "找不到输出 Collection")
            return {"CANCELLED"}

        objects = generated_objects(collection)
        if not objects:
            self.report({"ERROR"}, "输出 Collection 中没有本插件生成的对象")
            return {"CANCELLED"}

        info = arrange_preserve_orientation(scene, settings, objects)
        if settings.use_sheet_size:
            create_sheet_preview(scene, settings)
        else:
            remove_sheet_preview(collection)

        scene.flatfab_last_result = f"保持方向排版 {len(objects)} 个板件"
        if info["group_count"]:
            scene.flatfab_last_result += (
                f"；识别 {info['group_count']} 个锁组 / {info['locked_parts']} 个组内零件"
            )
        if info["overflow"]:
            scene.flatfab_last_result += f"；{len(info['overflow'])} 个排版单元超出板材边界"
            self.report({"WARNING"}, scene.flatfab_last_result)
        else:
            scene.flatfab_last_result += "。"
            self.report({"INFO"}, scene.flatfab_last_result)
        return {"FINISHED"}


class FLATFAB_OT_mirror_selected(Operator):
    bl_idname = "flatfab.mirror_selected"
    bl_label = "镜像选中零件"
    bl_description = "围绕当前选中零件整体的中心沿板面 X/Y 镜像；保持组内相对关系"
    bl_options = {"REGISTER", "UNDO"}

    axis: EnumProperty(
        name="镜像轴",
        items=(("X", "X", "板面世界 X 镜像"), ("Y", "Y", "板面世界 Y 镜像")),
        default="X",
    )

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT" and any(
            obj.type == "MESH" and obj.get(GENERATED_TAG, False)
            for obj in context.selected_objects
        )

    def execute(self, context):
        objects = [
            obj for obj in context.selected_objects
            if obj.type == "MESH" and obj.get(GENERATED_TAG, False)
        ]
        mirror_objects_world(objects, self.axis)
        context.scene.flatfab_last_result = f"已沿局部 {self.axis} 镜像 {len(objects)} 个零件。"
        self.report({"INFO"}, context.scene.flatfab_last_result)
        return {"FINISHED"}


class FLATFAB_OT_rotate_selected_90(Operator):
    bl_idname = "flatfab.rotate_selected_90"
    bl_label = "旋转选中零件 90°"
    bl_description = "围绕当前选中零件整体的中心旋转 90°；多选时保持彼此相对位置"
    bl_options = {"REGISTER", "UNDO"}

    direction: EnumProperty(
        name="方向",
        items=(("CCW", "+90°", "逆时针 90°"), ("CW", "-90°", "顺时针 90°")),
        default="CCW",
    )

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT" and any(
            obj.type == "MESH" and obj.get(GENERATED_TAG, False)
            for obj in context.selected_objects
        )

    def execute(self, context):
        objects = [
            obj for obj in context.selected_objects
            if obj.type == "MESH" and obj.get(GENERATED_TAG, False)
        ]
        angle = math.radians(90.0 if self.direction == "CCW" else -90.0)
        rotate_objects_world(objects, angle)
        context.scene.flatfab_last_result = f"已旋转 {len(objects)} 个零件。"
        self.report({"INFO"}, context.scene.flatfab_last_result)
        return {"FINISHED"}


class FLATFAB_OT_assign_lock_group(Operator):
    bl_idname = "flatfab.assign_lock_group"
    bl_label = "连接为锁组"
    bl_description = "把选中的展平零件标记为同一个锁组；保持方向排版时整体移动"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT" and any(
            obj.get(GENERATED_TAG, False) for obj in context.selected_objects
        )

    def execute(self, context):
        settings = context.scene.flatfab_settings
        group = normalize_lock_name(settings.lock_group_name)
        if not group:
            self.report({"ERROR"}, "请输入锁组名称，例如 lock1")
            return {"CANCELLED"}
        objects = [obj for obj in context.selected_objects if obj.get(GENERATED_TAG, False)]
        for obj in objects:
            obj[LOCK_TAG] = group
        context.scene.flatfab_last_result = f"已将 {len(objects)} 个零件连接为锁组 {group}。"
        self.report({"INFO"}, context.scene.flatfab_last_result)
        return {"FINISHED"}


class FLATFAB_OT_clear_lock_group(Operator):
    bl_idname = "flatfab.clear_lock_group"
    bl_label = "解除锁组"
    bl_description = "清除选中展平零件的手动锁组标记；名称中的 _lockN 仍可继续被自动识别"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT" and any(
            obj.get(GENERATED_TAG, False) for obj in context.selected_objects
        )

    def execute(self, context):
        objects = [obj for obj in context.selected_objects if obj.get(GENERATED_TAG, False)]
        cleared = 0
        for obj in objects:
            if LOCK_TAG in obj:
                del obj[LOCK_TAG]
                cleared += 1
        context.scene.flatfab_last_result = f"已解除 {cleared} 个零件的手动锁组标记。"
        self.report({"INFO"}, context.scene.flatfab_last_result)
        return {"FINISHED"}


class FLATFAB_OT_update_sheet(Operator):
    bl_idname = "flatfab.update_sheet"
    bl_label = "更新板材边框"
    bl_description = "在 XY 平面生成/更新板材尺寸预览框（不会作为零件导出）"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        settings = scene.flatfab_settings
        if not settings.use_sheet_size:
            self.report({"ERROR"}, "请先启用‘按板材尺寸排版’")
            return {"CANCELLED"}
        create_sheet_preview(scene, settings)
        self.report({"INFO"}, f"板材预览：{settings.sheet_width_mm:g} × {settings.sheet_height_mm:g} mm")
        return {"FINISHED"}


class FLATFAB_OT_select_layout(Operator):
    bl_idname = "flatfab.select_layout"
    bl_label = "选择展平结果"
    bl_description = "选择输出 Collection 中由 FlatFab 生成的零件对象"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.flatfab_settings
        collection = bpy.data.collections.get(settings.output_collection)
        if collection is None:
            self.report({"ERROR"}, "找不到输出 Collection")
            return {"CANCELLED"}

        objects = generated_objects(collection)
        bpy.ops.object.select_all(action="DESELECT")
        for obj in objects:
            obj.hide_set(False)
            obj.select_set(True)
        if objects:
            context.view_layer.objects.active = objects[0]
        self.report({"INFO"}, f"已选择 {len(objects)} 个展平对象")
        return {"FINISHED"}


class FLATFAB_OT_update_uv(Operator):
    bl_idname = "flatfab.update_uv"
    bl_label = "更新 UV"
    bl_description = "把当前整张 SVG 导出画布的排版坐标写入所有原始板件的 FlatFab_Layout UV"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        settings = scene.flatfab_settings
        collection = bpy.data.collections.get(settings.output_collection)
        if collection is None:
            self.report({"ERROR"}, "找不到输出 Collection")
            return {"CANCELLED"}

        flat_objects = generated_objects(collection)
        if not flat_objects:
            self.report({"ERROR"}, "输出 Collection 中没有本插件生成的对象")
            return {"CANCELLED"}

        try:
            # UV always represents the complete current manufacturing layout.
            # Border/labels never enter this geometry calculation.
            object_data = export_object_data(scene, flat_objects)
            page = compute_export_page(settings, object_data)
        except Exception as exc:
            self.report({"ERROR"}, f"无法计算 UV 画布：{exc}")
            return {"CANCELLED"}

        depsgraph = context.evaluated_depsgraph_get()
        updated = []
        failed = []
        rebuilt_projection = 0
        outside = 0
        helper_uvs = 0
        seen_sources = set()

        for flat_obj in flat_objects:
            source_name = str(flat_obj.get(SOURCE_TAG, "")).strip()
            source_obj = bpy.data.objects.get(source_name) if source_name else None
            if source_obj is None:
                failed.append((flat_obj.name, f"找不到源物体：{source_name or '(空)'}"))
                continue
            if source_obj.type != "MESH":
                failed.append((source_name, "源物体不是 Mesh"))
                continue
            if source_obj.name in seen_sources:
                failed.append((source_name, "同一源物体对应多个展平零件，无法写入唯一 UV 位置"))
                continue
            seen_sources.add(source_obj.name)

            try:
                rows, rebuilt = get_projection_rows_for_uv(
                    flat_obj, source_obj, depsgraph, scene, settings
                )
                if rebuilt:
                    rebuilt_projection += 1
                bounds = write_layout_uv(source_obj, flat_obj, rows, scene, page)
                helper_uvs += update_union_helper_uvs(source_obj, flat_obj, rows, scene, page)
                if bounds[0] < -1.0e-6 or bounds[1] < -1.0e-6 or bounds[2] > 1.0 + 1.0e-6 or bounds[3] > 1.0 + 1.0e-6:
                    outside += 1
                updated.append(source_obj)
            except Exception as exc:
                failed.append((source_name, str(exc)))

        if not updated:
            detail = "; ".join(f"{name}: {msg}" for name, msg in failed[:3])
            scene.flatfab_last_result = f"UV 更新失败。{detail}"
            self.report({"ERROR"}, scene.flatfab_last_result[:250])
            return {"CANCELLED"}

        msg = (
            f"已更新 {len(updated)} 个原始板件的 {UV_MAP_NAME} UV；"
            f"画布 {page['width']:.1f} × {page['height']:.1f} mm"
        )
        if rebuilt_projection:
            msg += f"；兼容重建 {rebuilt_projection} 个旧版投影"
        if helper_uvs:
            msg += f"；同步 {helper_uvs} 个插齿辅助体 UV"
        if outside:
            msg += f"；{outside} 个板件 UV 超出 0–1（对应当前页面越界）"
        if failed:
            detail = "; ".join(f"{name}: {text}" for name, text in failed[:3])
            msg += f"；失败 {len(failed)} 个：{detail}"
        else:
            msg += "。"

        scene.flatfab_last_result = msg
        if failed or outside or page["overflow"]:
            self.report({"WARNING"}, msg[:250])
        else:
            self.report({"INFO"}, msg[:250])
        return {"FINISHED"}


class FLATFAB_OT_clear_layout(Operator):
    bl_idname = "flatfab.clear_layout"
    bl_label = "清空展平结果"
    bl_description = "删除输出 Collection 中由 FlatFab 生成的对象与板材预览"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.flatfab_settings
        collection = bpy.data.collections.get(settings.output_collection)
        if collection is None:
            self.report({"INFO"}, "没有需要清理的输出 Collection")
            return {"FINISHED"}

        count = clear_generated(collection)
        remove_sheet_preview(collection)
        context.scene.flatfab_last_result = f"已清理 {count} 个展平对象。"
        self.report({"INFO"}, context.scene.flatfab_last_result)
        return {"FINISHED"}


class FLATFAB_OT_export_layout(Operator, ExportHelper):
    bl_idname = "flatfab.export_layout"
    bl_label = "导出制造文件"
    bl_description = "按当前设置导出 1:1 SVG / DXF / PDF"

    filename_ext = ".svg"
    filter_glob: StringProperty(default="*.svg;*.dxf;*.pdf", options={"HIDDEN"})

    def invoke(self, context, event):
        fmt = context.scene.flatfab_settings.export_format
        self.filename_ext = "." + fmt.lower()
        base = "flatfab_layout"
        if bpy.data.filepath:
            base = clean_name(os.path.splitext(bpy.path.basename(bpy.data.filepath))[0]) + "_flatfab"
        self.filepath = ensure_extension(os.path.join(bpy.path.abspath("//"), base), fmt)
        return ExportHelper.invoke(self, context, event)

    def check(self, context):
        fmt = context.scene.flatfab_settings.export_format
        desired = ensure_extension(self.filepath, fmt)
        if desired != self.filepath:
            self.filepath = desired
            return True
        return False

    def execute(self, context):
        scene = context.scene
        settings = scene.flatfab_settings
        collection = bpy.data.collections.get(settings.output_collection)
        if collection is None:
            self.report({"ERROR"}, "找不到输出 Collection")
            return {"CANCELLED"}

        objects = generated_objects(collection)
        if settings.export_scope == "SELECTED":
            selected = set(context.selected_objects)
            objects = [obj for obj in objects if obj in selected]

        if not objects:
            self.report({"ERROR"}, "没有符合导出范围的 FlatFab 展平对象")
            return {"CANCELLED"}

        try:
            data = export_object_data(scene, objects)
            page = compute_export_page(settings, data)
            filepath = ensure_extension(self.filepath, settings.export_format)

            if settings.export_format == "SVG":
                export_svg(filepath, settings, data, page)
            elif settings.export_format == "DXF":
                export_dxf(filepath, settings, data, page)
            elif settings.export_format == "PDF":
                export_pdf(filepath, settings, data, page)
            else:
                raise RuntimeError("未知导出格式")

            self.filepath = filepath
            msg = (
                f"已导出 {len(data)} 个零件 → {os.path.basename(filepath)}；"
                f"页面 {page['width']:.1f} × {page['height']:.1f} mm"
            )
            if page["overflow"]:
                msg += "。警告：内容超出页面/板材可用范围"
                self.report({"WARNING"}, msg)
            else:
                self.report({"INFO"}, msg)
            scene.flatfab_last_result = msg
            return {"FINISHED"}
        except Exception as exc:
            self.report({"ERROR"}, f"导出失败：{exc}")
            return {"CANCELLED"}


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------


class FLATFAB_PT_main(Panel):
    bl_label = "FlatFab 板件制造"
    bl_idname = "FLATFAB_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "制造"

    def fold_header(self, layout, settings, prop_name, title, icon="NONE"):
        box = layout.box()
        row = box.row(align=True)
        opened = getattr(settings, prop_name)
        row.prop(settings, prop_name, text="", icon="TRIA_DOWN" if opened else "TRIA_RIGHT", emboss=False)
        row.label(text=title, icon=icon)
        return box, opened

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        settings = scene.flatfab_settings

        # 1. Units / Solidify
        box, opened = self.fold_header(layout, settings, "ui_fold_setup", "① 单位 / 实体化", "MOD_SOLIDIFY")
        if opened:
            row = box.row(align=True)
            op = row.operator("flatfab.set_scene_unit", text="毫米 (mm)")
            op.unit = "MM"
            op = row.operator("flatfab.set_scene_unit", text="厘米 (cm)")
            op.unit = "CM"
            units = scene.unit_settings
            current_unit = getattr(units, "length_unit", "") or "ADAPTIVE"
            box.label(text=f"当前：METRIC / {current_unit} / scale {units.scale_length:g}", icon="INFO")
            row = box.row(align=True)
            row.prop(settings, "solidify_thickness_mm")
            row.prop(settings, "solidify_offset", expand=True)
            col = box.column(align=True); col.scale_y = 1.15
            col.operator("flatfab.apply_solidify", text="给选中板件添加 / 更新实体化", icon="MOD_SOLIDIFY")
            box.label(text="已有 Solidify 会直接更新；厚度按实际 mm 换算，偏移固定为 +1 / -1", icon="INFO")

        # 2. Parametric joints
        box, opened = self.fold_header(layout, settings, "ui_fold_joints", "② 参数化插口 / 插槽", "MOD_BOOLEAN")
        if opened:
            box.label(text="选择两个原始板件；连接角色由几何自动识别，不依赖活动对象", icon="INFO")
            box.prop(settings, "joint_type", expand=True)
            row = box.row(align=True)
            row.prop(settings, "joint_tolerance_mm")
            row.prop(settings, "joint_tolerance_width_only")
            adv = box.row(align=True)
            adv.prop(settings, "joint_detect_tolerance_mm")
            adv.prop(settings, "joint_angle_tolerance_deg")

            if settings.joint_type == "EDGE_INSERT":
                box.separator()
                box.label(text="边插：自动识别‘边 = 插齿方 / 面 = 插槽方’")
                box.prop(settings, "joint_tooth_mode", expand=True)
                if settings.joint_tooth_mode == "AVERAGE":
                    box.prop(settings, "joint_average_input", expand=True)
                    if settings.joint_average_input == "COUNT":
                        box.prop(settings, "joint_tooth_count")
                    else:
                        box.prop(settings, "joint_tooth_width_mm", text="目标齿宽 (mm)")
                elif settings.joint_tooth_mode == "POINT":
                    row = box.row(align=True)
                    row.prop(settings, "joint_tooth_count")
                    row.prop(settings, "joint_tooth_width_mm")
                    box.label(text="点齿中心均匀分布；齿宽不能超过均匀间隙", icon="INFO")
                else:
                    box.label(text="标记间距从上一齿末端开始累计；第 1 个从边起点累计")
                    for i, marker in enumerate(settings.joint_markers):
                        row = box.row(align=True)
                        row.label(text=f"标记 {i + 1}")
                        row.prop(marker, "gap_mm", text="间距")
                        row.prop(marker, "width_mm", text="宽度")
                        op = row.operator("flatfab.remove_joint_marker", text="", icon="X")
                        op.index = i
                    box.operator("flatfab.add_joint_marker", text="增加标记", icon="ADD")
                box.label(text="齿深自动等于接收板厚；槽宽按插齿板厚 + 公差计算", icon="INFO")

            elif settings.joint_type == "CROSS":
                box.separator()
                box.prop(settings, "joint_cross_reverse")
                box.label(text="每个真实重叠区间固定二等分：一半切第一块板，一半切第二块板", icon="INFO")
                box.label(text="若一块板位于另一块重叠区间内部，会自动向外补进入缝；反转可换方向", icon="INFO")
                box.label(text="斜交槽宽 = 对方板厚 / sin(交角)，无‘段数’参数", icon="INFO")

            elif settings.joint_type == "EDGE_CONNECT":
                box.separator()
                row = box.row(align=True)
                row.prop(settings, "joint_tooth_count", text="齿对数")
                row.prop(settings, "joint_edge_depth_mm")
                box.prop(settings, "joint_edge_wedge")
                if settings.joint_edge_wedge:
                    box.prop(settings, "joint_edge_wedge_ratio")
                box.label(text="共面贴边后 A/B 双向生成 UNION 齿 + DIFFERENCE 槽；切刀会跨过接缝保证真实相交", icon="INFO")

            else:
                box.separator()
                box.prop(settings, "joint_through_reverse")
                box.label(text="贯穿方默认按重叠线支持长度/板面范围自动判断；必要时可反转", icon="INFO")
                box.label(text="使用多区间 polygon clipping：凹轮廓、孔洞、多个分离穿越区间会分别开槽", icon="INFO")

            col = box.column(align=True); col.scale_y = 1.25
            col.operator("flatfab.create_joint", text="生成参数化连接", icon="MOD_BOOLEAN")

            if len(scene.flatfab_joints):
                box.separator()
                box.label(text=f"已记录连接：{len(scene.flatfab_joints)}")
                for record in scene.flatfab_joints:
                    sub = box.box()
                    row = sub.row(align=True)
                    type_name = {"EDGE_INSERT":"边插", "CROSS":"十字插", "EDGE_CONNECT":"边连接", "THROUGH":"贯穿"}.get(record.joint_type, record.joint_type)
                    arrow = "→" if record.joint_type in {"EDGE_INSERT", "THROUGH"} else "↔"
                    row.label(text=f"{type_name}  {record.object_a} {arrow} {record.object_b}")
                    op = row.operator("flatfab.load_joint_params", text="载入"); op.joint_id = record.joint_id
                    op = row.operator("flatfab.apply_joint_params", text="应用"); op.joint_id = record.joint_id
                    op = row.operator("flatfab.regenerate_joint", text="重算", icon="FILE_REFRESH"); op.joint_id = record.joint_id
                    op = row.operator("flatfab.clear_joint_geometry", text="清几何"); op.joint_id = record.joint_id
                    op = row.operator("flatfab.delete_joint", text="", icon="TRASH"); op.joint_id = record.joint_id
                    if record.status:
                        sub.label(text=record.status[:78])
                row = box.row(align=True)
                row.operator("flatfab.regenerate_all_joints", text="重算全部", icon="FILE_REFRESH")
                row.operator("flatfab.clear_joint_geometry", text="清除全部几何")
                box.operator("flatfab.delete_all_joints", text="删除全部连接定义", icon="TRASH")

        # 3. Flatten
        box, opened = self.fold_header(layout, settings, "ui_fold_flatten", "③ 1:1 展平", "MODIFIER")
        if opened:
            col = box.column(align=True); col.scale_y = 1.2
            col.operator("flatfab.flatten_selected", icon="MODIFIER")
            box.prop(settings, "projection_mode")
            row = box.row(align=True); row.prop(settings, "angle_tolerance"); row.prop(settings, "plane_tolerance_mm")
            box.prop(settings, "use_modifiers")
            box.prop(settings, "long_side_horizontal")
            box.prop(settings, "clear_before_build")

        # 4. Transform / lock
        box, opened = self.fold_header(layout, settings, "ui_fold_transform", "④ 方向 / 镜像 / 锁组", "ORIENTATION_GLOBAL")
        if opened:
            row = box.row(align=True)
            op=row.operator("flatfab.rotate_selected_90", text="+90°"); op.direction="CCW"
            op=row.operator("flatfab.rotate_selected_90", text="-90°"); op.direction="CW"
            row = box.row(align=True)
            op=row.operator("flatfab.mirror_selected", text="镜像 X"); op.axis="X"
            op=row.operator("flatfab.mirror_selected", text="镜像 Y"); op.axis="Y"
            box.separator()
            box.prop(settings, "use_lock_groups")
            if settings.use_lock_groups:
                box.prop(settings, "use_name_lock_groups")
                if settings.use_name_lock_groups:
                    box.label(text="命名示例：Left_lock1 / Right_lock1", icon="INFO")
                row=box.row(align=True); row.prop(settings,"lock_group_name", text="组"); row.operator("flatfab.assign_lock_group", text="连接")
                box.operator("flatfab.clear_lock_group", text="解除手动锁组")

        # 5. Layout
        box, opened = self.fold_header(layout, settings, "ui_fold_layout", "⑤ 排版 / 板材", "MESH_GRID")
        if opened:
            row=box.row(align=True)
            row.operator("flatfab.repack_preserve", text="保持方向排版", icon="CON_LOCLIKE")
            row.operator("flatfab.repack", text="自动重排（重置）", icon="FILE_REFRESH")
            box.operator("flatfab.select_layout", icon="RESTRICT_SELECT_OFF")
            box.prop(settings, "use_sheet_size")
            if settings.use_sheet_size:
                row=box.row(align=True); row.prop(settings,"sheet_width_mm"); row.prop(settings,"sheet_height_mm")
                box.prop(settings,"sheet_margin_mm")
                box.operator("flatfab.update_sheet", icon="MESH_GRID")
            else:
                box.prop(settings,"row_width_mm")
            box.prop(settings,"gap_mm")
            box.prop(settings,"square_packing")
            if settings.square_packing:
                box.label(text="方形拼板会自动估算接近 1:1 的行宽", icon="INFO")
            box.prop(settings,"sort_mode")
            box.prop(settings,"allow_rotate_90")
            collection=bpy.data.collections.get(settings.output_collection)
            objs=generated_objects(collection) if collection else []
            if objs:
                bounds=world_xy_bounds_mm(scene, objs)
                if bounds:
                    box.label(text=f"当前布局：{len(objs)} 件 / {bounds[2]-bounds[0]:.1f} × {bounds[3]-bounds[1]:.1f} mm")

        # 6. Export
        box, opened = self.fold_header(layout, settings, "ui_fold_export", "⑥ SVG / DXF / PDF 导出", "EXPORT")
        if opened:
            box.prop(settings,"export_format", expand=True)
            box.prop(settings,"export_scope")
            box.prop(settings,"export_page_mode")
            if settings.export_page_mode == "SHEET":
                box.label(text=f"页面：{settings.sheet_width_mm:g} × {settings.sheet_height_mm:g} mm")
                box.label(text="SHEET 模式保持当前 XY 坐标，不重新归位", icon="INFO")
            else:
                box.prop(settings,"export_margin_mm")
            box.prop(settings,"export_line_width_mm")
            box.prop(settings,"export_sheet_border")
            box.prop(settings,"export_labels")
            if settings.export_labels:
                box.prop(settings,"label_size_mm")
                if settings.export_format == "PDF":
                    box.label(text="PDF 内置字体不含完整中文；中文名可能显示为 ?", icon="INFO")
            col=box.column(align=True); col.scale_y=1.2; col.operator("flatfab.export_layout", icon="EXPORT")

        # 7. UV
        box, opened = self.fold_header(layout, settings, "ui_fold_uv", "⑦ 更新 UV", "UV_DATA")
        if opened:
            box.label(text=f"UV 图层：{UV_MAP_NAME}")
            box.label(text="按当前整张 SVG 页面坐标映射原始板件", icon="INFO")
            box.label(text="固定按全部展平结果，与‘导出范围=全部’对应", icon="INFO")
            box.label(text="板材边框/文字不参与；PS 中请勿裁切画布", icon="INFO")
            col=box.column(align=True); col.scale_y=1.2; col.operator("flatfab.update_uv", icon="UV_DATA")

        # 8. Manage
        box, opened = self.fold_header(layout, settings, "ui_fold_manage", "⑧ 输出管理", "OUTLINER_COLLECTION")
        if opened:
            box.prop(settings,"output_collection")
            box.prop(settings,"prefix")
            box.operator("flatfab.clear_layout", icon="TRASH")

        if scene.flatfab_last_result:
            info=layout.box(); info.label(text="最近结果")
            text=scene.flatfab_last_result
            while len(text)>42:
                info.label(text=text[:42]); text=text[42:]
            if text: info.label(text=text)

        layout.separator()
        layout.label(text="连接非破坏可重算；刀线/排版保持真实 XY；UV 同步原始板件。")


classes = (
    FLATFAB_OT_set_scene_unit,
    FLATFAB_OT_apply_solidify,
    FLATFAB_PG_JointMarker,
    FLATFAB_PG_JointRecord,
    FLATFAB_PG_Settings,
    FLATFAB_OT_add_joint_marker,
    FLATFAB_OT_remove_joint_marker,
    FLATFAB_OT_create_joint,
    FLATFAB_OT_regenerate_joint,
    FLATFAB_OT_regenerate_all_joints,
    FLATFAB_OT_clear_joint_geometry,
    FLATFAB_OT_delete_joint,
    FLATFAB_OT_delete_all_joints,
    FLATFAB_OT_load_joint_params,
    FLATFAB_OT_apply_joint_params,
    FLATFAB_OT_flatten_selected,
    FLATFAB_OT_repack,
    FLATFAB_OT_repack_preserve,
    FLATFAB_OT_mirror_selected,
    FLATFAB_OT_rotate_selected_90,
    FLATFAB_OT_assign_lock_group,
    FLATFAB_OT_clear_lock_group,
    FLATFAB_OT_update_sheet,
    FLATFAB_OT_select_layout,
    FLATFAB_OT_update_uv,
    FLATFAB_OT_clear_layout,
    FLATFAB_OT_export_layout,
    FLATFAB_PT_main,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.flatfab_settings = PointerProperty(type=FLATFAB_PG_Settings)
    bpy.types.Scene.flatfab_joints = CollectionProperty(type=FLATFAB_PG_JointRecord)
    bpy.types.Scene.flatfab_last_result = StringProperty(default="")


def unregister():
    if hasattr(bpy.types.Scene, "flatfab_last_result"):
        del bpy.types.Scene.flatfab_last_result
    if hasattr(bpy.types.Scene, "flatfab_joints"):
        del bpy.types.Scene.flatfab_joints
    if hasattr(bpy.types.Scene, "flatfab_settings"):
        del bpy.types.Scene.flatfab_settings

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
