#!/usr/bin/env python3
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom


def text_or(elem, default="0"):
    if elem is None or elem.text is None:
        return default
    return elem.text.strip()


def parse_pose(elem):
    pose = text_or(elem, "0 0 0 0 0 0").split()
    if len(pose) != 6:
        pose = ["0", "0", "0", "0", "0", "0"]
    return pose


def geometry_to_urdf(geom_elem):
    box = geom_elem.find("box")
    if box is not None:
        size = text_or(box.find("size"), "0 0 0")
        g = ET.Element("box")
        g.set("size", size)
        return g
    sphere = geom_elem.find("sphere")
    if sphere is not None:
        radius = text_or(sphere.find("radius"), "0")
        g = ET.Element("sphere")
        g.set("radius", radius)
        return g
    cyl = geom_elem.find("cylinder")
    if cyl is not None:
        radius = text_or(cyl.find("radius"), "0")
        length = text_or(cyl.find("length"), "0")
        g = ET.Element("cylinder")
        g.set("radius", radius)
        g.set("length", length)
        return g
    return None


def add_origin(parent, pose):
    origin = ET.SubElement(parent, "origin")
    origin.set("xyz", " ".join(pose[:3]))
    origin.set("rpy", " ".join(pose[3:]))


def add_inertial(link, inertial_elem):
    if inertial_elem is None:
        return
    inertial = ET.SubElement(link, "inertial")
    pose = parse_pose(inertial_elem.find("pose"))
    add_origin(inertial, pose)
    mass = ET.SubElement(inertial, "mass")
    mass.set("value", text_or(inertial_elem.find("mass"), "0.0"))
    inertia = ET.SubElement(inertial, "inertia")
    ixx = text_or(inertial_elem.find("inertia/ixx"), "0")
    ixy = text_or(inertial_elem.find("inertia/ixy"), "0")
    ixz = text_or(inertial_elem.find("inertia/ixz"), "0")
    iyy = text_or(inertial_elem.find("inertia/iyy"), "0")
    iyz = text_or(inertial_elem.find("inertia/iyz"), "0")
    izz = text_or(inertial_elem.find("inertia/izz"), "0")
    inertia.set("ixx", ixx)
    inertia.set("ixy", ixy)
    inertia.set("ixz", ixz)
    inertia.set("iyy", iyy)
    inertia.set("iyz", iyz)
    inertia.set("izz", izz)


def add_visual_or_collision(link, tag, elem):
    if elem is None:
        return
    out = ET.SubElement(link, tag)
    pose = parse_pose(elem.find("pose"))
    add_origin(out, pose)
    geometry = elem.find("geometry")
    if geometry is None:
        return
    g = geometry_to_urdf(geometry)
    if g is None:
        return
    geom_out = ET.SubElement(out, "geometry")
    geom_out.append(g)


def add_link(parent, sdf_link):
    name = sdf_link.get("name")
    link = ET.SubElement(parent, "link")
    link.set("name", name)

    add_inertial(link, sdf_link.find("inertial"))
    add_visual_or_collision(link, "visual", sdf_link.find("visual"))
    add_visual_or_collision(link, "collision", sdf_link.find("collision"))

    return link


def add_joint(parent, sdf_joint, child_pose):
    name = sdf_joint.get("name")
    joint_type = sdf_joint.get("type")
    joint = ET.SubElement(parent, "joint")
    joint.set("name", name)
    joint.set("type", joint_type)

    parent_link = text_or(sdf_joint.find("parent"), "")
    child_link = text_or(sdf_joint.find("child"), "")
    pl = ET.SubElement(joint, "parent")
    pl.set("link", parent_link)
    cl = ET.SubElement(joint, "child")
    cl.set("link", child_link)

    add_origin(joint, child_pose)

    axis = sdf_joint.find("axis/xyz")
    if axis is not None:
        axis_elem = ET.SubElement(joint, "axis")
        axis_elem.set("xyz", text_or(axis, "0 0 0"))
        limit = sdf_joint.find("axis/limit")
        if limit is not None:
            lim = ET.SubElement(joint, "limit")
            lim.set("lower", text_or(limit.find("lower"), "0"))
            lim.set("upper", text_or(limit.find("upper"), "0"))
            lim.set("effort", "0")
            lim.set("velocity", "0")


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sdf_path = os.path.join(repo_root, "ai_slam_ws", "src", "ai_slam_description", "models", "diffbot.sdf")
    urdf_path = os.path.join(repo_root, "ai_slam_ws", "src", "ai_slam_description", "urdf", "diffbot.urdf")

    tree = ET.parse(sdf_path)
    root = tree.getroot()
    model = root.find("model")
    if model is None:
        raise SystemExit("No <model> in SDF.")

    robot = ET.Element("robot")
    robot.set("name", model.get("name", "diffbot"))
    robot.append(ET.Comment(" Auto-generated from models/diffbot.sdf. Do not edit manually. "))
    robot.append(ET.Comment(" TF tree: odom -> base_link -> laser_link "))

    links = {link.get("name"): link for link in model.findall("link")}

    # Add links in deterministic order, then any remaining
    ordered = ["base_link", "left_wheel_link", "right_wheel_link", "laser_link", "caster_link"]
    for name in ordered:
        if name in links:
            add_link(robot, links[name])
    for name, link in links.items():
        if name not in ordered:
            add_link(robot, link)

    # Add joints based on child poses
    for joint in model.findall("joint"):
        child = text_or(joint.find("child"), "")
        child_link = links.get(child)
        child_pose = parse_pose(child_link.find("pose")) if child_link is not None else ["0"] * 6
        add_joint(robot, joint, child_pose)

    # Pretty print
    xml_raw = ET.tostring(robot, encoding="utf-8")
    xml_str = minidom.parseString(xml_raw).toprettyxml(indent="  ")
    with open(urdf_path, "w", encoding="utf-8") as f:
        f.write(xml_str)


if __name__ == "__main__":
    main()
