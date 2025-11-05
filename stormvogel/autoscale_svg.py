import io
import re

"""Used to autoscale an svg to remove unused space from the image."""


def remove_invalid_paths(svg_string: str) -> str:
    """Remove <path> elements that have no 'd' attribute"""
    return re.sub(r"<path(?![^>]* d=)[^>]*/>", "", svg_string)


def autoscale_svg(raw_svg: str, target_width: float) -> str:
    """Autoscale an SVG string to the target width while maintaining aspect ratio."""
    from lxml import etree
    from svgpathtools import svg2paths2

    # Parse the SVG from the raw string
    # Using lxml to parse the raw SVG string
    clean_svg = remove_invalid_paths(raw_svg)
    root = etree.fromstring(clean_svg)

    # Extract paths and calculate the bounding box using svgpathtools
    paths, attributes, svg_attr = svg2paths2(io.StringIO(clean_svg))

    # Calculate the bounding box of all paths
    xmin, xmax, ymin, ymax = 0, 0, 0, 0
    for i, path in enumerate(paths):
        atr = attributes[i]
        if atr["stroke"] != "none" and "d" in atr:
            box = path.bbox()
            x0, x1, y0, y1 = box
            xmin = x0 if xmin is None else min(xmin, x0)
            xmax = x1 if xmax is None else max(xmax, x1)
            ymin = y0 if ymin is None else min(ymin, y0)
            ymax = y1 if ymax is None else max(ymax, y1)
    width = xmax - xmin
    height = ymax - ymin

    # Set the viewBox and the width/height attributes to match the content size
    root.attrib["viewBox"] = f"{xmin} {ymin} {width} {height}"
    root.attrib["width"] = str(width)
    root.attrib["height"] = str(height)

    # Now scale according to the provided target width while keeping the aspect ratio
    aspect_ratio = height / width
    new_width = target_width
    new_height = target_width * aspect_ratio

    # Update width and height attributes in the SVG
    root.attrib["width"] = str(new_width)
    root.attrib["height"] = str(new_height)

    # Return the modified SVG as a string
    return etree.tostring(
        root,
        pretty_print=True,
        xml_declaration=True,
        encoding="utf-8",
    ).decode()
