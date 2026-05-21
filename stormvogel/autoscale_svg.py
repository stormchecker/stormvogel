import io
import re
import xml.etree.ElementTree as ET

"""Used to autoscale an svg to remove unused space from the image."""


def remove_invalid_paths(svg_string: str) -> str:
    """Remove ``<path>`` elements that have no ``d`` attribute.

    :param svg_string: The raw SVG string to clean.
    :returns: The SVG string with invalid paths removed.
    """
    return re.sub(r"<path(?![^>]* d=)[^>]*/>", "", svg_string)


def autoscale_svg(raw_svg: str, target_width: float) -> str:
    """Autoscale an SVG string to the target width while maintaining aspect ratio.

    :param raw_svg: The raw SVG string to scale.
    :param target_width: The desired width for the output SVG.
    :returns: The modified SVG as a string.
    """
    from svgpathtools import svg2paths2

    clean_svg = remove_invalid_paths(raw_svg)

    # Register all namespace declarations so ET preserves prefixes on serialisation.
    for prefix, uri in re.findall(r'xmlns:?(\w*)\s*=\s*"([^"]+)"', clean_svg):
        ET.register_namespace(prefix, uri)

    root = ET.fromstring(clean_svg)

    output_tuple = svg2paths2(io.StringIO(clean_svg))
    assert len(output_tuple) == 3
    paths, attributes, _ = output_tuple

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
    width = xmax - xmin + 10
    height = ymax - ymin + 10

    aspect_ratio = height / width
    new_width = target_width
    new_height = target_width * aspect_ratio

    root.attrib["viewBox"] = f"{xmin} {ymin} {width} {height}"
    root.attrib["width"] = str(new_width)
    root.attrib["height"] = str(new_height)

    return ET.tostring(root, encoding="utf-8", xml_declaration=True).decode("utf-8")
