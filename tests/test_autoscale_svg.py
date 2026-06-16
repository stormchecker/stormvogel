import xml.etree.ElementTree as ET

import pytest

pytest.importorskip("svgpathtools")

from stormvogel.autoscale_svg import autoscale_svg, remove_invalid_paths

SVG_NS = "http://www.w3.org/2000/svg"

SAMPLE_SVG = """\
<?xml version="1.0" encoding="utf-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="800" height="600">
  <path d="M 100 100 L 200 200" stroke="black" fill="none"/>
  <path d="M 150 150 L 300 300" stroke="black" fill="none"/>
  <path stroke="none" fill="red"/>
</svg>"""


def test_remove_invalid_paths_strips_paths_without_d():
    result = remove_invalid_paths(SAMPLE_SVG)
    assert '<path stroke="none"' not in result


def test_remove_invalid_paths_keeps_valid_paths():
    result = remove_invalid_paths(SAMPLE_SVG)
    assert 'd="M 100 100 L 200 200"' in result
    assert 'd="M 150 150 L 300 300"' in result


def test_autoscale_svg_produces_valid_xml():
    result = autoscale_svg(SAMPLE_SVG, target_width=400)
    root = ET.fromstring(
        result.split("\n", 1)[1] if result.startswith("<?xml") else result
    )
    assert root.tag == f"{{{SVG_NS}}}svg"


def test_autoscale_svg_sets_target_width():
    result = autoscale_svg(SAMPLE_SVG, target_width=400)
    root = ET.fromstring(
        result.split("\n", 1)[1] if result.startswith("<?xml") else result
    )
    assert float(root.attrib["width"]) == 400.0


def test_autoscale_svg_preserves_aspect_ratio():
    result = autoscale_svg(SAMPLE_SVG, target_width=400)
    root = ET.fromstring(
        result.split("\n", 1)[1] if result.startswith("<?xml") else result
    )
    vb = [float(v) for v in root.attrib["viewBox"].split()]
    vb_width, vb_height = vb[2], vb[3]
    expected_height = 400.0 * (vb_height / vb_width)
    assert float(root.attrib["height"]) == pytest.approx(expected_height)


def test_autoscale_svg_has_xml_declaration():
    result = autoscale_svg(SAMPLE_SVG, target_width=400)
    assert result.startswith("<?xml")


def test_autoscale_svg_preserves_svg_namespace():
    result = autoscale_svg(SAMPLE_SVG, target_width=400)
    assert 'xmlns="http://www.w3.org/2000/svg"' in result
