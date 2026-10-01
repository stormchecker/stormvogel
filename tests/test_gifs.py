import base64
import shutil
import subprocess
import sys

import IPython.display as ipd
import pytest

pytest.importorskip("imageio")
Image = pytest.importorskip("PIL.Image")

from stormvogel.extensions.gifs import embed_gif


def test_embed_gif_includes_image_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    filename = tmp_path / "animation.gif"
    frames = [Image.new("RGB", (2, 2), color) for color in ("red", "blue")]
    frames[0].save(filename, save_all=True, append_images=frames[1:], duration=100)
    displayed = []
    monkeypatch.setattr(ipd, "display", displayed.append)

    embed_gif(str(filename))

    assert len(displayed) == 1
    html = displayed[0]._repr_html_()
    encoded = html.split("data:image/gif;base64,", 1)[1].split('"', 1)[0]
    assert base64.b64decode(encoded) == filename.read_bytes()
    assert not (tmp_path / "GIF.html").exists()


def test_embed_gif_survives_sphinx_build(tmp_path, monkeypatch):
    pytest.importorskip("nbsphinx")
    nbformat = pytest.importorskip("nbformat")
    if shutil.which("pandoc") is None:
        pytest.skip("pandoc is required for nbsphinx")

    filename = tmp_path / "animation.gif"
    frames = [Image.new("RGB", (2, 2), color) for color in ("red", "blue")]
    frames[0].save(filename, save_all=True, append_images=frames[1:], duration=100)
    displayed = []
    monkeypatch.setattr(ipd, "display", displayed.append)
    embed_gif(str(filename))
    html = displayed[0]._repr_html_()

    source = tmp_path / "source"
    source.mkdir()
    (source / "conf.py").write_text(
        'extensions = ["nbsphinx"]\nnbsphinx_execute = "never"\n'
        'master_doc = "index"\n'
    )
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                outputs=[
                    nbformat.v4.new_output("display_data", data={"text/html": html})
                ]
            )
        ]
    )
    nbformat.write(notebook, source / "index.ipynb")
    # The build must not need access to the original GIF.
    filename.unlink()
    output = tmp_path / "html"
    subprocess.run(
        [sys.executable, "-m", "sphinx", "-b", "html", str(source), str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert html in (output / "index.html").read_text()
