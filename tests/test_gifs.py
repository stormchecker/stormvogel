import base64

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
    data, _ = displayed[0]._repr_mimebundle_()
    assert base64.b64decode(data["image/gif"]) == filename.read_bytes()
    assert "text/html" not in data
    assert not (tmp_path / "GIF.html").exists()
