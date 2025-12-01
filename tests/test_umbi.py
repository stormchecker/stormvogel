import stormvogel.umbi as svu
import stormvogel
import umbi.ats

def _export_and_import(model):
    ats = svu.translate_to_umbi(model)
    umbi.io.write_ats(ats, "tmp.umb")
    loaded_ats = umbi.io.read_ats("tmp.umb")
    assert loaded_ats == ats

def test_ctmc():

    model = stormvogel.examples.create_nuclear_fusion_ctmc()
    ats = svu.translate_to_umbi(model)
    umbi.io.write_ats(ats, "fusion.umb")
    loaded_ats = umbi.io.read_ats("fusion.umb")
    assert loaded_ats == ats
    loaded_model = svu.translate_to_stormvogel(loaded_ats)
