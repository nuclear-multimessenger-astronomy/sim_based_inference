import json

import pandas as pd

from nmma.mlmodel.dataprocessing import (
    gen_append_filler,
    get_names,
    json_to_df,
    pad_the_data,
)


def test_get_names_builds_expected_paths():
    assert get_names("data", "sample", 2, 2) == [
        "data/sample_batch_2/sample_2_0.json",
        "data/sample_batch_2/sample_2_1.json",
    ]


def test_json_to_df_unpacks_photometry(tmp_path):
    data_file = tmp_path / "lightcurve.json"
    data_file.write_text(
        json.dumps(
            {
                "ztfg": [[1.0, 20.0, 0.1], [2.0, 22.0, 0.2]],
                "ztfr": [[1.0, 21.0, 0.1], [2.0, 22.0, 0.2]],
            }
        )
    )

    result = json_to_df(
        [str(data_file)],
        1,
        bands=["ztfg", "ztfr"],
    )[0]

    assert list(result.columns) == ["t", "ztfg", "ztfr", "num_detections", "sim_id"]
    assert result["num_detections"].tolist() == [2, 2]
    assert result["sim_id"].tolist() == [0, 0]


def test_pad_the_data_adds_time_and_detection_fillers():
    actual = pd.DataFrame(
        {
            "t": [44242.50021937881, 44242.75021937881],
            "ztfg": [20.0, 21.0],
        }
    )

    padded = pad_the_data(
        actual,
        ["t", "ztfg"],
        desired_count=4,
        filler_time_step=0.25,
        filler_data=22.0,
    )

    assert padded["t"].tolist() == [0.0, 0.25, 0.5, 0.75]
    assert padded["ztfg"].tolist() == [22.0, 22.0, 20.0, 21.0]


def test_gen_append_filler_uses_requested_count():
    filler = gen_append_filler(["t", "ztfg"], 22.0, 1.0, 3, step=0.25)

    assert filler["t"].tolist() == [1.0, 1.25, 1.5]
    assert filler["ztfg"].tolist() == [22.0, 22.0, 22.0]