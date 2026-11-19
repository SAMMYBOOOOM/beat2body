"""Entry point inside a SageMaker training job (copied to the root of the uploaded source).

Unpacks the data archive from the "training" channel and runs `beat2body train` with the job's
hyperparameters (keys are train options without the leading dashes, e.g. {"model": "diffusion"}).
"""
import json
import os
import sys
import tarfile

CHANNEL = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
HPS_FILE = os.path.join(os.environ.get("SM_INPUT_CONFIG_DIR", "/opt/ml/input/config"), "hyperparameters.json")
DATA = "/tmp/beat2body_data"


def hyperparameters():
    with open(HPS_FILE) as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        if k.startswith("sagemaker_") or k.startswith("_"):
            continue
        try:
            v = json.loads(v) if isinstance(v, str) else v
        except json.JSONDecodeError:
            pass
        out[k] = v
    return out


def main():
    archives = [f for f in os.listdir(CHANNEL) if f.endswith((".tar", ".tar.gz"))]
    if len(archives) != 1:
        sys.exit(f"expected one data archive in {CHANNEL}, found {archives}")
    os.makedirs(DATA, exist_ok=True)
    with tarfile.open(os.path.join(CHANNEL, archives[0])) as t:
        t.extractall(DATA, filter="data")
    argv = ["--data-root", os.path.join(DATA, "2d"), "train"]
    for k, v in hyperparameters().items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            argv += [flag] if v else []
        else:
            argv += [flag, str(v)]
    print("[entry] python -m beat2body " + " ".join(argv), flush=True)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from beat2body.cli import main as cli_main
    cli_main(argv)


if __name__ == "__main__":
    main()
