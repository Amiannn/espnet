import argparse
import os
import json

parser = argparse.ArgumentParser("Clarity")
parser.add_argument(
    "--clarity_root",
    type=str,
    help="Path to Clarity Challenge root folder "
    "(Folder containing train, dev and metadata dirs)",
)


def prepare_data(clarity_root):
    output_folder = "./data"

    ids = {"train": set(), "dev": set()}

    for ds_split in ids.keys():
        metafile = os.path.join(
            clarity_root, "metadata", "scenes.{}.json".format(ds_split)
        )
        with open(metafile, "r") as f:
            metadata = json.load(f)
        for ex in metadata:
            ids[ds_split].add(ex["scene"])

    # create wav.scp
    for ds_split in ids.keys():
        with open(os.path.join(output_folder, ds_split, "wav.scp"), "w") as f:
            for ex_id in ids[ds_split]:
                array_files = " ".join(
                    [
                        os.path.join(
                            clarity_root,
                            ds_split,
                            "scenes",
                            "{}_mixed_CH{}.wav".format(ex_id, idx),
                        )
                        for idx in range(1, 4)
                    ]
                )
                assert all([os.path.exists(x) for x in array_files]), (
                    "Some file do not seem to exist, "
                    "please check your root folder, is the path correct ?"
                )
                f.write("{} sox -M {} -c 6 -t wav - |\n".format(ex_id, array_files))

        with open(os.path.join(output_folder, ds_split, "noise1.scp"), "w") as f:
            for ex_id in ids[ds_split]:
                array_file = os.path.join(
                    clarity_root,
                    ds_split,
                    "scenes",
                    "{}_interferer_CH1.wav".format(ex_id),
                )
                assert os.path.exists(array_file), (
                    "Some file do not seem to exist, "
                    "please check your root folder, is the path correct ?"
                )
                f.write("{} sox {} remix 1 -t wav - |\n".format(ex_id, array_file))

        with open(os.path.join(output_folder, ds_split, "spk1.scp"), "w") as f:
            for ex_id in ids[ds_split]:
                array_file = os.path.join(
                    clarity_root, ds_split, "scenes", "{}_target_CH1.wav".format(ex_id)
                )
                assert os.path.exists(array_file), (
                    "Some file do not seem to exist, "
                    "please check your root folder, is the path correct ?"
                )
                f.write("{} sox {} remix 1 -t wav - |\n".format(ex_id, array_file))

        with open(os.path.join(output_folder, ds_split, "text.scp"), "w") as f:
            for ex_id in ids[ds_split]:
                f.write("{} dummy\n".format(ex_id))

        with open(os.path.join(output_folder, ds_split, "spk2utt.scp"), "w") as f:
            for ex_id in ids[ds_split]:
                f.write("{} dummy\n".format(ex_id))


if __name__ == "__main__":
    args = parser.parse_args()
    prepare_data(args.clarity_root)
