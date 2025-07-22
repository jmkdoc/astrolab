# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Pile dataset. (Uncopyrighted subset)"""

import json

import datasets


_CITATION = """\
@misc{gao2020pile,
      title={The Pile: An 800GB Dataset of Diverse Text for Language Modeling},
      author={Leo Gao and Stella Biderman and Sid Black and Laurence Golding and Travis Hoppe and Charles Foster and Jason Phang and Horace He and Anish Thite and Noa Nabeshima and Shawn Presser and Connor Leahy},
      year={2020},
      eprint={2101.00027},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

_DESCRIPTION = """\
The Pile is a 825 GiB diverse, open source language modelling data set that consists of 22 smaller, high-quality
datasets combined together.
"""

_HOMEPAGE = "https://pile.eleuther.ai/"

_LICENSES = {
    "all": "Multiple: see each subset license",
    "enron_emails": "Unknown",
    "europarl": "Unknown",
    "free_law": "Unknown",
    "hacker_news": "Unknown",
    "nih_exporter": "Unknown",
    "pubmed": "Unknown",
    "pubmed_central": "Unknown",
    "ubuntu_irc": "Unknown",
    "uspto": "Unknown",
    "github": "Unknown",
}

#_HOST_URL = "https://the-eye.eu"
_HOST_URL = "https://huggingface.co/datasets/monology/pile-uncopyrighted/resolve/main"
_DATA_URLS = {
    "all": {
        "train": [f"{_HOST_URL}/train/{i:0>2}.jsonl.zst" for i in range(30)],
        "validation": [f"{_HOST_URL}/val.jsonl.zst"],
        "test": [f"{_HOST_URL}/test.jsonl.zst"],
    },
    # "all": {
    #     "train": [f"{_HOST_URL}/public/AI/pile/train/{i:0>2}.jsonl.zst" for i in range(30)],
    #     "validation": [f"{_HOST_URL}/public/AI/pile/val.jsonl.zst"],
    #     "test": [f"{_HOST_URL}/public/AI/pile/test.jsonl.zst"],
    # },
    "enron_emails": "https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz",
    "europarl": f"{_HOST_URL}/public/AI/pile_preliminary_components/EuroParliamentProceedings_1996_2011.jsonl.zst",
    "free_law": f"{_HOST_URL}/public/AI/pile_preliminary_components/FreeLaw_Opinions.jsonl.zst",
    "hacker_news": f"{_HOST_URL}/public/AI/pile_preliminary_components/hn.tar.gz",
    "nih_exporter": f"{_HOST_URL}/public/AI/pile_preliminary_components/NIH_ExPORTER_awarded_grant_text.jsonl.zst",
    "pubmed": f"{_HOST_URL}/public/AI/pile_preliminary_components/PUBMED_title_abstracts_2019_baseline.jsonl.zst",
    "pubmed_central": f"{_HOST_URL}/public/AI/pile_preliminary_components/PMC_extracts.tar.gz",
    "ubuntu_irc": f"{_HOST_URL}/public/AI/pile_preliminary_components/ubuntu_irc_until_2020_9_1.jsonl.zst",
    "uspto": f"{_HOST_URL}/public/AI/pile_preliminary_components/pile_uspto.tar",
    "github": f"{_HOST_URL}/public/AI/pile_preliminary_components/github.tar",
}

_FEATURES = {
    "all": datasets.Features(
        {
            "text": datasets.Value("string"),
            "meta": {"pile_set_name": datasets.Value("string")},
        }
    ),
    "enron_emails": datasets.Features(
        {
            "text": datasets.Value("string"),
            "meta": datasets.Value("string"),
        }
    ),
    "europarl": datasets.Features(
        {
            "text": datasets.Value("string"),
            "meta": datasets.Value("string"),
        }
    ),
    "free_law": datasets.Features(
        {
            "text": datasets.Value("string"),
            "meta": datasets.Value("string"),
        }
    ),
    "hacker_news": datasets.Features(
        {
            "text": datasets.Value("string"),
            "meta": datasets.Value("string"),
        }
    ),
    "nih_exporter": datasets.Features(
        {
            "text": datasets.Value("string"),
            "meta": datasets.Value("string"),
        }
    ),
    "pubmed": datasets.Features(
        {
            "text": datasets.Value("string"),
            "meta": datasets.Value("string"),
        }
    ),
    "pubmed_central": datasets.Features(
        {
            "text": datasets.Value("string"),
            "meta": datasets.Value("string"),
        }
    ),
    "ubuntu_irc": datasets.Features(
        {
            "text": datasets.Value("string"),
            "meta": datasets.Value("string"),
        }
    ),
    "uspto": datasets.Features(
        {
            "text": datasets.Value("string"),
            "meta": datasets.Value("string"),
        }
    ),
    "github": datasets.Features(
        {
            "text": datasets.Value("string"),
            "meta": datasets.Value("string"),
        }
    ),
}


class ThePileConfig(datasets.BuilderConfig):
    """BuilderConfig for The Pile."""

    def __init__(self, *args, subsets, **kwargs):
        """BuilderConfig for The Pile.

        Args:
            subsets (:obj:`List[str]`): List of subsets to load.
            **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(
            *args,
            name="+".join(subsets),
            **kwargs,
        )
        self.subsets = subsets


class ThePile(datasets.GeneratorBasedBuilder):
    """The Pile dataset."""

    VERSION = datasets.Version("1.1.0")

    BUILDER_CONFIG_CLASS = ThePileConfig
    BUILDER_CONFIGS = [ThePileConfig(subsets=[subset]) for subset in _DATA_URLS]
    DEFAULT_CONFIG_NAME = "all"

    def _info(self):
        """Give information and typings for the dataset."""
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=_FEATURES.get(self.config.name),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSES.get(self.config.name, "Multiple: see each subset license"),
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Return SplitGenerators."""
        if self.config.name == "all":
            data_dir = dl_manager.download(_DATA_URLS[self.config.name])
            return [
                datasets.SplitGenerator(
                    name=split,
                    gen_kwargs={
                        "files": data_dir[split],
                    },
                )
                for split in [datasets.Split.TRAIN, datasets.Split.VALIDATION, datasets.Split.TEST]
            ]
        else:
            data_urls = {subset: _DATA_URLS[subset] for subset in self.config.subsets}
            archive = dl_manager.download(data_urls)
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "files": {
                            subset: dl_manager.iter_archive(archive[subset])
                            if ".tar" in data_urls[subset]
                            else archive[subset]
                            for subset in self.config.subsets
                        },
                    },
                ),
            ]

    def _generate_examples(self, files):
        """Yield examples as (key, example) tuples."""
        key = 0
        if isinstance(files, list):
            import zstandard as zstd

            for path in files:
                with zstd.open(open(path, "rb"), "rt", encoding="utf-8") as f:
                    for row in f:
                        data = json.loads(row)
                        yield key, data
                        key += 1
        else:
            for subset in files:
                if subset in {"europarl", "free_law", "nih_exporter", "pubmed", "ubuntu_irc"}:
                    import zstandard as zstd

                    with zstd.open(open(files[subset], "rb"), "rt", encoding="utf-8") as f:
                        for row in f:
                            data = json.loads(row)
                            yield key, data
                            key += 1
                elif subset in {"enron_emails", "hacker_news", "pubmed_central"}:
                    for path, file in files[subset]:
                        if subset == "enron_emails":
                            meta = {"file": path}
                        else:
                            id_ = path.split("/")[-1].split(".")[0]
                            meta = {"id": id_}
                        text = file.read().decode("utf-8", errors="ignore")  # encoding errors in enron_emails
                        yield key, {
                            "text": text,
                            "meta": str(meta),
                        }
                        key += 1
                elif subset in {"uspto", "github"}:
                    import zstandard as zstd

                    for path, file in files[subset]:
                        with zstd.open(file, "rt", encoding="utf-8") as f:
                            for row in f:
                                data = json.loads(row)
                                yield key, data
                                key += 1


# pile.py (inside ThePile class)

def _generate_examples(self, files):
    """Yield examples as (key, example) tuples."""
    key = 0
    if isinstance(files, list): # This branch is for the "all" config (train/val/test splits)
        import zstandard as zstd

        for path in files:
            with zstd.open(open(path, "rb"), "rt", encoding="utf-8") as f:
                for row in f:
                    try:
                        data = json.loads(row)
                        # --- DEBUG CHECK 1: Data integrity ---
                        if not isinstance(data, dict):
                            print(f"🚨 WARNING: Row {key} from {path} is not a dictionary. Skipping.")
                            continue
                        if "text" not in data or not isinstance(data["text"], str) or not data["text"].strip():
                            print(f"🚨 WARNING: Row {key} from {path} has missing, non-string, or empty 'text' field. Skipping.")
                            continue
                        if "meta" not in data: # Meta field might be optional or structured differently per subset
                            print(f"ℹ️ INFO: Row {key} from {path} is missing 'meta' field.")

                        # Print a snippet of the text
                        if key < 10: # Only print for the first few examples
                            print(f"DEBUG: Example {key} from '{data.get('meta', {}).get('pile_set_name', 'N/A')}' (Length: {len(data['text'])})")
                            print(f"  Text snippet: '{data['text'][:100].replace('\\n', ' ').strip()}'")
                        # --- END DEBUG CHECK 1 ---

                        yield key, data
                        key += 1
                    except json.JSONDecodeError as e:
                        print(f"🚨 WARNING: Failed to parse JSON row {key} in {path}: {row[:100]}... Error: {e}. Skipping.")
                        continue
                    except Exception as e:
                        print(f"🚨 WARNING: Unexpected error processing row {key} in {path}: {e}. Skipping.")
                        continue
    else: # This branch is for individual subset configs
        for subset in files:
            if subset in {"europarl", "free_law", "nih_exporter", "pubmed", "ubuntu_irc"}:
                import zstandard as zstd

                with zstd.open(open(files[subset], "rb"), "rt", encoding="utf-8") as f:
                    for row in f:
                        try:
                            data = json.loads(row)
                            # --- DEBUG CHECK 2: Data integrity ---
                            if not isinstance(data, dict):
                                print(f"🚨 WARNING: Row {key} from {subset} is not a dictionary. Skipping.")
                                continue
                            if "text" not in data or not isinstance(data["text"], str) or not data["text"].strip():
                                print(f"🚨 WARNING: Row {key} from {subset} has missing, non-string, or empty 'text' field. Skipping.")
                                continue
                            # Check meta as per your _FEATURES definition for this subset
                            if "meta" not in data or not isinstance(data["meta"], str): # Assuming meta is string for these
                                 print(f"ℹ️ INFO: Row {key} from {subset} has missing or non-string 'meta' field.")
                            # Print a snippet
                            if key < 10:
                                print(f"DEBUG: Example {key} from '{subset}' (Length: {len(data['text'])})")
                                print(f"  Text snippet: '{data['text'][:100].replace('\\n', ' ').strip()}'")
                            # --- END DEBUG CHECK 2 ---

                            yield key, data
                            key += 1
                        except json.JSONDecodeError as e:
                            print(f"🚨 WARNING: Failed to parse JSON row {key} in {subset}: {row[:100]}... Error: {e}. Skipping.")
                            continue
                        except Exception as e:
                            print(f"🚨 WARNING: Unexpected error processing row {key} in {subset}: {e}. Skipping.")
                            continue
            elif subset in {"enron_emails", "hacker_news", "pubmed_central"}:
                for path, file in files[subset]:
                    try:
                        # Note: errors="ignore" might mask problematic characters.
                        # But direct NaN is unlikely from this.
                        text = file.read().decode("utf-8", errors="ignore")
                        if subset == "enron_emails":
                            meta = {"file": path}
                        else:
                            id_ = path.split("/")[-1].split(".")[0]
                            meta = {"id": id_}

                        # --- DEBUG CHECK 3: Data integrity ---
                        if not isinstance(text, str) or not text.strip():
                            print(f"🚨 WARNING: File '{path}' has empty or non-string content. Skipping.")
                            continue
                        # Print a snippet
                        if key < 10:
                            print(f"DEBUG: Example {key} from '{subset}' (Length: {len(text)})")
                            print(f"  Text snippet: '{text[:100].replace('\\n', ' ').strip()}'")
                        # --- END DEBUG CHECK 3 ---

                        yield key, {
                            "text": text,
                            "meta": str(meta), # meta is expected as string by _FEATURES
                        }
                        key += 1
                    except Exception as e:
                        print(f"🚨 WARNING: Error reading/decoding file '{path}' for subset '{subset}': {e}. Skipping.")
                        continue
            elif subset in {"uspto", "github"}:
                import zstandard as zstd

                for path, file in files[subset]:
                    with zstd.open(file, "rt", encoding="utf-8") as f:
                        for row in f:
                            try:
                                data = json.loads(row)
                                # --- DEBUG CHECK 4: Data integrity ---
                                if not isinstance(data, dict):
                                    print(f"🚨 WARNING: Row {key} from {path} is not a dictionary. Skipping.")
                                    continue
                                if "text" not in data or not isinstance(data["text"], str) or not data["text"].strip():
                                    print(f"🚨 WARNING: Row {key} from {path} has missing, non-string, or empty 'text' field. Skipping.")
                                    continue
                                # Check meta as per your _FEATURES definition for this subset
                                if "meta" not in data or not isinstance(data["meta"], str): # Assuming meta is string for these
                                    print(f"ℹ️ INFO: Row {key} from {path} is missing or non-string 'meta' field.")

                                # Print a snippet
                                if key < 10:
                                    print(f"DEBUG: Example {key} from '{subset}' (Length: {len(data['text'])})")
                                    print(f"  Text snippet: '{data['text'][:100].replace('\\n', ' ').strip()}'")
                                # --- END DEBUG CHECK 4 ---

                                yield key, data
                                key += 1
                            except json.JSONDecodeError as e:
                                print(f"🚨 WARNING: Failed to parse JSON row {key} in {path}: {row[:100]}... Error: {e}. Skipping.")
                                continue
                            except Exception as e:
                                print(f"🚨 WARNING: Unexpected error processing row {key} in {path}: {e}. Skipping.")
                                continue