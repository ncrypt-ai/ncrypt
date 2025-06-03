import argparse
from enum import Enum

from ncrypt.models import CRNNv1
from ncrypt.preprocessors import DefaultPreprocessor, EastPreprocessor
from ncrypt.utils import ARTIFACT_DIR, OUT_DIR


class PreProcessors(Enum):
    DEFAULT = DefaultPreprocessor
    EAST = EastPreprocessor


class Models(Enum):
    CRNN_V1 = CRNNv1


class PostProcessors(Enum):
    pass


def run_job(args):
    pass


def get_job_status(args):
    pass


def main():
    parser = argparse.ArgumentParser(description="Ncrypt CLI")
    subparsers = parser.add_subparsers(dest="command", help="Subcommands", required=True)

    # Arguments for the primary parser
    parser.add_argument("--api_key", type=str, required=True, help="The Ncrypt API key")
    parser.add_argument("--artifact_dir", type=str, default=ARTIFACT_DIR, help="The directory name where model artifacts will be stored. Default is ~/.ncrypt")
    parser.add_argument("--out_dir", type=str, default=OUT_DIR, help="The directory where all PDF files are saved/serialized to")

    # Arguments for the run_job parser
    parser_run_job = subparsers.add_parser("run_job", help="Submit an OCR job")

    parser_run_job.add_argument("--input", type=str, required=True, help="The path to the PDF file you wish to process")
    parser_run_job.add_argument("--save", type=bool, default=False, help="Whether the PDF file should be saved after it goes through preprocessing")
    parser_run_job.add_argument("--save_name", type=str, help="The name to use when saving/serializing the preprocessed PDF")

    parser_run_job.add_argument("--model", type=lambda s: Models[s.upper()], choices=list(Models), default="crnn_v1", help="OCR model type")
    parser_run_job.add_argument("--model_args", metavar="KEY=VALUE", nargs="+", help="An arbitrary number of key-value pairs to be provided to the selected model")
    parser_run_job.add_argument("--preprocessor", type=lambda s: PreProcessors[s.upper()], choices=list(PreProcessors), default="default", help="OCR preprocessor type")
    parser_run_job.add_argument("--preprocessor_args", metavar="KEY=VALUE", nargs="+", help="An arbitrary number of key-value pairs to be provided to the selected preprocessor")
    parser_run_job.add_argument("--postprocessor", type=lambda s: PostProcessors[s.upper()], choices=list(PostProcessors), default="default", help="OCR postprocessor type")
    parser_run_job.add_argument("--postprocessor_args", metavar="KEY=VALUE", nargs="+", help="An arbitrary number of key-value pairs to be provided to the selected postprocessor")

    parser_run_job.set_defaults(func=run_job)

    # Arguments for the get_job_status parser
    parser_get_job_status = subparsers.add_parser("get_job_status", help="Get the status of an OCR job that has already been submitted")

    parser_get_job_status.set_defaults(func=get_job_status)


if __name__ == "__main__":
    main()
