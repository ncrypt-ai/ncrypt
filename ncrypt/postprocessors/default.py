from typing import Dict, List, Tuple
from ncrypt.utils._constants import DEFAULT_VOCAB
from ncrypt.utils._interfaces import PostProcessor, TextRegion
from ncrypt.pdf import PDFFile
from ncrypt.utils._types import JobResults


class DefaultPostprocessor(PostProcessor):
    def __init__(self, vocabulary: Dict[str, int] = DEFAULT_VOCAB):
        """
        Initializes the DefaultPostprocessor instance.
        """
        self.vocabulary = vocabulary

    @property
    def vocab(self):
        return self.vocabulary

    def char_to_int(self, char: str) -> int:
        return self.vocab[char]

    def int_to_char(self, idx: int) -> str:
        idx_map = {self.vocab[i]: i for i in range(len(self.vocab))}

        return idx_map[idx]

    def process(self, results: JobResults, file: PDFFile | None = None, page_ids: List[str] = [], job_ids: List[List[str]] = [], text_regions: List[List[TextRegion]] | None = None) -> Tuple[PDFFile, list[list[str]]]:
        if results.get("status", 500) != 200:
            raise ValueError("Not all of the jobs have completed yet.")
        
        if not file and (not text_regions or not page_ids or not job_ids):
            raise RuntimeError("The file and text_regions parameters are both None. At least one must be provided.")
        
        if file:
            text_regions = file.get_text_regions()
            all_ids: Dict[str, List[str] | List[List[str]]] = file.get_page_ids()

            page_ids: List[str] = all_ids.get("page_ids", [])
            job_ids: List[List[str]] = all_ids.get("job_ids", [])


        for page_num in range(len(page_ids)):
            page = page_ids[page_num]
            jobs = job_ids[page_num]
            regions = text_regions[page_num]

            if sum([len(region.bounding_boxes) for region in regions]) != len(jobs):
                raise RuntimeError("The number of jobs does not match the number of bounding boxes in the provided text_regions")
            
            processed: int = 0

            for region in regions:
                num_bboxes = len(region.bounding_boxes)
                relevant_jobs = jobs[processed:num_bboxes]
                processed += num_bboxes

                # TODO: Call beam search for that text snippet

            """
            For each region, get its number of bounding boxes
            Grab that many results out of the jobs array
            Pass the whole thing to a beam search
            """

            for job in jobs:
                result = results["pages"][page][job]
                

        while job_ids:
            jobs = job_ids.pop(0)
            
        return

    def __beam_search(self):
        pass
