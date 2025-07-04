import os

BUCKET_NAME = "ncrypt-files"
REGION = "us-east-2"
SERVICE_NAME = "ncrypt"
USER_NAME = "ncrypt"
SEARCH_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "search_client.zip")
EMBED_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "embed_client.zip")
SIMILARITY_THRESHOLD = 0.80
SCALE_FACTOR = 1000
AUDIO_EXTENSIONS = [
    "flac", "m4a", "mp3", "mp4", "mpeg", "mpga", "oga", "ogg", "wav", "webm"
]
IMAGE_EXTENSIONS = [
    "bmp", "gif", "hdr", "jpeg", "jpg", "pic", "pgm", "png", "tif", "tiff", "webp", "wmf"
]
IMAGE_EMBED_MODEL = "facebook/dinov2-small"
PLAINTEXT_EXTENSIONS = [
    "asm", "bat", "c", "cc", "cfg", "clj", "conf", "cpp", "cs", "csv",
    "css", "cxx", "diff", "ini", "java", "js", "json", "jsx", "log", "lua",
    "md", "mjs", "ps1", "py", "r", "rb", "rst", "sass", "scala", "scss",
    "sh", "sql", "srt", "ts", "tsx", "txt", "vue", "yml", "yaml", "toml",
    "xml", "xsl", "sql", "conf", "cfg"
]
MARKUP_EXTENSIONS = ["html", "htm", "svg", "xhtml"]
PDF_EXTENSIONS = ["pdf"]
OFFICE_DOCX_EXTENSIONS = ["docx", "docm", "dotx"]
OFFICE_PPTX_EXTENSIONS = ["pptx", "pptm"]
OFFICE_XLSX_EXTENSIONS = ["xlsx", "xlsm"]
MAC_EXTENSIONS = ["pages", "numbers", "key"]
EMAIL_EXTENSIONS = ["eml", "msg"]
RTF_EXTENSIONS = ["rtf", "wpd", "wps"]
TEXT_EXTENSIONS = PLAINTEXT_EXTENSIONS + PDF_EXTENSIONS + MARKUP_EXTENSIONS + OFFICE_DOCX_EXTENSIONS + OFFICE_PPTX_EXTENSIONS + OFFICE_XLSX_EXTENSIONS + MAC_EXTENSIONS + EMAIL_EXTENSIONS + RTF_EXTENSIONS
TEXT_EMBED_MODEL = "facebook/bart-base"
TEXT_SUMMARY_MODEL = "Falconsai/text_summarization"
