IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".heic", ".heif"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS

EXIF_DATE_FIELDS = ["DateTimeOriginal", "CreateDate", "FileModifyDate"]

FORBIDDEN_FILENAME_CHARS = '<>:"/\\|?*'
