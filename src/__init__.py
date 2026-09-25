# Importing config first, from the package itself, guarantees HF_HOME and the
# .env token are set before any entry point imports huggingface_hub & co.
from . import config  # noqa: F401
