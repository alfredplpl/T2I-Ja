from ._script_path import add_repo_root_to_path

add_repo_root_to_path()

from scripts.caption_florence_jsonl import main


if __name__ == "__main__":
    main()
