from ._script_path import add_repo_root_to_path

add_repo_root_to_path()

from scripts.train_basic_t2i import main


if __name__ == "__main__":
    main()
