from ._script_path import add_repo_root_to_path

add_repo_root_to_path()

from scripts.build_pixart_sigma_pipeline import main


if __name__ == "__main__":
    main()
