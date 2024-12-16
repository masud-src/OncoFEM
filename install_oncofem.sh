#!/bin/bash

conda env create -f oncofem.yaml

if [[ -z "${ONCOFEM_DIR}" ]]; then
    ONCOFEM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

USER_HOME="$HOME"

add_to_path_unix() {
    if ! grep -q "export ONCOFEM=" ~/.bashrc; then
        echo "export ONCOFEM=$ONCOFEM_DIR" >> ~/.bashrc
        echo "ONCOFEM has been added to your PATH."
        echo "Please run 'source ~/.bashrc' to apply the changes."
    else
        echo "ONCOFEM is already set in your PATH."
    fi
}

add_to_path_macos() {
    if ! grep -q "export ONCOFEM=" ~/.zshrc; then
        echo "export ONCOFEM=$ONCOFEM_DIR" >> ~/.zshrc
        echo 'export PATH="$ONCOFEM/bin:$PATH"' >> ~/.zshrc
        echo "ONCOFEM has been added to your PATH."
        echo "Please run 'source ~/.zshrc' to apply the changes."
    else
        echo "ONCOFEM is already set in your PATH."
    fi
}

add_to_path_windows() {
    local script_file="$HOME/install_oncofem.bat"
    if ! grep -q "setx PATH" "$script_file" 2>/dev/null; then
        echo "@echo off" > "$script_file"
        echo "setx PATH \"%PATH%;$ONCOFEM_DIR\"" >> "$script_file"
        echo "ONCOFEM has been added to your PATH."
        echo "Please restart your command prompt to apply the changes."
    else
        echo "ONCOFEM is already set in your PATH."
    fi
}

create_config_file(){
    CONFIG_FILE="$ONCOFEM_DIR/config.ini"
    {
        echo "[directories]"
        echo "STUDIES_DIR: $USER_HOME/studies/"
    } > "$CONFIG_FILE"
    echo "Config file created."
}

case "$(uname -s)" in
    Linux*)     add_to_path_unix ;;
    Darwin*)    add_to_path_macos ;;
    *)          echo "Unsupported OS. Please add the ONCOFEM directory to your PATH manually." ;;
esac

if [[ "$OS" == "Windows_NT" ]]; then
    add_to_path_windows
fi

create_config_file
conda init
conda activate oncofem
python3 -m pip install .
