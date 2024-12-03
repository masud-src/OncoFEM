#!/bin/bash

ONCOFEM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_HOME="$HOME"

add_to_path_unix() {
        echo "export ONCOFEM=$ONCOFEM_DIR" >> ~/.bashrc
        echo "ONCOFEM has been added to your PATH."
        echo "Please run 'source ~/.bashrc' to apply the changes."
}

add_to_path_windows() {
    local script_file="$HOME/set_config.bat"
    echo "@echo off" > "$script_file"
    echo "setx PATH \"%PATH%;$ONCOFEM_DIR\"" >> "$script_file"
    echo "ONCOFEM has been added to your PATH."
    echo "Please restart your command prompt to apply the changes."
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
    Darwin*)    add_to_path_unix ;;
    *)          echo "Unsupported OS. Please add the ONCOFEM directory to your PATH manually." ;;
esac

if [[ "$OS" == "Windows_NT" ]]; then
    add_to_path_windows
fi

create_config_file
