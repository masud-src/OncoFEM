#!/bin/bash

ONCOFEM_DIR="/path/to/oncofem"  # Replace with the actual path to ONCOFEM

add_to_path_unix() {
    if [[ ":$PATH:" != *":$ONCOFEM_DIR:"* ]]; then
        echo "export PATH=\"\$PATH:$ONCOFEM_DIR\"" >> ~/.bashrc
        echo "ONCOFEM has been added to your PATH."
        echo "Please run 'source ~/.bashrc' to apply the changes."
    else
        echo "ONCOFEM is already in your PATH."
    fi
}

add_to_path_windows() {
    local script_file="$HOME/add_oncofem_to_path.bat"
    echo "@echo off" > "$script_file"
    echo "setx PATH \"%PATH%;$ONCOFEM_DIR\"" >> "$script_file"
    echo "ONCOFEM has been added to your PATH."
    echo "Please restart your command prompt to apply the changes."
}

case "$(uname -s)" in
    Linux*)     add_to_path_unix ;;
    Darwin*)    add_to_path_unix ;;
    *)          echo "Unsupported OS. Please add the ONCOFEM directory to your PATH manually." ;;
esac

if [[ "$OS" == "Windows_NT" ]]; then
    add_to_path_windows
fi
