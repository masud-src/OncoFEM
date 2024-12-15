#!/bin/bash

# Function to install packages for Linux
install_linux() {
    echo "Detected Linux OS. Proceeding with installation."
    sudo apt update
    sudo apt upgrade -y
    sudo apt install -y build-essential python3-pytest gmsh libz-dev cmake libeigen3-dev libgmp-dev libgmp3-dev libmpfr-dev libboost-all-dev python3-pip git
}

# Function to install packages for macOS
install_macos() {
    echo "Detected macOS. Proceeding with installation."
    # Check if Homebrew is installed
    if ! command -v brew &>/dev/null; then
        echo "Homebrew is not installed. Installing Homebrew."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        echo "Homebrew installed successfully."
    fi

    brew update
    brew upgrade
    brew install \
        cmake \
        eigen \
        gmp \
        mpfr \
        boost \
        gmsh \
        python3 \
        git

    # Ensure command-line tools are installed
    xcode-select --install 2>/dev/null || echo "Command line tools already installed."

    # Install Python testing tool
    pip3 install --upgrade pytest
}

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    install_linux
elif [[ "$OSTYPE" == "darwin"* ]]; then
    install_macos
else
    echo "Unsupported operating system: $OSTYPE"
    exit 1
fi


OS=$(uname -s)
ARCH=$(uname -m)

if [[ "$OS" == "Linux" ]]; then
    OS="Linux"
elif [[ "$OS" == "Darwin" ]]; then
    OS="MacOSX"
elif [[ "$OS" =~ MINGW64 || "$OS" =~ MSYS ]]; then
    OS="Windows"
else
    echo "Unsupported OS: $OS"
    exit 1
fi

if [[ "$OS" == "MacOSX" && "$ARCH" == "arm64" ]]; then
    ARCH="arm64"
elif [[ "$ARCH" == "x86_64" ]]; then
    ARCH="x86_64"
else
    echo "Unsupported architecture: $ARCH"
    exit 1
fi

cd ..
URL="https://repo.anaconda.com/archive/Anaconda3-latest-$OS-$ARCH.sh"
echo "Downloading Anaconda installer from: $URL"
curl -o AnacondaInstaller.sh "$URL"
bash Anaconda.sh -b -p $HOME/anaconda3
eval "$($HOME/anaconda3/bin/conda shell.bash hook)"
conda init
cd OncoFEM
echo "Installation of prerequisites is completed successfully!"
echo "Some changes require the terminal to be restarted."
echo "Please close and reopen your terminal to apply the changes."
exit 0
