{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  # Specify the Python version you want to use
  buildInputs = with pkgs; [
    python312
    poetry
  ];

  # Environment setup
  shellHook = ''
    poetry install --no-root

    # Extract OPENAI_API_KEY from .env file and export it
    if [ -f .env ]; then
      export OPENAI_API_KEY=$(grep -E '^OPENAI_API_KEY=' .env | cut -d '=' -f2)
    fi
  '';
}