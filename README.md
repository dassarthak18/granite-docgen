# Documentation Generator powered by IBM Granite Code

Automated documentation generator for git repositories using an 8-bit quantized version of [``ibm-granite/granite-8b-code-instruct-128k``](https://huggingface.co/ibm-granite/granite-8b-code-instruct-128k) and [``bigcode/starcoder``](https://huggingface.co/bigcode/starcoder) for tokenization.

## Setup

All the dependencies are listed in ``requirements.txt``.

The setup script assumes the following:
* all system components are upgraded, and
* both ``python3``and``python3-pip`` are already installed.

You will require a Hugging Face user access token with ``READ`` permissions to run the script. ``bigcode/starcoder`` requires accepting their license agreement so make sure to do that before generating a token. The Hugging Face token is a one time thing and is only required for the script to set up the models for docgen.

Once these configurations are ensured, simply clone this repo and run ``setup.sh``:

```bash
git clone https://github.com/dassarthak18/granite-docgen.git
chmod u+x setup.sh
./setup.sh <HF_TOKEN>
```
