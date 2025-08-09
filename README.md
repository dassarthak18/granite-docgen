# Automated Documentation Generator powered by Langchain and IBM Granite Code

Automated documentation generator for git repositories using [**Langchain**](https://www.langchain.com/) and [``ibm-granite/granite-8b-code-instruct-128k``](https://huggingface.co/ibm-granite/granite-8b-code-instruct-128k).

## System Requirements

## Setup

All the dependencies are listed in ``requirements.txt``.

The setup script assumes the following:
* all system components are up-to-date, and
* both ``python3``and``python3-pip`` are already installed.

You will require a Hugging Face user access token with ``READ`` permissions to run the script. The Hugging Face token is a one time thing and is only required for the script to set up the model for docgen. Once you have the token, simply clone this repo and run ``setup.sh``:

```bash
git clone https://github.com/dassarthak18/granite-docgen.git
chmod u+x setup.sh
./setup.sh <HF_TOKEN>
```

## Usage

To use the documentation generator, run the ``run_docgen.sh`` script:

```bash
chmod u+x run_docgen.sh
./run_docgen.sh <REPO_URL> <BRANCH_OR_COMMIT> <GITHUB_TOKEN or 'none'> [USE_GPU]
```

``GITHUB_TOKEN`` is the personal access token required to access private repositories in GitHub. It is not required for public repos. ``USE_GPU`` flag is set to 1 to enable CUDA usage and to 0 to run the entire inference session on CPU.

## TO-DO

- [ ] Make it so that prompt is stored in ``prompt.txt``.
- [ ] Test it out on Google Colab with ``granite-3b-code-instruct-2k``.
