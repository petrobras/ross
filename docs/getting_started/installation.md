# Installation

(introduction)=

## Install Python

The first step is to install Python. Since ROSS requires several packages to be installed besides Python, such as
numpy and scipy, we recommend installing [miniforge](https://conda-forge.org/download/) or [Anaconda](https://docs.anaconda.com/free/anaconda/index.html) (version 3.9 or higher) which is a
scientific Python distribution that aims to simplify package management and deployment. It contains Python and a large
number of packages that are commonly used.
Alternatively, you may refer to the [Python website](http://www.python.org/).
ROSS code is tested in Python 3.9 and higher.

## Install ROSS

Using the terminal (or the Anaconda prompt if on Windows) you can install the latest release version with:

```{code-block}
pip install ross-rotordynamics
```

Alternatively, you can install the development version from GitHub:

```{code-block}
pip install git+https://github.com/petrobras/ross.git
```

## AI assistance

Need help building your rotor model or running an analysis? ROSS supports
AI-assisted workflows in two ways: an agent skill for AI coding agents, and
ROSS GPT, a chat assistant in your browser.

### In your coding agent

ROSS ships with an agent skill — a set of concise rotordynamics recipes in the
[Agent Skills](https://agentskills.io) open standard that teaches AI coding
agents how to build rotor models and run analyses with ROSS. After installing
ROSS, install the skill with:

```{code-block}
ross-install-skill
```

This detects the AI coding agents on your machine (Claude Code, GitHub
Copilot, Cursor, Codex) and copies the skill to each one's personal skills
directory. Useful variations:

```{code-block}
ross-install-skill --project          # install into the current project (shared with your team)
ross-install-skill --agent claude     # install for a specific agent only
ross-install-skill --uninstall        # remove the skill
```

Once installed, the skill activates automatically whenever you ask your agent
about rotordynamics with ROSS — for example, "create a rotor with 6 shaft
elements, 2 disks and 2 bearings, then plot the Campbell diagram". In Claude
Code you can also invoke it explicitly with the `/ross` slash command.

The skill is a snapshot of the recipes for the installed ROSS version, so
re-run `ross-install-skill` after upgrading ROSS.

### In your browser: ROSS GPT

Meet [**ROSS GPT**](https://chatgpt.com/g/g-6a0776b675588191a111daf172ecfcfe-ross-gpt-2-0), a virtual assistant trained specifically for the ROSS package. You can:

- Generate rotor models in Python with just a description.
- Run and interpret modal analysis, Campbell diagrams, and more.
- Understand technical aspects of ROSS elements like ShaftElement, DiskElement, BearingElement, etc.

👉 [Click here to start using ROSS GPT](https://chatgpt.com/g/g-6a0776b675588191a111daf172ecfcfe-ross-gpt-2-0).
