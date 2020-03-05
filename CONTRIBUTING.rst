Feedback and Contribution
-------------------------
We welcome any contribution via `ross's issue tracker <https://github.com/ross-rotordynamics/ross/issues>`_.
These include bug reports, problems on the documentation, feedback, enhancement proposals etc.
The issue tracker can also be used for questions and further information since the project does not use a mailing list.

Version-control system: Git
---------------------------
Git is a version control system (VCS) for tracking changes in code during software development.
To download the ross source code and contribute to its development,
you need Git installed in your machine. Refer to the `Git website
<https://git-scm.com/>`_ and follow the instructions to download and install it.
Once you have Git installed, you will be able to follow the instructions in `How to contribute to ross using git`_,
which explains how to download and contribute to ross.

Code style: Black
-----------------
To format our code we use `Black <https://black.readthedocs.io/en/stable/>`_, which is the *"uncompromising Python
code formatter"*. You can configure your development environment to use Black before a commit. More information on how
to set this is given at `Black's documentation <https://black.readthedocs.io/en/stable/editor_integration.html>`_.

Tests
-----
We use pytest to test the code. Unit tests are placed in the `~/ross/ross/tests` folder. We also test our docstrings to
assure that the examples are working.
If you want to run all the tests you can do it with (from the `~/ross/ross` folder)::

   $ pytest

Code is only merged to master if tests pass. This is checked by services such as Travis CI and Appveyor, so make sure
tests are passing before pushing your code to github.

Documentation
-------------
We use `sphinx <http://www.sphinx-doc.org/en/master/>`_ to generate the project's documentation. We keep the source
files at ~/ross/docs, and we keep the html files used to build the website in a
`separate repository <https://github.com/ross-rotordynamics/ross-website>`_.
The website tracks the documentation for the released version with the following procedure:

 1. Travis runs the deploy_docs.sh file ('after_success' phase);
 2. The deploy_docs script checks if the branch being updated has the same name as the current released ROSS' version (ross.__version__);
 3. If 2 is True, the script will build the docs for that branch and push to the ross-website repo.

So, if you want to modify the documentation website, modify the source files and then make a pull request
to branch named as the current released version.

If you want to test the documentation locally:

- Clone the ross-website to ``~/ross-website/html``::

    $ git clone https://github.com/ross-rotordynamics/ross-website ~/ross-website/html

- From the docs source directory <~/ross/docs/> run sphinx::

    $ make html BUILDDIR=~/ross-website

- Go to the builddir and run a html server::

    $ cd ~/ross-website/html
    $ python -m http.server

After that you can access your local server (http://0.0.0.0:8000/) and see the generated docs.

Integrated development environment: PyCharm
-------------------------------------------
The ross development team adopted PyCharm as integrated development environment (IDE).
You don't need PyCharm to run or contribute to ross, as you can choose your preferred IDE or
even no IDE at all. But in case you want to use PyCharm, go to the `PyCharm website
<https://www.jetbrains.com/pycharm/>`_ to download and install it.

How to contribute to ross using git
-----------------------------------
.. _git-configuration:

To use git to contribute to ross project, follow the steps below:
*For Windows users: commands provided here can be executed using Git Bash instead of Git GUI.*

Step 1: Make your copy (fork) of ross
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Go to https://github.com/ross-rotordynamics/ross
In the top-right corner of the page, click Fork, to fork it to your GitHub account.

From the command line:

::

    git clone https://github.com/your-user-name/ross.git
    cd ross
    git remote add upstream https://github.com/ross-rotordynamics/ross.git


Step 2: Keep in sync with changes in Ross
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Setup your local repository, so it pulls from upstream by default:

::

    git config branch.master.remote upstream
    git config branch.master.merge refs/heads/master

This can also be done by editing the config file inside your .git directory.
It should look like this:

::

    [core]
            repositoryformatversion = 0
            filemode = true
            bare = false
            logallrefupdates = true
    [remote "origin"]
            url = https://github.com/your-user-name/ross.git
            fetch = +refs/heads/*:refs/remotes/origin/*
    [remote "upstream"]
            url = https://github.com/ross-rotordynamics/ross.git
            fetch = +refs/heads/*:refs/remotes/upstream/*
            fetch = +refs/pull/*/head:refs/remotes/upstream/pr/*
    [branch "master"]
            remote = origin
            merge = refs/heads/master

The part :code:`fetch = +refs/pull/*/head:refs/remotes/upstream/pr/*` will make pull requests available.

Step 3: Make a new feature branch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
::

    git fetch upstream
    git checkout -b my-new-feature upstream/master

Step 4: Push changes to your git repository
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
After a complete working set of related changes are made:

::

    git add modified_file
    git commit
    git push origin my-new-feature

The following blog posts have some good information on how to write commit messages:

`A Note About Git Commit Messages <https://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html>`_

`On commit messages <https://who-t.blogspot.com/2009/12/on-commit-messages.html>`_

Step 5: Push changes to the main repo
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To create a Pull Request (PR), refer to `the github PR guide <https://help.github.com/articles/about-pull-requests/>`_.

Making new releases
-------------------
To make a new release we need only to create a tag using git and push to GitHub:

    $ git tag <version number>

    $ git push upstream --tags

Pushing the new tag to the GitHub repository will start a new build on Travis CI. If all the tests succeed, Travis will
upload the new package to PyPI (see the deploy command on .travis.yml).
