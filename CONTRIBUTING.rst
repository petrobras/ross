Feedback and Contribution
-------------------------
We welcome any contribution via `ross's issue tracker <https://github.com/ross-rotordynamics/ross/issues>`_.
These include bug reports, problems on the documentation, feedback, enhancement proposals etc.
The issue tracker can also be used for questions and further information, since the project does not use a mailing list.

Version-control system: Git
---------------------------
Git is a version-control system (VCS) for tracking changes in code during software development.
In order to download the ross source code and contribute to its development,
you need Git installed in your machine. Refer to the `Git website
<https://git-scm.com/>`_ and follow the instructions to download and install it.
Once you have Git installed, you will be able to follow the instructions in :ref:`git-configuration`,
which explains how to download and contribute to ross.

Code style: Black
-----------------
Code formatting is done with `Black <https://black.readthedocs.io/en/stable/>`_, which is the *"uncompromising Python
code formatter"*. You can configure your development environment to use Black before a commit. More information on how
to set this is given at `Black's documentation<https://black.readthedocs.io/en/stable/editor_integration.html>`_.

Integrated development environment: PyCharm
-------------------------------------------

The ross development team adopted PyCharm as integrated development environment (IDE).
You don't need PyCharm to run or contribute to ross, as you can choose your preferred IDE or
even no IDE at all. But in case you want to use PyCharm, go to the `PyCharm website
<https://www.jetbrains.com/pycharm/>`_ to download and install it.

How to contribute to ROSS using git
===================================

.. _git-configuration:

In order to use git to contribute to ROSS project, follow the steps bellow:
*For Windows users: commands provided here can be executed using Git Bash instead of Git GUI.*

----------------------------------------
Step 1: Make you own copy (fork) of ROSS
----------------------------------------
Go to https://github.com/ross-rotordynamics/ross
In the top-right corner of the page, click Fork, to fork it to your GitHub account.

From the command line:

::

    git clone https://github.com/your-user-name/ross.git
    cd ross
    git remote add upstream https://github.com/ross-rotordynamics/ross.git


-----------------------------------------
Step 2: Keep in sync with changes in Ross
-----------------------------------------

Setup your local repository so it pulls from upstream by default:

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

---------------------------------
Step 3: Make a new feature branch
---------------------------------

::

    git fetch upstream
    git checkout -b my-new-feature upstream/master

-------------------------------------------
Step 4: Push changes to your git repository
-------------------------------------------
After a complete working set of related changes are made:

::

    git add modified_file
    git commit
    git push origin my-new-feature

The following blog posts have some good information on how to write commit messages:

`A Note About Git Commit Messages <https://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html>`_

`On commit messages <https://who-t.blogspot.com/2009/12/on-commit-messages.html>`_

-------------------------------------
Step 5: Push changes to the main repo
-------------------------------------

^^^^^^^^^^^^^^^^
For contributors
^^^^^^^^^^^^^^^^
To create a Pull Request (PR), refer to https://help.github.com/articles/about-pull-requests/

^^^^^^^^^^^^^^^^^^^
For core developers
^^^^^^^^^^^^^^^^^^^
If there are only a few, unrelated commits:

::

    git fetch upstream
    git rebase upstream/master
    git log -p upstream/master..
    git log --oneline --graph
    git push upstream my-feature-branch:master

Otherwise, if all commits are related:

::

    git fetch upstream
    git merge --no-ff upstream/master
    git log -p upstream/master..
    git log --oneline --graph
    git push upstream my-feature-branch:master
