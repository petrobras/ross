=================
Git configuration
=================

In order to use git to contribute to ROSS project, follow the steps bellow:

----------------------------------------
Step 1: Make you own copy (fork) of ROSS
----------------------------------------
From the command line:

::
    git clone https://github.com/ross-rotordynamics/ross.git
    cd ross
    git remote add upstream https://github.com/ross-rotordynamics/ross.git

-----------------------------------------
Step 2: Keep in sync with changes in Ross
-----------------------------------------

::
    git config branch.master.remote upstream
    git config branch.master.merge refs/heads/master

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

-------------------------------------
Step 5: Push changes to the main repo
-------------------------------------
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
