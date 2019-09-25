#!/bin/bash
# Deploy docs to github using Travis CI - based on this gist https://gist.github.com/domenic/ec8b0fc8ab45f39403dd

echo "Building and deploying ross-website"
set -e # Exit with nonzero exit code if anything fails

echo "Python version: $TRAVIS_PYTHON_VERSION"
if [ $TRAVIS_PYTHON_VERSION != '3.7' ]; then
    echo "Skipping documentation deployment. This is done only on the 3.7 build"
    exit 0
fi

cd $HOME
# clone with ssh
git clone https://github.com/ross-rotordynamics/ross-website.git ross-website/html

# Delete all existing contents except .git and deploy_key.enc (we will re-create them)
echo "Removing existing content"
cd $HOME/ross-website/html
find -maxdepth 1 ! -name .git ! -name .gitignore ! -name deploy_key.enc ! -name . | xargs rm -rf

cd $HOME/build/ross-rotordynamics/ross/docs
echo "Building html files"
make html BUILDDIR=$HOME/ross-website

cd $HOME/ross-website/html
git config user.name "Travis CI"
git config user.email "raphaelts@gmail.com"

# If there are no changes (e.g. this is a README update) then just bail.
echo "Checking diff"
if [ -z `git diff --exit-code` ]; then
    echo "No changes to the spec on this push; exiting."
    exit 0
fi

echo "Commiting changes"
git add .
git commit -m "Docs deployed from Travis CI - build: $TRAVIS_BUILD_NUMBER"

echo "Getting keys on $PWD"
echo "key: $encrypted_b7aa2d550089_key"
echo "iv: $encrypted_b7aa2d550089_iv"

openssl aes-256-cbc -K $encrypted_b7aa2d550089_key -iv $encrypted_b7aa2d550089_iv -in deploy_key.enc -out deploy_key -d
chmod 600 deploy_key
eval `ssh-agent -s`
ssh-add deploy_key

echo "Pushing to repository"
git push git@github.com:ross-rotordynamics/ross.git master
