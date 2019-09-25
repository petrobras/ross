#!/bin/bash
# Deploy docs to github using Travis CI - based on this gist https://gist.github.com/domenic/ec8b0fc8ab45f39403dd

echo "Building and deploying ross-website"
set -e # Exit with nonzero exit code if anything fails

cd $HOME
# clone with ssh
git clone git@github.com:ross-rotordynamics/ross-website.git ross-website/html

# Delete all existing contents except .git and deploy_key.enc (we will re-create them)
cd $HOME/ross-website/html
find -maxdepth 1 ! -name .git ! -name deploy_key.enc ! -name . | xargs rm -rf

cd $HOME/ross-rotordynamics/ross/docs
make html BUILDDIR=$HOME/ross-website

cd $HOME/ross-website/html
git config user.name "Travis CI"
git config user.email "raphaelts@gmail.com"

# If there are no changes (e.g. this is a README update) then just bail.
if [ -z `git diff --exit-code` ]; then
    echo "No changes to the spec on this push; exiting."
    exit 0
fi

git add .
git commit -m "Docs deployed from Travis CI - build: $TRAVIS_BUILD_NUMBER"

openssl aes-256-cbc -K $encrypted_b7aa2d550089_key -iv $encrypted_b7aa2d550089_iv -in deploy_key.enc -out deploy_key -d
chmod 600 deploy_key
eval `ssh-agent -s`
ssh-add deploy_key

git push origin master
