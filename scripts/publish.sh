#!/bin/bash

# Path to your public repo folder
name="result2conclusion"
PUBLIC_REPO_PATH="../${name}"

# 1. Clear the public repo (except for the .git folder!)
# This ensures that if you delete a file in private, it vanishes in public.
find $PUBLIC_REPO_PATH -maxdepth 1 ! -name '.git' ! -name '.gitignore' ! -name $name -exec rm -rf {} +

# 2. Sync files using the exclusion list
rsync -av . "$PUBLIC_REPO_PATH" \
    --exclude-from='.public-ignore' \
    --exclude='.git/'

# 3. Commit and Push the public repo
cd "$PUBLIC_REPO_PATH"
# git add .
git commit -m "Sync from private repo: $(date)"
git push origin main